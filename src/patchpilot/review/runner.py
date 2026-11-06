"""End-to-end PR review runner for local `--repo-dir` and remote `--repo` + `--pr` modes."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from patchpilot.config import LangfuseSettings, Settings
from patchpilot.pipeline.graph import run_pipeline
from patchpilot.pipeline.models import Finding, PipelineState
from patchpilot.pipeline.nodes import VerifyCallable
from patchpilot.providers import get_provider
from patchpilot.providers.base import DataClass, LLMProvider
from patchpilot.providers.mock import MockProvider
from patchpilot.retrieval import HashEmbedder, OllamaEmbedder, QdrantStore
from patchpilot.review.formatter import format_review_comments
from patchpilot.review.github import GitHubClient, GitHubError, PostedReviewComment
from patchpilot.sandbox.runner import SandboxResult, SandboxRunner
from patchpilot.sandbox.tar_and_diff import JUnitReport, PatchApplyError, apply_unified_diff
from patchpilot.tracing import PatchPilotTracer


class ReviewResult(BaseModel):
    """Structured result of a `patchpilot review` execution."""

    source: str
    provider: str
    dry_run: bool
    findings_count: int
    verified_patches_count: int
    comments_posted_count: int
    findings: list[Finding] = Field(default_factory=list)
    verified_patch: str = ""
    comments: list[PostedReviewComment] = Field(default_factory=list)

    def summary_line(self) -> str:
        """Format the exact single-line CLI summary required by PLAN.md Phase 7."""
        base = (
            f"findings={self.findings_count} "
            f"verified_patches={self.verified_patches_count} "
            f"comments_posted={self.comments_posted_count}"
        )
        if self.dry_run:
            return f"{base} (dry-run)"
        return base


def _load_local_repo_dir(repo_dir: Path) -> dict[str, str]:
    """Load repository files from `repo_dir/repo/` (if case-style) or directly from `repo_dir`."""
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"Repository directory not found: {repo_dir}")

    target_root = repo_dir / "repo" if (repo_dir / "repo").is_dir() else repo_dir
    files: dict[str, str] = {}

    for p in sorted(target_root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(target_root).as_posix()
        if rel in ("patch.diff", "meta.json", "__symlinks__.json"):
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in p.parts):
            continue
        try:
            files[rel] = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

    return files


def _load_local_mock_patch(repo_dir: Path) -> str:
    """Load `patch.diff` or `meta.json` exclusively for the `MockProvider` path."""
    patch_file = repo_dir / "patch.diff"
    if patch_file.is_file():
        return patch_file.read_text(encoding="utf-8")
    meta_file = repo_dir / "meta.json"
    if meta_file.is_file():
        try:
            raw = json.loads(meta_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("patch_diff"), str):
                return str(raw["patch_diff"])
        except (ValueError, OSError):
            pass
    return ""


def _resolve_verify_mechanism(
    provider_name: str,
    sandbox_runner: SandboxRunner | None,
    verify_fn: VerifyCallable | None,
) -> tuple[SandboxRunner | None, VerifyCallable | None]:
    """Return a sandbox runner or a fast in-memory diff verifier for CI/mock environments."""
    if sandbox_runner is not None or verify_fn is not None:
        return (sandbox_runner, verify_fn)

    image_ready = False
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", "patchpilot-sandbox:0.0.0"],
            capture_output=True,
            timeout=3.0,
            check=False,
        )
        image_ready = proc.returncode == 0
    except (subprocess.SubprocessError, OSError):
        image_ready = False

    if image_ready:
        return (SandboxRunner(), None)

    if provider_name == "mock":

        async def _plumbing_mock_verify(fs: dict[str, str], patch: str) -> SandboxResult:
            if not patch.strip():
                return SandboxResult(exit_code=1, patch_applied=False, error="empty patch")
            with tempfile.TemporaryDirectory() as tmp:
                wdir = Path(tmp)
                for rel_p, content in fs.items():
                    dest = wdir / rel_p
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                try:
                    apply_unified_diff(patch, wdir)
                except PatchApplyError as exc:
                    return SandboxResult(exit_code=1, patch_applied=False, error=str(exc))
            return SandboxResult(
                exit_code=0,
                patch_applied=True,
                junit=JUnitReport(passed=1, failures=0, errors=0),
            )

        return (None, _plumbing_mock_verify)

    return (SandboxRunner(), None)


async def run_review(
    *,
    repo_dir: Path | str | None = None,
    repo: str | None = None,
    pr_number: int | None = None,
    dry_run: bool = False,
    provider_name: str = "router",
    provider_override: LLMProvider | None = None,
    github_client: GitHubClient | None = None,
    sandbox_runner: SandboxRunner | None = None,
    verify_fn: VerifyCallable | None = None,
    tracer: PatchPilotTracer | None = None,
    report_out: Path | None = None,
) -> ReviewResult:
    """Execute PatchPilot PR review in local (`--repo-dir`) or remote (`--repo` + `--pr`) mode."""
    # Allow `--repo <local_path>` if `--repo-dir` was omitted and `--pr` is not provided
    if repo_dir is None and repo is not None and pr_number is None:
        candidate_dir = Path(repo)
        if candidate_dir.is_dir():
            repo_dir = candidate_dir
            repo = None

    if repo_dir is None and (not repo or pr_number is None):
        raise ValueError(
            "review requires either --repo-dir <path> (local mode) "
            "or --repo <owner/repo> --pr <number> (remote GitHub mode)"
        )

    gh = github_client or GitHubClient()
    if repo is not None and pr_number is not None and not dry_run:
        # Fail fast before running LLM/sandbox if GITHUB_TOKEN is missing for live GitHub review
        gh.require_token_for_live_write()

    settings = Settings()
    files: dict[str, str] = {}
    failing_context = ""
    data_class: DataClass = "private"
    source_label = ""
    head_sha = ""
    known_mock_patch = ""

    if repo_dir is not None:
        local_path = Path(repo_dir)
        files = _load_local_repo_dir(local_path)
        known_mock_patch = _load_local_mock_patch(local_path)
        source_label = local_path.as_posix()
        data_class = (
            "public" if "fixtures" in source_label or "evals" in source_label else "private"
        )
    else:
        assert repo is not None and pr_number is not None
        pr_ctx = await gh.fetch_pull_request(repo, pr_number)
        files = pr_ctx.files
        failing_context = pr_ctx.diff
        head_sha = pr_ctx.head_sha
        source_label = f"{pr_ctx.repo}#{pr_ctx.pr_number}"
        data_class = "private" if pr_ctx.is_private else "public"

    # Set up LLM provider
    provider: LLMProvider
    if provider_override is not None:
        provider = provider_override
    elif provider_name == "mock":
        provider = MockProvider(canned=known_mock_patch or "MOCK_RESPONSE")
    else:
        provider = get_provider(provider_name, settings)  # type: ignore[arg-type]

    eff_runner: SandboxRunner | None = sandbox_runner
    eff_verify: VerifyCallable | None = verify_fn
    if eff_runner is None and eff_verify is None:
        eff_runner, eff_verify = _resolve_verify_mechanism(provider_name, sandbox_runner, verify_fn)
    embedder = (
        OllamaEmbedder(base_url=settings.ollama_base_url, model=settings.ollama_model)
        if provider_name in ("ollama", "router") and provider_override is None
        else HashEmbedder(dim=64)
    )
    qdrant = QdrantStore(base_url="http://127.0.0.1:6333", timeout_s=1.0)
    active_tracer = tracer or PatchPilotTracer(
        settings=LangfuseSettings(host="", public_key="", secret_key=""),
        enabled=False,
    )

    initial_state = PipelineState(
        case_id=f"review:{source_label}",
        files=files,
        failing_test_output=failing_context,
        data_class=data_class,
        provider=provider_name,
    )

    final_state = await run_pipeline(
        initial_state,
        provider=provider,
        sandbox_runner=eff_runner,
        verify_fn=eff_verify,
        qdrant=qdrant,
        embedder=embedder,
        tracer=active_tracer,
    )

    findings = list(final_state.findings)
    verified_patch = final_state.reported_patch if final_state.verified else ""
    verified_patches_count = 1 if (final_state.verified and verified_patch.strip()) else 0

    comments = format_review_comments(findings, verified_patch)

    comments_posted_count = 0
    if not dry_run and repo is not None and pr_number is not None and comments:
        comments_posted_count = await gh.post_review_comments(
            repo,
            pr_number,
            comments,
            commit_id=head_sha,
        )

    result = ReviewResult(
        source=source_label,
        provider=provider_name,
        dry_run=dry_run,
        findings_count=len(findings),
        verified_patches_count=verified_patches_count,
        comments_posted_count=comments_posted_count,
        findings=findings,
        verified_patch=verified_patch,
        comments=comments,
    )

    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    return result


__all__ = ["GitHubError", "ReviewResult", "run_review"]
