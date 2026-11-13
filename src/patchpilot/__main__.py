"""CLI entrypoint for PatchPilot: python -m patchpilot run | trace-smoke ..."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from patchpilot.config import PROJECT_ROOT, Settings
from patchpilot.pipeline.graph import run_pipeline
from patchpilot.pipeline.models import PipelineRunSummary, PipelineState
from patchpilot.providers import get_provider
from patchpilot.providers.base import LLMProvider, ProviderError
from patchpilot.providers.mock import MockProvider
from patchpilot.retrieval import Embedder, HashEmbedder, OllamaEmbedder, QdrantStore
from patchpilot.sandbox.runner import SandboxError, SandboxRunner
from patchpilot.tracing import run_trace_smoke

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("patchpilot.cli")


def _resolve_case_dir(case_arg: str) -> Path:
    """Resolve a case argument (e.g. 'seeded_eval/0001' or '0001') to its directory."""
    cleaned = case_arg.strip()
    if not cleaned or cleaned in (".", ".."):
        raise FileNotFoundError(f"Case directory not found for {case_arg!r}")

    raw = Path(cleaned)
    if raw.is_dir():
        return raw

    candidates = (
        Path("evals/datasets") / cleaned,
        Path("evals/datasets/seeded_eval") / cleaned,
        PROJECT_ROOT / "evals/datasets" / cleaned,
        PROJECT_ROOT / "evals/datasets/seeded_eval" / cleaned,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Case directory not found for {case_arg!r}")


def _load_case_files(case_dir: Path) -> dict[str, str]:
    """Load only the repository files from a case directory (never patch.diff or meta.json)."""
    repo_dir = case_dir / "repo"
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"repo directory not found in {case_dir}")

    files: dict[str, str] = {}
    for p in sorted(repo_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(repo_dir)).replace("\\", "/")
            parts = rel.split("/")
            if (
                "__pycache__" in parts
                or ".pytest_cache" in parts
                or ".git" in parts
                or rel.endswith(".pyc")
            ):
                continue
            try:
                files[rel] = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

    return files


def load_mock_ground_truth(case_dir: Path) -> str:
    """Load ground-truth patch exclusively for the MockProvider path."""
    patch_path = case_dir / "patch.diff"
    if patch_path.is_file():
        try:
            return patch_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    meta_path = case_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and meta.get("patch_diff") is not None:
                return str(meta["patch_diff"])
        except (ValueError, OSError):
            pass

    return ""


async def _run_command(case_arg: str, provider_name: str) -> int:
    try:
        settings = Settings()
        case_dir = _resolve_case_dir(case_arg)
        files = _load_case_files(case_dir)

        # Provider setup: only MockProvider reads ground-truth patch.diff / meta.json
        provider: LLMProvider
        if provider_name == "mock":
            known_patch = load_mock_ground_truth(case_dir)
            provider = MockProvider(canned=known_patch)
        else:
            provider = get_provider(provider_name, settings)  # type: ignore[arg-type]

        # Embedder setup: HashEmbedder for mock/CI, OllamaEmbedder for live providers
        embedder: Embedder = (
            HashEmbedder(dim=64)
            if provider_name == "mock"
            else OllamaEmbedder(base_url=settings.ollama_base_url, model=settings.ollama_model)
        )

        qdrant = QdrantStore(base_url="http://127.0.0.1:6333", timeout_s=1.0)

        initial_state = PipelineState(
            case_id=case_arg,
            files=files,
            data_class="public",
            provider=provider_name,
        )

        final_state = await run_pipeline(
            initial_state,
            provider=provider,
            sandbox_runner=SandboxRunner(),
            qdrant=qdrant,
            embedder=embedder,
        )
    except (FileNotFoundError, ValueError, ProviderError, SandboxError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = PipelineRunSummary(
        case=case_arg,
        provider=provider_name,
        patch_applies=final_state.patch_applies,
        target_test_passes=final_state.target_test_passes,
        regressions=final_state.regressions,
        verified=final_state.verified,
    )
    print(summary.model_dump_json())
    return 0


async def _trace_smoke_command(provider_name: str = "mock") -> int:
    _, msg = await run_trace_smoke(provider_name=provider_name)
    print(msg)
    return 0


async def _review_command(
    repo_dir: str | None,
    repo: str | None,
    pr_number: int | None,
    dry_run: bool,
    provider_name: str,
    report_out: str | None,
    assume_public: bool = False,
) -> int:
    from patchpilot.review.github import GitHubError
    from patchpilot.review.runner import run_review

    try:
        res = await run_review(
            repo_dir=Path(repo_dir) if repo_dir else None,
            repo=repo,
            pr_number=pr_number,
            dry_run=dry_run,
            provider_name=provider_name,
            report_out=Path(report_out) if report_out else None,
            assume_public=assume_public,
        )
    except (GitHubError, ProviderError, SandboxError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(res.summary_line())
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="patchpilot", description="PatchPilot CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_parser = subparsers.add_parser("run", help="Run the LangGraph repair pipeline on a case")
    run_parser.add_argument("--case", required=True, help="Case identifier (e.g. seeded_eval/0001)")
    run_parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "ollama", "gemini", "openrouter", "router"],
        help="LLM provider name (default: mock)",
    )

    trace_smoke_parser = subparsers.add_parser(
        "trace-smoke",
        help="Verify Langfuse trace ingestion/visibility or degrade to no-op when down",
    )
    trace_smoke_parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "ollama", "gemini", "openrouter", "router"],
        help="LLM provider name (default: mock)",
    )

    review_parser = subparsers.add_parser(
        "review",
        help="Review a local repository directory or a GitHub Pull Request",
    )
    review_parser.add_argument(
        "--repo-dir",
        default=None,
        help="Local repository directory to review (offline / CI mode)",
    )
    review_parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repository ('owner/repo' or URL) or local path",
    )
    review_parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="GitHub Pull Request number (used with --repo)",
    )
    review_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify patches and format comments without posting to GitHub",
    )
    review_parser.add_argument(
        "--provider",
        default="router",
        choices=["mock", "ollama", "gemini", "openrouter", "router"],
        help="LLM provider name (default: router)",
    )
    review_parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write full JSON review report",
    )

    review_parser.add_argument(
        "--assume-public",
        action="store_true",
        help="Declare a local --repo-dir as public data (default: private, local models only)",
    )

    args = parser.parse_args(argv)

    if args.subcommand == "run":
        exit_code = asyncio.run(_run_command(args.case, args.provider))
        sys.exit(exit_code)
    elif args.subcommand == "trace-smoke":
        exit_code = asyncio.run(_trace_smoke_command(args.provider))
        sys.exit(exit_code)
    elif args.subcommand == "review":
        exit_code = asyncio.run(
            _review_command(
                repo_dir=args.repo_dir,
                repo=args.repo,
                pr_number=args.pr,
                dry_run=bool(args.dry_run),
                provider_name=args.provider,
                report_out=args.out,
                assume_public=bool(args.assume_public),
            )
        )
        sys.exit(exit_code)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
