"""Phase 5 evaluation runner with resumable checkpoints, Wilson CIs, and Langfuse datasets."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import subprocess
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from patchpilot.__main__ import _load_case_files, load_mock_ground_truth
from patchpilot.config import LangfuseSettings, Settings
from patchpilot.evals.freeze import verify_frozen_eval_split
from patchpilot.evals.realworld_mini import (
    DEFAULT_FROZEN_REALWORLD_MINI,
    REALWORLD_MINI_POOL,
    SPEC_BY_CASE_ID,
    verify_frozen_realworld_mini,
)
from patchpilot.pipeline.graph import run_pipeline
from patchpilot.pipeline.models import PipelineState
from patchpilot.pipeline.nodes import VerifyCallable, compute_prompt_version_hash
from patchpilot.providers import get_provider
from patchpilot.providers.base import LLMProvider
from patchpilot.providers.mock import MockProvider
from patchpilot.retrieval import Embedder, HashEmbedder, OllamaEmbedder, QdrantStore
from patchpilot.sandbox.runner import SandboxResult, SandboxRunner
from patchpilot.sandbox.tar_and_diff import JUnitReport, PatchApplyError, apply_unified_diff
from patchpilot.tracing import LLMCallMetric, PatchPilotTracer

logger = logging.getLogger("patchpilot.evals.runner")


class EvalIntegrityError(RuntimeError):
    """Raised when eval integrity checks fail (frozen split or prompt-version hash mismatch)."""


class EvalCaseRunResult(BaseModel):
    """Per-case, per-run evaluation telemetry and verification outcome."""

    case_id: str
    pool: str
    taxonomy: str = "unknown"
    template: str = "unknown"
    run_index: int
    seed: int
    provider: str
    model_name: str
    prompt_version_hash: str
    attempts: int
    verified: bool
    pass_at_1: bool
    pass_at_3: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    max_call_prompt_tokens: int = 0
    context_window_tokens: int = 32768
    context_overflow: bool = False
    borderline: bool = False
    failure_cause: str = ""
    llm_calls: list[LLMCallMetric] = Field(default_factory=list)
    local_calls: int = 0
    frontier_calls: int = 0
    fallback_count: int = 0
    warning_flags: list[str] = Field(default_factory=list)
    total_time_s: float = 0.0
    trace_id: str = ""
    resumed: bool = False


class BreakdownRow(BaseModel):
    """Aggregated pass@1 and pass@3 metrics with 95% Wilson CIs for a slice (class or template)."""

    key: str
    total: int
    pass_at_1_count: int
    pass_at_1_rate: float
    pass_at_1_ci_95: tuple[float, float]
    pass_at_3_count: int
    pass_at_3_rate: float
    pass_at_3_ci_95: tuple[float, float]


class EvalRunReport(BaseModel):
    """Single-run JSON report written to `evals/results/`."""

    run_id: str
    run_index: int
    timestamp: str
    pool: str
    provider: str
    model_name: str
    seed: int
    prompt_version_hash: str
    total_cases: int
    pass_at_1_count: int
    pass_at_1_rate: float
    pass_at_1_ci_95: tuple[float, float]
    pass_at_3_count: int
    pass_at_3_rate: float
    pass_at_3_ci_95: tuple[float, float]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_time_s: float
    local_calls: int = 0
    frontier_calls: int = 0
    fallback_count: int = 0
    warning_flags: list[str] = Field(default_factory=list)
    langfuse_dataset_synced: bool = False
    output_file: str = ""
    cases: list[EvalCaseRunResult] = Field(default_factory=list)


class EvalSummaryReport(BaseModel):
    """Multi-run summary report with Wilson CIs and per-class/per-template breakdowns."""

    run_id: str
    pool: str
    provider: str
    model_name: str
    seed: int
    n_runs: int
    prompt_version_hash: str
    total_observations: int
    pass_at_1_count: int
    pass_at_1_rate: float
    pass_at_1_ci_95: tuple[float, float]
    pass_at_3_count: int
    pass_at_3_rate: float
    pass_at_3_ci_95: tuple[float, float]
    local_calls: int = 0
    frontier_calls: int = 0
    fallback_count: int = 0
    warning_flags: list[str] = Field(default_factory=list)
    by_taxonomy: list[BreakdownRow] = Field(default_factory=list)
    by_template: list[BreakdownRow] = Field(default_factory=list)
    run_files: list[str] = Field(default_factory=list)
    cases: list[EvalCaseRunResult] = Field(default_factory=list)


def wilson_ci_95(successes: int, total: int) -> tuple[float, float]:
    """Compute the 95% Wilson score confidence interval (low, high) for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    z2 = z * z
    n = float(total)
    p = max(0.0, min(1.0, float(successes) / n))
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    radicand = max(0.0, (p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    half = (z * math.sqrt(radicand)) / denom
    low = round(max(0.0, center - half), 6)
    high = round(min(1.0, center + half), 6)
    return (low, high)


def _compute_config_hash(
    *,
    pool: str,
    provider: str,
    model_name: str,
    seed: int,
    prompt_version_hash: str,
    limit: int | None = None,
) -> str:
    raw = f"{pool}:{provider}:{model_name}:{seed}:{prompt_version_hash}"
    if limit is not None:
        raw = f"{raw}:limit={limit}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _resolve_default_model_name(
    provider_name: str,
    settings: Settings,
    provider_override: LLMProvider | None = None,
) -> str:
    """Resolve the default model identifier used for config hashing and fallback reporting."""
    if provider_override is not None:
        return str(getattr(provider_override, "default_model", provider_override.name))
    if provider_name == "ollama":
        return settings.ollama_model
    if provider_name == "gemini":
        return settings.gemini_model or "gemini"
    if provider_name == "openrouter":
        return settings.openrouter_model or "openrouter"
    if provider_name == "router":
        frontier_model = (
            settings.gemini_model or "gemini"
            if settings.frontier_provider == "gemini"
            else settings.openrouter_model or "openrouter"
        )
        return f"router({settings.ollama_model}+{frontier_model})"
    return "mock-1"


def _build_effective_run_id(
    *,
    pool: str,
    provider_name: str,
    seed: int,
    config_hash: str,
    limit: int | None = None,
    run_id: str | None = None,
) -> str:
    """Construct the run identifier, preserving full-run IDs and isolating --limit runs."""
    limit_suffix = f"_limit{limit}" if limit is not None else ""
    if run_id:
        if limit_suffix and not run_id.endswith(limit_suffix):
            return f"{run_id}{limit_suffix}"
        return run_id
    return f"{pool}_{provider_name}_s{seed}_{config_hash[:8]}{limit_suffix}"


def _checkpoint_path(
    results_dir: Path,
    config_hash: str,
    run_index: int,
    case_id: str,
) -> Path:
    safe_case = case_id.replace("/", "_").replace("\\", "_")
    return results_dir / "checkpoints" / config_hash / f"run_{run_index:02d}_case_{safe_case}.json"


def _load_case_labels_after_run(case_dir: Path) -> tuple[str, str]:
    """Read only taxonomy and template labels from meta.json AFTER the pipeline finishes."""
    meta_path = case_dir / "meta.json"
    if not meta_path.is_file():
        return ("unknown", "unknown")
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return (
                str(raw.get("taxonomy", "unknown")),
                str(raw.get("template", "unknown")),
            )
    except (ValueError, OSError):
        pass
    return ("unknown", "unknown")


def _compute_breakdown(
    cases: list[EvalCaseRunResult],
    key_fn: Callable[[EvalCaseRunResult], str],
) -> list[BreakdownRow]:
    buckets: dict[str, list[EvalCaseRunResult]] = {}
    for c in cases:
        k = key_fn(c)
        buckets.setdefault(k, []).append(c)

    rows: list[BreakdownRow] = []
    for k in sorted(buckets):
        items = buckets[k]
        total = len(items)
        p1 = sum(1 for x in items if x.pass_at_1)
        p3 = sum(1 for x in items if x.pass_at_3)
        rows.append(
            BreakdownRow(
                key=k,
                total=total,
                pass_at_1_count=p1,
                pass_at_1_rate=round(p1 / total, 6) if total else 0.0,
                pass_at_1_ci_95=wilson_ci_95(p1, total),
                pass_at_3_count=p3,
                pass_at_3_rate=round(p3 / total, 6) if total else 0.0,
                pass_at_3_ci_95=wilson_ci_95(p3, total),
            )
        )
    return rows


def _select_embedder(provider_name: str, settings: Settings) -> Embedder:
    """Select `HashEmbedder` only for mock mode and `OllamaEmbedder` for live providers."""
    if provider_name == "mock":
        return HashEmbedder(dim=64)
    return OllamaEmbedder(base_url=settings.ollama_base_url, model=settings.ollama_model)


def _extract_call_telemetry(
    llm_calls: list[LLMCallMetric],
) -> tuple[int, int, int, list[str]]:
    """Aggregate local_calls, frontier_calls, fallback_count, and unique warning_flags."""
    local_calls = 0
    frontier_calls = 0
    fallback_count = 0
    warning_flags: list[str] = []
    seen_warnings: set[str] = set()

    for call in llm_calls:
        if call.route_target == "local":
            local_calls += 1
        elif call.route_target == "frontier":
            frontier_calls += 1
        elif call.provider == "ollama":
            local_calls += 1
        elif call.provider in ("gemini", "openrouter"):
            frontier_calls += 1

        if call.fallback_used:
            fallback_count += 1
        if call.warning_flag and call.warning_flag not in seen_warnings:
            seen_warnings.add(call.warning_flag)
            warning_flags.append(call.warning_flag)

    return local_calls, frontier_calls, fallback_count, warning_flags


def _aggregate_case_telemetry(
    cases: list[EvalCaseRunResult],
) -> tuple[int, int, int, list[str]]:
    """Aggregate routing and fallback telemetry across `EvalCaseRunResult` records."""
    local_calls = 0
    frontier_calls = 0
    fallback_count = 0
    warning_flags: list[str] = []
    seen_warnings: set[str] = set()

    for case in cases:
        if case.llm_calls:
            c_local, c_frontier, c_fallbacks, c_warnings = _extract_call_telemetry(case.llm_calls)
        else:
            c_local = case.local_calls
            c_frontier = case.frontier_calls
            c_fallbacks = case.fallback_count
            c_warnings = case.warning_flags

        local_calls += c_local
        frontier_calls += c_frontier
        fallback_count += c_fallbacks
        for flag in c_warnings:
            if flag and flag not in seen_warnings:
                seen_warnings.add(flag)
                warning_flags.append(flag)

    return local_calls, frontier_calls, fallback_count, warning_flags


def build_eval_summary(run_reports: list[EvalRunReport]) -> EvalSummaryReport:
    """Validate prompt-version hash consistency across runs and build the aggregated summary."""
    if not run_reports:
        raise ValueError("Cannot build summary from empty run_reports list")

    hashes: set[str] = set()
    for rep in run_reports:
        hashes.add(rep.prompt_version_hash)
        for c in rep.cases:
            hashes.add(c.prompt_version_hash)

    if len(hashes) > 1:
        raise EvalIntegrityError(
            "Refusing to publish summary: prompt_version_hash changed between runs of the same "
            f"comparison ({sorted(hashes)})"
        )

    prompt_hash = next(iter(hashes))
    all_cases = [c for rep in run_reports for c in rep.cases]
    total_obs = len(all_cases)
    p1 = sum(1 for c in all_cases if c.pass_at_1)
    p3 = sum(1 for c in all_cases if c.pass_at_3)

    total_local = 0
    total_frontier = 0
    total_fallbacks = 0
    summary_warnings: list[str] = []
    seen_warnings: set[str] = set()

    for rep in run_reports:
        c_local, c_frontier, c_fallbacks, c_warnings = _aggregate_case_telemetry(rep.cases)
        if c_local == 0 and c_frontier == 0 and c_fallbacks == 0 and not c_warnings:
            c_local = rep.local_calls
            c_frontier = rep.frontier_calls
            c_fallbacks = rep.fallback_count
            c_warnings = rep.warning_flags

        total_local += c_local
        total_frontier += c_frontier
        total_fallbacks += c_fallbacks
        for flag in c_warnings:
            if flag and flag not in seen_warnings:
                seen_warnings.add(flag)
                summary_warnings.append(flag)

    first = run_reports[0]
    return EvalSummaryReport(
        run_id=first.run_id,
        pool=first.pool,
        provider=first.provider,
        model_name=first.model_name,
        seed=first.seed,
        n_runs=len(run_reports),
        prompt_version_hash=prompt_hash,
        total_observations=total_obs,
        pass_at_1_count=p1,
        pass_at_1_rate=round(p1 / total_obs, 6) if total_obs else 0.0,
        pass_at_1_ci_95=wilson_ci_95(p1, total_obs),
        pass_at_3_count=p3,
        pass_at_3_rate=round(p3 / total_obs, 6) if total_obs else 0.0,
        pass_at_3_ci_95=wilson_ci_95(p3, total_obs),
        local_calls=total_local,
        frontier_calls=total_frontier,
        fallback_count=total_fallbacks,
        warning_flags=summary_warnings,
        by_taxonomy=_compute_breakdown(all_cases, lambda c: c.taxonomy),
        by_template=_compute_breakdown(all_cases, lambda c: c.template),
        run_files=[rep.output_file for rep in run_reports if rep.output_file],
        cases=all_cases,
    )


def format_eval_summary(summary: EvalSummaryReport) -> str:
    """Format the evaluation summary with 95% Wilson CIs and class/template breakdowns."""
    p1_lo, p1_hi = summary.pass_at_1_ci_95
    p3_lo, p3_hi = summary.pass_at_3_ci_95
    lines = [
        f"=== PatchPilot Eval Summary ({summary.run_id}) ===",
        (
            f"Pool: {summary.pool} | Provider: {summary.provider} "
            f"| Model: {summary.model_name} | Runs: {summary.n_runs} | Seed: {summary.seed}"
        ),
        f"Prompt-version hash: {summary.prompt_version_hash}",
        (
            f"Overall pass@1: {summary.pass_at_1_count}/{summary.total_observations} "
            f"({summary.pass_at_1_rate:.1%}) "
            f"[95% Wilson CI: {p1_lo:.1%}, {p1_hi:.1%}]"
        ),
        (
            f"Overall pass@3: {summary.pass_at_3_count}/{summary.total_observations} "
            f"({summary.pass_at_3_rate:.1%}) "
            f"[95% Wilson CI: {p3_lo:.1%}, {p3_hi:.1%}]"
        ),
        (
            f"Routing telemetry: local_calls={summary.local_calls} | "
            f"frontier_calls={summary.frontier_calls} | "
            f"fallback_count={summary.fallback_count}"
        ),
    ]
    if summary.warning_flags:
        lines.append(f"Warning flags ({len(summary.warning_flags)}):")
        for flag in summary.warning_flags[:10]:
            lines.append(f"  - {flag}")
        if len(summary.warning_flags) > 10:
            lines.append(f"  - ... (+{len(summary.warning_flags) - 10} more)")
    if summary.pool == REALWORLD_MINI_POOL or summary.pool.startswith(REALWORLD_MINI_POOL):
        overflow_cnt = sum(1 for c in summary.cases if c.context_overflow)
        lines.append(
            f"NOTE: {REALWORLD_MINI_POOL}, n={summary.total_observations}, small sample "
            f"(context_overflow={overflow_cnt})."
        )
    elif summary.pool.startswith("seeded"):
        lines.append(
            "NOTE: The eval pool (seeded_eval) has only 14 templates "
            "(2 per bug class across 7 classes; 42 templates are reserved for seeded_train)."
        )
    lines.append("\nBreakdown by bug class (taxonomy):")
    for row in summary.by_taxonomy:
        r1_lo, r1_hi = row.pass_at_1_ci_95
        r3_lo, r3_hi = row.pass_at_3_ci_95
        lines.append(
            f"  - {row.key:<18} n={row.total:<3} "
            f"pass@1={row.pass_at_1_count}/{row.total} ({row.pass_at_1_rate:.1%}) "
            f"[{r1_lo:.1%}, {r1_hi:.1%}] | "
            f"pass@3={row.pass_at_3_count}/{row.total} ({row.pass_at_3_rate:.1%}) "
            f"[{r3_lo:.1%}, {r3_hi:.1%}]"
        )

    lines.append("\nBreakdown by template:")
    for row in summary.by_template:
        r1_lo, r1_hi = row.pass_at_1_ci_95
        r3_lo, r3_hi = row.pass_at_3_ci_95
        lines.append(
            f"  - {row.key:<28} n={row.total:<3} "
            f"pass@1={row.pass_at_1_count}/{row.total} ({row.pass_at_1_rate:.1%}) "
            f"[{r1_lo:.1%}, {r1_hi:.1%}] | "
            f"pass@3={row.pass_at_3_count}/{row.total} ({row.pass_at_3_rate:.1%}) "
            f"[{r3_lo:.1%}, {r3_hi:.1%}]"
        )

    if (
        summary.pool == REALWORLD_MINI_POOL or summary.pool.startswith(REALWORLD_MINI_POOL)
    ) and summary.cases:
        lines.append("\nPer-case outcomes (realworld_mini):")
        for c in summary.cases:
            tag = " [borderline]" if c.borderline else ""
            overflow_tag = " [context overflow]" if c.context_overflow else ""
            status = "PASS" if c.verified else "FAIL"
            cause = f" | cause: {c.failure_cause}" if not c.verified and c.failure_cause else ""
            lines.append(
                f"  - {c.case_id}{tag}{overflow_tag} ({c.template}): {status} "
                f"(pass@1={c.pass_at_1}, pass@3={c.pass_at_3}, attempts={c.attempts}, "
                f"prompt_tokens={c.prompt_tokens}, "
                f"max_call_prompt_tokens={c.max_call_prompt_tokens}/"
                f"{c.context_window_tokens}, time={c.total_time_s:.1f}s){cause}"
            )

    if summary.run_files:
        lines.append("\nWritten run JSON files:")
        for rf in summary.run_files:
            lines.append(f"  - {rf}")

    return "\n".join(lines)


async def run_eval_pool(
    *,
    pool: str,
    provider_name: str,
    runs: int = 1,
    seed: int = 1337,
    limit: int | None = None,
    datasets_dir: Path = Path("evals/datasets"),
    results_dir: Path = Path("evals/results"),
    frozen_path: Path = Path("evals/FROZEN.json"),
    run_id: str | None = None,
    provider_override: LLMProvider | None = None,
    verify_fn: VerifyCallable | None = None,
    sandbox_runner: SandboxRunner | None = None,
    langfuse_settings: LangfuseSettings | None = None,
    langfuse_transport: object | None = None,
    before_case_hook: Callable[[str, int], None] | None = None,
) -> tuple[list[EvalRunReport], EvalSummaryReport]:
    """Execute `runs` evaluation passes over `pool` with checkpoints and fail-closed integrity."""
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")

    # Requirement 4: Fail closed — verify-frozen MUST pass before any seeded_eval run
    if pool == "seeded_eval" or pool.startswith("seeded_eval"):
        valid, msg = verify_frozen_eval_split(
            frozen_path=frozen_path,
            datasets_dir=datasets_dir,
        )
        if not valid:
            raise EvalIntegrityError(
                f"Fail-closed eval integrity check: verify-frozen failed before {pool} run: {msg}"
            )
    elif pool == REALWORLD_MINI_POOL or pool.startswith(REALWORLD_MINI_POOL):
        rw_frozen = (
            DEFAULT_FROZEN_REALWORLD_MINI
            if frozen_path == Path("evals/FROZEN.json")
            else frozen_path
        )
        valid, msg = verify_frozen_realworld_mini(
            frozen_path=rw_frozen,
            datasets_dir=datasets_dir,
        )
        if not valid:
            raise EvalIntegrityError(
                "Fail-closed eval integrity check: verify-frozen-realworld-mini failed "
                f"before {pool} run: {msg}"
            )

    pool_dir = datasets_dir / pool
    if not pool_dir.is_dir():
        raise FileNotFoundError(f"Evaluation pool directory not found: {pool_dir}")

    case_dirs = sorted([d for d in pool_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    if limit is not None and 0 < limit < len(case_dirs) and seed != 1337:
        rng = random.Random(seed)
        sampled = rng.sample(case_dirs, limit)
        case_dirs = sorted(sampled, key=lambda d: d.name)
    elif limit is not None and limit > 0:
        case_dirs = case_dirs[:limit]
    if not case_dirs:
        raise ValueError(f"No evaluation cases found in {pool_dir}")

    settings = Settings()
    prompt_hash = compute_prompt_version_hash()

    default_model_name = _resolve_default_model_name(
        provider_name,
        settings,
        provider_override=provider_override,
    )

    config_hash = _compute_config_hash(
        pool=pool,
        provider=provider_name,
        model_name=default_model_name,
        seed=seed,
        prompt_version_hash=prompt_hash,
        limit=limit,
    )
    effective_run_id = _build_effective_run_id(
        pool=pool,
        provider_name=provider_name,
        seed=seed,
        config_hash=config_hash,
        limit=limit,
        run_id=run_id,
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check if Langfuse observability is up for dataset run syncing (Requirement 5)
    lf_cfg = langfuse_settings or LangfuseSettings()
    probe_tracer = PatchPilotTracer(
        settings=lf_cfg,
        transport=langfuse_transport,  # type: ignore[arg-type]
    )
    obs_up = await probe_tracer.check_health()
    if obs_up:
        await probe_tracer.ensure_dataset(
            dataset_name=pool,
            description=f"PatchPilot {pool} dataset",
        )

    embedder = _select_embedder(provider_name, settings)
    qdrant = QdrantStore(base_url="http://127.0.0.1:6333", timeout_s=1.0)

    effective_verify_fn = verify_fn
    shared_runner: SandboxRunner | None = sandbox_runner
    if shared_runner is None and effective_verify_fn is None:
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

        if image_ready or provider_name != "mock":
            shared_runner = SandboxRunner()
        else:
            # CI plumbing fallback for mock provider when sandbox image is not built
            async def _plumbing_mock_verify(fs: dict[str, str], patch: str) -> SandboxResult:
                with tempfile.TemporaryDirectory() as tmp:
                    wdir = Path(tmp)
                    for rel_p, content in fs.items():
                        dest = wdir / rel_p
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_text(content, encoding="utf-8")
                    try:
                        apply_unified_diff(patch, wdir)
                    except PatchApplyError as exc:
                        return SandboxResult(
                            exit_code=1,
                            patch_applied=False,
                            error=str(exc),
                        )
                return SandboxResult(
                    exit_code=0,
                    patch_applied=True,
                    junit=JUnitReport(passed=1, failures=0, errors=0),
                )

            effective_verify_fn = _plumbing_mock_verify

    run_reports: list[EvalRunReport] = []

    for run_idx in range(1, runs + 1):
        run_cases: list[EvalCaseRunResult] = []
        run_start = time.perf_counter()

        for cdir in case_dirs:
            case_id = cdir.name
            ckpt_file = _checkpoint_path(results_dir, config_hash, run_idx, case_id)

            if ckpt_file.is_file():
                ckpt_res = EvalCaseRunResult.model_validate_json(
                    ckpt_file.read_text(encoding="utf-8")
                )
                if ckpt_res.prompt_version_hash != prompt_hash:
                    raise EvalIntegrityError(
                        f"Checkpoint prompt_version_hash mismatch for {case_id} "
                        f"({ckpt_res.prompt_version_hash} != {prompt_hash})"
                    )
                ckpt_res.resumed = True
                run_cases.append(ckpt_res)
                continue

            if before_case_hook is not None:
                before_case_hook(case_id, run_idx)

            # Ground-truth isolation: load only repo/ files for the pipeline
            files = _load_case_files(cdir)

            provider: LLMProvider
            if provider_override is not None:
                provider = provider_override
            elif provider_name == "mock":
                known_patch = load_mock_ground_truth(cdir)
                provider = MockProvider(canned=known_patch)
            else:
                provider = get_provider(provider_name, settings)  # type: ignore[arg-type]

            case_tracer = PatchPilotTracer(
                settings=lf_cfg,
                transport=langfuse_transport,  # type: ignore[arg-type]
            )
            initial_state = PipelineState(
                case_id=f"{pool}/{case_id}",
                files=files,
                data_class="public",
                provider=provider_name,
            )

            t_case = time.perf_counter()
            final_state = await run_pipeline(
                initial_state,
                provider=provider,
                sandbox_runner=shared_runner,
                verify_fn=effective_verify_fn,
                qdrant=qdrant,
                embedder=embedder,
                tracer=case_tracer,
            )
            case_elapsed = round(max(0.0, time.perf_counter() - t_case), 6)

            # Read taxonomy & template labels only AFTER the pipeline run completes
            taxonomy, template = _load_case_labels_after_run(cdir)
            max_call_ptok = max((c.prompt_tokens for c in final_state.llm_calls), default=0)
            ctx_window = 32768
            is_overflow = max_call_ptok >= ctx_window
            spec_obj = SPEC_BY_CASE_ID.get(case_id) if pool == REALWORLD_MINI_POOL else None
            is_borderline = bool(spec_obj.borderline) if spec_obj is not None else False
            fail_cause = ""
            if is_overflow:
                fail_cause = (
                    f"context overflow (max_call_prompt_tokens={max_call_ptok} >= {ctx_window})"
                )
            elif not final_state.verified:
                raw_fb = (final_state.feedback or "verification failed").strip()
                fail_cause = raw_fb.splitlines()[0][:240] if raw_fb else "verification failed"

            is_rw = pool == REALWORLD_MINI_POOL or pool.startswith(REALWORLD_MINI_POOL)
            verified_final = (
                final_state.verified and not is_overflow if is_rw else final_state.verified
            )
            p1_final = final_state.pass_at_1 and not is_overflow if is_rw else final_state.pass_at_1
            p3_final = final_state.pass_at_k and not is_overflow if is_rw else final_state.pass_at_k

            case_llm_calls = list(final_state.llm_calls)
            c_local, c_frontier, c_fallbacks, c_warnings = _extract_call_telemetry(case_llm_calls)

            case_result = EvalCaseRunResult(
                case_id=case_id,
                pool=pool,
                taxonomy=taxonomy,
                template=template,
                run_index=run_idx,
                seed=seed,
                provider=provider_name,
                model_name=final_state.model_name or str(default_model_name),
                prompt_version_hash=final_state.prompt_version_hash or prompt_hash,
                attempts=final_state.attempt,
                verified=verified_final,
                pass_at_1=p1_final,
                pass_at_3=p3_final,
                prompt_tokens=final_state.total_prompt_tokens,
                completion_tokens=final_state.total_completion_tokens,
                total_tokens=final_state.total_tokens,
                max_call_prompt_tokens=max_call_ptok,
                context_window_tokens=ctx_window,
                context_overflow=is_overflow,
                borderline=is_borderline,
                failure_cause=fail_cause,
                llm_calls=case_llm_calls,
                local_calls=c_local,
                frontier_calls=c_frontier,
                fallback_count=c_fallbacks,
                warning_flags=c_warnings,
                total_time_s=case_elapsed,
                trace_id=final_state.trace_id,
                resumed=False,
            )

            # Atomic checkpoint write per (case, run, config)
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_ckpt = ckpt_file.with_suffix(".json.tmp")
            tmp_ckpt.write_text(case_result.model_dump_json(indent=2), encoding="utf-8")
            tmp_ckpt.replace(ckpt_file)

            if obs_up:
                item_id = f"{pool}_{case_id}"
                await case_tracer.upsert_dataset_item(
                    dataset_name=pool,
                    item_id=item_id,
                    input_data={"case_id": case_id, "pool": pool},
                    expected_output={"verified": True},
                    metadata={"taxonomy": taxonomy, "template": template},
                )
                await case_tracer.create_dataset_run_item(
                    run_name=f"{effective_run_id}_run{run_idx}",
                    run_description=(
                        f"provider={provider_name} model={case_result.model_name} "
                        f"seed={seed} prompt_hash={prompt_hash[:12]}"
                    ),
                    item_id=item_id,
                    trace_id=case_result.trace_id,
                    metadata={
                        "verified": case_result.verified,
                        "pass_at_1": case_result.pass_at_1,
                        "pass_at_3": case_result.pass_at_3,
                        "attempts": case_result.attempts,
                        "prompt_version_hash": prompt_hash,
                    },
                )

            run_cases.append(case_result)

        run_duration = round(max(0.0, time.perf_counter() - run_start), 6)
        n_cases = len(run_cases)
        p1_cnt = sum(1 for c in run_cases if c.pass_at_1)
        p3_cnt = sum(1 for c in run_cases if c.pass_at_3)
        tot_prompt = sum(c.prompt_tokens for c in run_cases)
        tot_comp = sum(c.completion_tokens for c in run_cases)
        run_local, run_frontier, run_fallbacks, run_warnings = _aggregate_case_telemetry(run_cases)
        out_path = results_dir / f"{effective_run_id}_run{run_idx}.json"

        report = EvalRunReport(
            run_id=effective_run_id,
            run_index=run_idx,
            timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            pool=pool,
            provider=provider_name,
            model_name=run_cases[-1].model_name if run_cases else str(default_model_name),
            seed=seed,
            prompt_version_hash=prompt_hash,
            total_cases=n_cases,
            pass_at_1_count=p1_cnt,
            pass_at_1_rate=round(p1_cnt / n_cases, 6) if n_cases else 0.0,
            pass_at_1_ci_95=wilson_ci_95(p1_cnt, n_cases),
            pass_at_3_count=p3_cnt,
            pass_at_3_rate=round(p3_cnt / n_cases, 6) if n_cases else 0.0,
            pass_at_3_ci_95=wilson_ci_95(p3_cnt, n_cases),
            total_prompt_tokens=tot_prompt,
            total_completion_tokens=tot_comp,
            total_tokens=tot_prompt + tot_comp,
            total_time_s=round(sum(c.total_time_s for c in run_cases) or run_duration, 6),
            local_calls=run_local,
            frontier_calls=run_frontier,
            fallback_count=run_fallbacks,
            warning_flags=run_warnings,
            langfuse_dataset_synced=obs_up,
            output_file=out_path.as_posix(),
            cases=run_cases,
        )
        out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        run_reports.append(report)

    summary = build_eval_summary(run_reports)
    return run_reports, summary
