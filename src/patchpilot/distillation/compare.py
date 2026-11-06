"""Evaluation comparison and markdown report generator for distilled models vs. baseline."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from patchpilot.evals.freeze import verify_frozen_eval_split
from patchpilot.evals.generator import load_pool_cases
from patchpilot.evals.models import FrozenManifest
from patchpilot.evals.runner import EvalIntegrityError, EvalRunReport, build_eval_summary
from patchpilot.router.metrics import estimate_call_cost_usd


class ModelComparisonMetrics(BaseModel):
    """Aggregated evaluation metrics for a single model configuration (baseline or candidate)."""

    run_id: str
    model_name: str
    provider: str
    n_runs: int
    total_observations: int
    pass_at_1_count: int
    pass_at_1_rate: float
    pass_at_1_ci_95: tuple[float, float]
    pass_at_3_count: int
    pass_at_3_rate: float
    pass_at_3_ci_95: tuple[float, float]
    avg_latency_per_case_s: float
    total_latency_s: float
    estimated_cost_usd: float


class DistillationComparisonReport(BaseModel):
    """Full comparison report between baseline and distilled candidate runs on `seeded_eval`."""

    frozen_sha256: str
    train_eval_overlap: int
    prompt_version_hash: str
    baseline: ModelComparisonMetrics
    candidate: ModelComparisonMetrics
    pass_at_1_delta_pp: float
    pass_at_3_delta_pp: float
    latency_delta_s: float
    cost_delta_usd: float


def load_run_reports(
    run_id_or_path: str,
    *,
    results_dir: Path = Path("evals/results"),
) -> list[EvalRunReport]:
    """Load one or more `EvalRunReport` JSON files matching `run_id_or_path`."""
    direct = Path(run_id_or_path)
    if direct.is_file():
        return [EvalRunReport.model_validate_json(direct.read_text(encoding="utf-8"))]

    if results_dir.is_dir():
        matches = sorted(results_dir.glob(f"{run_id_or_path}*.json"))
        if matches:
            return [
                EvalRunReport.model_validate_json(p.read_text(encoding="utf-8")) for p in matches
            ]

    raise FileNotFoundError(
        f"No evaluation run report JSON files found for {run_id_or_path!r} in {results_dir}"
    )


def _compute_metrics_from_reports(reports: list[EvalRunReport]) -> ModelComparisonMetrics:
    summary = build_eval_summary(reports)
    total_lat = round(sum(r.total_time_s for r in reports), 6)
    avg_lat = (
        round(total_lat / summary.total_observations, 6) if summary.total_observations else 0.0
    )

    total_cost = 0.0
    for rep in reports:
        target = "frontier" if rep.provider == "gemini" else "local"
        for case in rep.cases:
            if case.llm_calls:
                for call in case.llm_calls:
                    if call.estimated_cost_usd > 0.0:
                        total_cost += call.estimated_cost_usd
                    elif call.provider == "gemini" or call.route_target == "frontier":
                        total_cost += estimate_call_cost_usd(
                            "frontier", call.prompt_tokens, call.completion_tokens
                        )
            elif target == "frontier":
                total_cost += estimate_call_cost_usd(
                    "frontier", case.prompt_tokens, case.completion_tokens
                )

    return ModelComparisonMetrics(
        run_id=summary.run_id,
        model_name=summary.model_name,
        provider=summary.provider,
        n_runs=summary.n_runs,
        total_observations=summary.total_observations,
        pass_at_1_count=summary.pass_at_1_count,
        pass_at_1_rate=summary.pass_at_1_rate,
        pass_at_1_ci_95=summary.pass_at_1_ci_95,
        pass_at_3_count=summary.pass_at_3_count,
        pass_at_3_rate=summary.pass_at_3_rate,
        pass_at_3_ci_95=summary.pass_at_3_ci_95,
        avg_latency_per_case_s=avg_lat,
        total_latency_s=total_lat,
        estimated_cost_usd=round(total_cost, 8),
    )


def compare_eval_runs(
    baseline_reports: list[EvalRunReport],
    candidate_reports: list[EvalRunReport],
    *,
    datasets_dir: Path = Path("evals/datasets"),
    frozen_path: Path = Path("evals/FROZEN.json"),
    require_min_runs: int = 3,
) -> DistillationComparisonReport:
    """Compare baseline and candidate eval runs with fail-closed split and prompt hash checks."""
    if len(baseline_reports) < require_min_runs or len(candidate_reports) < require_min_runs:
        raise EvalIntegrityError(
            f"Full comparison requires n_runs >= {require_min_runs} for both baseline "
            f"(got {len(baseline_reports)}) and candidate (got {len(candidate_reports)})"
        )

    valid, msg = verify_frozen_eval_split(frozen_path=frozen_path, datasets_dir=datasets_dir)
    if not valid:
        raise EvalIntegrityError(f"Frozen split verification failed before compare: {msg}")

    frozen_manifest = FrozenManifest.model_validate_json(frozen_path.read_text(encoding="utf-8"))
    eval_pool = frozen_manifest.pools.get("seeded_eval")
    eval_hashes = set(eval_pool.cases.values()) if eval_pool is not None else set()
    train_cases = load_pool_cases(datasets_dir, "seeded_train")
    train_hashes = {c.metadata.content_hash for c in train_cases}
    overlap = len(eval_hashes.intersection(train_hashes))
    if overlap > 0:
        raise EvalIntegrityError(f"Detected non-zero train/eval overlap: {overlap}")

    # Verify prompt_version_hash consistency across all baseline and candidate runs
    combined_summary = build_eval_summary(baseline_reports + candidate_reports)

    base_metrics = _compute_metrics_from_reports(baseline_reports)
    cand_metrics = _compute_metrics_from_reports(candidate_reports)

    p1_delta_pp = round((cand_metrics.pass_at_1_rate - base_metrics.pass_at_1_rate) * 100.0, 2)
    p3_delta_pp = round((cand_metrics.pass_at_3_rate - base_metrics.pass_at_3_rate) * 100.0, 2)
    lat_delta_s = round(
        cand_metrics.avg_latency_per_case_s - base_metrics.avg_latency_per_case_s, 4
    )
    cost_delta = round(cand_metrics.estimated_cost_usd - base_metrics.estimated_cost_usd, 8)

    return DistillationComparisonReport(
        frozen_sha256=frozen_manifest.overall_sha256,
        train_eval_overlap=overlap,
        prompt_version_hash=combined_summary.prompt_version_hash,
        baseline=base_metrics,
        candidate=cand_metrics,
        pass_at_1_delta_pp=p1_delta_pp,
        pass_at_3_delta_pp=p3_delta_pp,
        latency_delta_s=lat_delta_s,
        cost_delta_usd=cost_delta,
    )


def format_comparison_markdown(report: DistillationComparisonReport) -> str:
    """Format the comparative markdown report required by PLAN.md Phase 8."""
    b = report.baseline
    c = report.candidate
    b_p1_lo, b_p1_hi = b.pass_at_1_ci_95
    c_p1_lo, c_p1_hi = c.pass_at_1_ci_95
    b_p3_lo, b_p3_hi = b.pass_at_3_ci_95
    c_p3_lo, c_p3_hi = c.pass_at_3_ci_95

    sign_p1 = "+" if report.pass_at_1_delta_pp >= 0 else ""
    sign_p3 = "+" if report.pass_at_3_delta_pp >= 0 else ""
    sign_lat = "+" if report.latency_delta_s >= 0 else ""

    lines = [
        f"frozen split: OK (sha256={report.frozen_sha256})",
        f"train/eval overlap: {report.train_eval_overlap}",
        f"prompt_version_hash: {report.prompt_version_hash}",
        "",
        "| Metric | Baseline (`" + b.run_id + "`) | Candidate (`" + c.run_id + "`) | Delta |",
        "|---|---|---|---|",
        f"| Model | `{b.model_name}` | `{c.model_name}` | — |",
        (
            f"| Runs (`n_runs`) | {b.n_runs} ({b.total_observations} obs) | "
            f"{c.n_runs} ({c.total_observations} obs) | — |"
        ),
        (
            f"| `pass@1` (95% Wilson CI) | {b.pass_at_1_count}/{b.total_observations} "
            f"({b.pass_at_1_rate:.1%}) [{b_p1_lo:.1%}, {b_p1_hi:.1%}] | "
            f"{c.pass_at_1_count}/{c.total_observations} ({c.pass_at_1_rate:.1%}) "
            f"[{c_p1_lo:.1%}, {c_p1_hi:.1%}] | {sign_p1}{report.pass_at_1_delta_pp:.2f} pp |"
        ),
        (
            f"| `pass@3` (95% Wilson CI) | {b.pass_at_3_count}/{b.total_observations} "
            f"({b.pass_at_3_rate:.1%}) [{b_p3_lo:.1%}, {b_p3_hi:.1%}] | "
            f"{c.pass_at_3_count}/{c.total_observations} ({c.pass_at_3_rate:.1%}) "
            f"[{c_p3_lo:.1%}, {c_p3_hi:.1%}] | {sign_p3}{report.pass_at_3_delta_pp:.2f} pp |"
        ),
        (
            f"| Avg Latency / Case | {b.avg_latency_per_case_s:.3f}s | "
            f"{c.avg_latency_per_case_s:.3f}s | {sign_lat}{report.latency_delta_s:.3f}s |"
        ),
        (
            f"| Estimated API Cost | ${b.estimated_cost_usd:.6f} | "
            f"${c.estimated_cost_usd:.6f} | ${report.cost_delta_usd:+.6f} |"
        ),
    ]
    return "\n".join(lines)
