"""Smoke benchmark runner for the Dynamic Model Router on a sample of `seeded_train` cases."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, Field

from patchpilot.config import LangfuseSettings
from patchpilot.providers.base import LLMRequest, LLMResponse, Usage
from patchpilot.router.features import extract_task_features
from patchpilot.router.metrics import RouterMetricsSummary, estimate_call_cost_usd
from patchpilot.router.policy import RoutingPolicy
from patchpilot.router.router import ModelRouter
from patchpilot.sandbox.runner import SandboxResult
from patchpilot.sandbox.tar_and_diff import JUnitReport, PatchApplyError, apply_unified_diff
from patchpilot.tracing import PatchPilotTracer


class BenchmarkModeSummary(BaseModel):
    """Aggregated outcome for one provider configuration (ollama_only, router, gemini_only)."""

    mode: str
    total_cases: int
    pass_at_1_count: int
    pass_at_1_rate: float
    pass_at_3_count: int
    pass_at_3_rate: float
    local_calls: int = 0
    frontier_calls: int = 0
    estimated_cost_usd: float = 0.0
    simulated_latency_s: float = 0.0


class RouterBenchmarkReport(BaseModel):
    """Comparison report of Ollama-only vs. Router vs. Gemini-only on `seeded_train`."""

    pool: str = "seeded_train"
    sample_size: int = 10
    case_ids: list[str] = Field(default_factory=list)
    ollama_only: BenchmarkModeSummary
    router: BenchmarkModeSummary
    gemini_only: BenchmarkModeSummary
    router_metrics: RouterMetricsSummary


class _SimulatedBenchmarkProvider:
    """Fixture provider simulating local weak spots vs frontier strength without API quota."""

    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        known_patch: str,
        simulate_local_weak_spots: bool,
        simulated_latency_s: float,
    ) -> None:
        self.name = provider_name
        self.default_model = model_name
        self._known_patch = known_patch
        self._simulate_local_weak_spots = simulate_local_weak_spots
        self._simulated_latency_s = simulated_latency_s
        self.call_count = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        features = extract_task_features(request)
        p_tokens = max(10, features.estimated_prompt_tokens)

        # Simulate local 7B model struggling on none_handling / sql_injection weak spots
        if self._simulate_local_weak_spots and (
            features.predicted_taxonomy in ("none_handling", "sql_injection")
            or bool(features.matched_weak_patterns)
        ):
            bad_text = (
                "FINDING: unknown.py:1: weak_spot: simulated local weak spot\n"
                if features.node_kind == "analyze"
                else "# incomplete local patch on weak-spot pattern\n"
            )
            return LLMResponse(
                text=bad_text,
                model=self.default_model,
                provider=self.name,
                usage=Usage(prompt_tokens=p_tokens, completion_tokens=24),
            )

        c_tokens = max(12, len(self._known_patch.split()))
        return LLMResponse(
            text=self._known_patch,
            model=self.default_model,
            provider=self.name,
            usage=Usage(prompt_tokens=p_tokens, completion_tokens=c_tokens),
        )


async def _fast_in_memory_verify(files: dict[str, str], patch_diff: str) -> SandboxResult:
    """Verify whether the normalized diff matches and applies cleanly in a temp directory."""
    import tempfile

    if not patch_diff.strip():
        return SandboxResult(exit_code=1, patch_applied=False, error="empty patch")

    with tempfile.TemporaryDirectory() as tmp:
        wdir = Path(tmp)
        for rel_p, content in files.items():
            dest = wdir / rel_p
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        try:
            apply_unified_diff(patch_diff, wdir)
        except PatchApplyError as exc:
            return SandboxResult(exit_code=1, patch_applied=False, error=str(exc))

    return SandboxResult(
        exit_code=0,
        patch_applied=True,
        junit=JUnitReport(passed=1, failures=0, errors=0),
    )


async def run_router_smoke_benchmark(
    *,
    datasets_dir: Path = Path("evals/datasets"),
    pool: str = "seeded_train",
    limit: int = 10,
    policy: RoutingPolicy | None = None,
) -> RouterBenchmarkReport:
    """Run a 10-case smoke benchmark on `seeded_train` comparing Ollama, Router, and Gemini."""
    from patchpilot.__main__ import _load_case_files, load_mock_ground_truth
    from patchpilot.pipeline.graph import run_pipeline
    from patchpilot.pipeline.models import PipelineState

    pool_dir = datasets_dir / pool
    if not pool_dir.is_dir():
        raise FileNotFoundError(f"Pool directory not found: {pool_dir}")

    case_dirs = sorted([d for d in pool_dir.iterdir() if d.is_dir()], key=lambda d: d.name)[:limit]
    if not case_dirs:
        raise ValueError(f"No cases found in {pool_dir}")

    case_ids = [d.name for d in case_dirs]
    disabled_lf = LangfuseSettings(host="", public_key="", secret_key="")

    # Track stats for each mode
    ollama_p1 = 0
    ollama_p3 = 0
    ollama_calls = 0
    ollama_sim_lat = 0.0

    gemini_p1 = 0
    gemini_p3 = 0
    gemini_calls = 0
    gemini_cost = 0.0
    gemini_sim_lat = 0.0

    router_p1 = 0
    router_p3 = 0
    aggregate_router: ModelRouter | None = None

    for cdir in case_dirs:
        files = _load_case_files(cdir)
        known_patch = load_mock_ground_truth(cdir)

        # 1. Ollama-only run
        ollama_prov = _SimulatedBenchmarkProvider(
            provider_name="ollama",
            model_name="qwen2.5-coder:7b",
            known_patch=known_patch,
            simulate_local_weak_spots=True,
            simulated_latency_s=0.25,
        )
        st_ollama = await run_pipeline(
            PipelineState(
                case_id=f"{pool}/{cdir.name}",
                files=files,
                data_class="public",
                provider="ollama",
            ),
            provider=ollama_prov,
            verify_fn=_fast_in_memory_verify,
            tracer=PatchPilotTracer(settings=disabled_lf, enabled=False),
        )
        if st_ollama.pass_at_1:
            ollama_p1 += 1
        if st_ollama.pass_at_k:
            ollama_p3 += 1
        ollama_calls += ollama_prov.call_count
        ollama_sim_lat += ollama_prov.call_count * 0.25

        # 2. Gemini-only run
        gemini_prov = _SimulatedBenchmarkProvider(
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            known_patch=known_patch,
            simulate_local_weak_spots=False,
            simulated_latency_s=0.85,
        )
        st_gemini = await run_pipeline(
            PipelineState(
                case_id=f"{pool}/{cdir.name}",
                files=files,
                data_class="public",
                provider="gemini",
            ),
            provider=gemini_prov,
            verify_fn=_fast_in_memory_verify,
            tracer=PatchPilotTracer(settings=disabled_lf, enabled=False),
        )
        if st_gemini.pass_at_1:
            gemini_p1 += 1
        if st_gemini.pass_at_k:
            gemini_p3 += 1
        gemini_calls += gemini_prov.call_count
        gemini_sim_lat += gemini_prov.call_count * 0.85

        gemini_cost += estimate_call_cost_usd(
            "frontier",
            st_gemini.total_prompt_tokens,
            st_gemini.total_completion_tokens,
        )

        # 3. Dynamic Model Router run
        r_local = _SimulatedBenchmarkProvider(
            provider_name="ollama",
            model_name="qwen2.5-coder:7b",
            known_patch=known_patch,
            simulate_local_weak_spots=True,
            simulated_latency_s=0.25,
        )
        r_frontier = _SimulatedBenchmarkProvider(
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            known_patch=known_patch,
            simulate_local_weak_spots=False,
            simulated_latency_s=0.85,
        )
        case_router = ModelRouter(r_local, r_frontier, policy=policy)
        if aggregate_router is None:
            aggregate_router = ModelRouter(r_local, r_frontier, policy=policy)

        t0 = time.perf_counter()
        st_router = await run_pipeline(
            PipelineState(
                case_id=f"{pool}/{cdir.name}",
                files=files,
                data_class="public",
                provider="router",
            ),
            provider=case_router,
            verify_fn=_fast_in_memory_verify,
            tracer=PatchPilotTracer(settings=disabled_lf, enabled=False),
        )
        _ = time.perf_counter() - t0
        if st_router.pass_at_1:
            router_p1 += 1
        if st_router.pass_at_k:
            router_p3 += 1

        # Record realistic simulated latencies on the router decisions for comparison
        for dec in case_router.decisions:
            dec.latency_s = 0.25 if dec.route_target == "local" else 0.85
            aggregate_router.decisions.append(dec)

    n = len(case_dirs)
    assert aggregate_router is not None
    r_metrics = aggregate_router.get_metrics_summary(baseline_ollama_latency_s=ollama_sim_lat)

    return RouterBenchmarkReport(
        pool=pool,
        sample_size=n,
        case_ids=case_ids,
        ollama_only=BenchmarkModeSummary(
            mode="ollama_only",
            total_cases=n,
            pass_at_1_count=ollama_p1,
            pass_at_1_rate=round(ollama_p1 / n, 4),
            pass_at_3_count=ollama_p3,
            pass_at_3_rate=round(ollama_p3 / n, 4),
            local_calls=ollama_calls,
            frontier_calls=0,
            estimated_cost_usd=0.0,
            simulated_latency_s=round(ollama_sim_lat, 4),
        ),
        router=BenchmarkModeSummary(
            mode="router",
            total_cases=n,
            pass_at_1_count=router_p1,
            pass_at_1_rate=round(router_p1 / n, 4),
            pass_at_3_count=router_p3,
            pass_at_3_rate=round(router_p3 / n, 4),
            local_calls=r_metrics.local_calls,
            frontier_calls=r_metrics.frontier_calls,
            estimated_cost_usd=r_metrics.router_total_cost_usd,
            simulated_latency_s=round(r_metrics.router_total_latency_s, 4),
        ),
        gemini_only=BenchmarkModeSummary(
            mode="gemini_only",
            total_cases=n,
            pass_at_1_count=gemini_p1,
            pass_at_1_rate=round(gemini_p1 / n, 4),
            pass_at_3_count=gemini_p3,
            pass_at_3_rate=round(gemini_p3 / n, 4),
            local_calls=0,
            frontier_calls=gemini_calls,
            estimated_cost_usd=round(gemini_cost, 8),
            simulated_latency_s=round(gemini_sim_lat, 4),
        ),
        router_metrics=r_metrics,
    )
