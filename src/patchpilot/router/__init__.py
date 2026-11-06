"""Phase 6 Dynamic Model Router (`src/patchpilot/router/`)."""

from __future__ import annotations

from patchpilot.router.benchmark import (
    BenchmarkModeSummary,
    RouterBenchmarkReport,
    run_router_smoke_benchmark,
)
from patchpilot.router.features import TaskFeatures, extract_task_features
from patchpilot.router.metrics import (
    GEMINI_INPUT_COST_PER_1M_USD,
    GEMINI_OUTPUT_COST_PER_1M_USD,
    RouterMetricsSummary,
    RoutingDecision,
    compute_router_metrics,
    estimate_call_cost_usd,
)
from patchpilot.router.policy import (
    CascadeConfidencePolicy,
    PolicyEvaluation,
    PolicyMode,
    RouteTarget,
    RoutingPolicy,
    StaticRulePolicy,
)
from patchpilot.router.router import ModelRouter

__all__ = [
    "BenchmarkModeSummary",
    "CascadeConfidencePolicy",
    "GEMINI_INPUT_COST_PER_1M_USD",
    "GEMINI_OUTPUT_COST_PER_1M_USD",
    "ModelRouter",
    "PolicyEvaluation",
    "PolicyMode",
    "RouteTarget",
    "RouterBenchmarkReport",
    "RouterMetricsSummary",
    "RoutingDecision",
    "RoutingPolicy",
    "StaticRulePolicy",
    "TaskFeatures",
    "compute_router_metrics",
    "estimate_call_cost_usd",
    "extract_task_features",
    "run_router_smoke_benchmark",
]
