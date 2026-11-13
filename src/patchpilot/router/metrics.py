"""Cost, latency, and routing decision telemetry models for the Dynamic Model Router."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from patchpilot.providers.base import _sanitize_token_count
from patchpilot.router.policy import RouteTarget

# Standard reference token pricing for Gemini Flash ($0.10 / 1M input, $0.40 / 1M output).
# Local Ollama inference on the Windows host GPU has $0.00 marginal API cost.
GEMINI_INPUT_COST_PER_1M_USD = 0.10
GEMINI_OUTPUT_COST_PER_1M_USD = 0.40
OLLAMA_INPUT_COST_PER_1M_USD = 0.00
OLLAMA_OUTPUT_COST_PER_1M_USD = 0.00


def estimate_call_cost_usd(
    target: RouteTarget,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    *,
    input_cost_per_1m: float = GEMINI_INPUT_COST_PER_1M_USD,
    output_cost_per_1m: float = GEMINI_OUTPUT_COST_PER_1M_USD,
) -> float:
    """Compute estimated API cost in USD for a single LLM generation call."""
    if target == "local":
        return 0.0
    p_tok = _sanitize_token_count(prompt_tokens)
    c_tok = _sanitize_token_count(completion_tokens)
    in_rate = max(0.0, float(input_cost_per_1m))
    out_rate = max(0.0, float(output_cost_per_1m))
    cost = (p_tok / 1_000_000.0) * in_rate + (c_tok / 1_000_000.0) * out_rate
    return round(cost, 8)


class RoutingDecision(BaseModel):
    """Per-call routing decision record with reason, latency, cost, and fallback warning."""

    route_target: RouteTarget
    initial_target: RouteTarget
    selected_provider: str
    selected_model: str
    decision_reason: str
    privacy_forced: bool = False
    fallback_used: bool = False
    warning_flag: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    hypothetical_frontier_cost_usd: float = 0.0

    @field_validator("prompt_tokens", "completion_tokens", mode="before")
    @classmethod
    def _coerce_non_negative_tokens(cls, value: object) -> int:
        return _sanitize_token_count(value)


class RouterMetricsSummary(BaseModel):
    """Aggregated routing metrics comparing router cost vs Gemini and latency vs Ollama."""

    total_calls: int = 0
    local_calls: int = 0
    frontier_calls: int = 0
    fallback_count: int = 0
    privacy_override_count: int = 0
    router_total_cost_usd: float = 0.0
    all_gemini_cost_usd: float = 0.0
    cost_savings_usd: float = 0.0
    cost_savings_pct: float = 0.0
    router_total_latency_s: float = 0.0
    all_ollama_estimated_latency_s: float = 0.0
    latency_delta_vs_ollama_s: float = 0.0
    latency_ratio_vs_ollama: float = 1.0


def compute_router_metrics(
    decisions: list[RoutingDecision],
    *,
    baseline_ollama_latency_s: float | None = None,
    default_local_call_latency_s: float = 0.25,
) -> RouterMetricsSummary:
    """Compute cost savings vs. 100% Gemini routing and latency vs. 100% Ollama routing."""
    if not decisions:
        return RouterMetricsSummary()

    total_calls = len(decisions)
    local_calls = sum(1 for d in decisions if d.route_target == "local")
    frontier_calls = sum(1 for d in decisions if d.route_target == "frontier")
    fallback_count = sum(1 for d in decisions if d.fallback_used)
    privacy_count = sum(1 for d in decisions if d.privacy_forced)

    router_cost = round(sum(d.estimated_cost_usd for d in decisions), 8)
    all_gemini_cost = round(sum(d.hypothetical_frontier_cost_usd for d in decisions), 8)
    savings_usd = round(max(0.0, all_gemini_cost - router_cost), 8)
    savings_pct = (
        round((savings_usd / all_gemini_cost) * 100.0, 2)
        if all_gemini_cost > 0.0
        else (100.0 if local_calls == total_calls else 0.0)
    )

    router_latency = round(sum(d.latency_s for d in decisions), 6)

    if baseline_ollama_latency_s is not None and baseline_ollama_latency_s > 0.0:
        all_ollama_latency = round(baseline_ollama_latency_s, 6)
    else:
        local_lats = [
            d.latency_s for d in decisions if d.route_target == "local" and d.latency_s > 0
        ]
        avg_local_lat = (
            sum(local_lats) / len(local_lats) if local_lats else default_local_call_latency_s
        )
        all_ollama_latency = round(avg_local_lat * total_calls, 6)

    latency_delta = round(router_latency - all_ollama_latency, 6)
    latency_ratio = (
        round(router_latency / all_ollama_latency, 4) if all_ollama_latency > 0.0 else 1.0
    )

    return RouterMetricsSummary(
        total_calls=total_calls,
        local_calls=local_calls,
        frontier_calls=frontier_calls,
        fallback_count=fallback_count,
        privacy_override_count=privacy_count,
        router_total_cost_usd=router_cost,
        all_gemini_cost_usd=all_gemini_cost,
        cost_savings_usd=savings_usd,
        cost_savings_pct=savings_pct,
        router_total_latency_s=router_latency,
        all_ollama_estimated_latency_s=all_ollama_latency,
        latency_delta_vs_ollama_s=latency_delta,
        latency_ratio_vs_ollama=latency_ratio,
    )
