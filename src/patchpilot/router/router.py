"""Dynamic Model Router implementing `LLMProvider` with privacy gate and fallback resilience."""

from __future__ import annotations

import logging
import time

from patchpilot.providers.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderError,
    _redact_secrets,
    _sanitize_token_count,
)
from patchpilot.router.features import extract_task_features
from patchpilot.router.metrics import (
    RouterMetricsSummary,
    RoutingDecision,
    compute_router_metrics,
    estimate_call_cost_usd,
)
from patchpilot.router.policy import (
    CascadeConfidencePolicy,
    RouteTarget,
    RoutingPolicy,
)

logger = logging.getLogger("patchpilot.router")


def _provider_secrets(provider: object) -> list[str]:
    secrets: list[str] = []
    for attr in ("_api_key", "api_key", "_token", "token"):
        val = getattr(provider, attr, None)
        if isinstance(val, str) and len(val.strip()) >= 4:
            secrets.append(val.strip())
    return secrets


class ModelRouter:
    """Dynamic model router that implements `LLMProvider` and routes between local and frontier."""

    name = "router"

    def __init__(
        self,
        local_provider: LLMProvider,
        frontier_provider: LLMProvider,
        *,
        policy: RoutingPolicy | None = None,
    ) -> None:
        self.local_provider = local_provider
        self.frontier_provider = frontier_provider
        self.policy: RoutingPolicy = policy or CascadeConfidencePolicy()
        self.decisions: list[RoutingDecision] = []
        self.default_model = f"router({self.local_provider.name}+{self.frontier_provider.name})"

    def __repr__(self) -> str:
        return (
            f"ModelRouter(local_provider={self.local_provider!r}, "
            f"frontier_provider={self.frontier_provider!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def last_decision(self) -> RoutingDecision | None:
        """Return the most recent `RoutingDecision`, or `None` if no calls have occurred."""
        return self.decisions[-1] if self.decisions else None

    def reset_metrics(self) -> None:
        """Clear recorded per-call routing decisions."""
        self.decisions.clear()

    def get_metrics_summary(
        self,
        *,
        baseline_ollama_latency_s: float | None = None,
    ) -> RouterMetricsSummary:
        """Return aggregated cost savings vs 100% Gemini and latency vs 100% Ollama."""
        return compute_router_metrics(
            self.decisions,
            baseline_ollama_latency_s=baseline_ollama_latency_s,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not request.messages:
            raise ProviderError("router: request.messages must not be empty")

        # 1. Extract lightweight static task features
        features = extract_task_features(request)

        # 2. Enforce strict privacy gate & evaluate routing policy
        if request.data_class != "public":
            initial_target: RouteTarget = "local"
            reason = f"privacy_gate:data_class={request.data_class}"
            privacy_forced = True
        else:
            eval_res = self.policy.evaluate(features)
            initial_target = eval_res.target
            reason = eval_res.reason
            privacy_forced = eval_res.privacy_forced

        # Defense-in-depth: never allow non-public requests to target frontier
        if request.data_class != "public":
            initial_target = "local"
            privacy_forced = True

        route_target: RouteTarget = initial_target
        fallback_used = False
        warning_flag = ""

        t0 = time.perf_counter()

        if initial_target == "frontier":
            try:
                resp = await self.frontier_provider.generate(request)
            except ProviderError as exc:
                # Resiliency / Fallback: HTTP 429, timeout, or quota error falls back to local
                fallback_used = True
                route_target = "local"
                secrets = _provider_secrets(self.frontier_provider) + _provider_secrets(
                    self.local_provider
                )
                err_text = _redact_secrets(str(exc), secrets)
                if "429" in err_text or "quota" in err_text.lower() or "rate" in err_text.lower():
                    warning_flag = f"frontier_fallback:rate_limit_429:{err_text}"
                elif "timeout" in err_text.lower() or "transport" in err_text.lower():
                    warning_flag = f"frontier_fallback:network_timeout:{err_text}"
                else:
                    warning_flag = f"frontier_fallback:provider_error:{err_text}"
                reason = f"{reason}->fallback_to_local"
                logger.warning(
                    "Frontier provider failed (%s); falling back to local Ollama.",
                    err_text,
                )
                try:
                    resp = await self.local_provider.generate(request)
                except ProviderError as local_exc:
                    safe_local = _redact_secrets(str(local_exc), secrets)
                    raise ProviderError(safe_local) from None
        else:
            try:
                resp = await self.local_provider.generate(request)
            except ProviderError as local_exc:
                secrets = _provider_secrets(self.local_provider) + _provider_secrets(
                    self.frontier_provider
                )
                safe_local = _redact_secrets(str(local_exc), secrets)
                raise ProviderError(safe_local) from None

        latency_s = round(max(0.0, time.perf_counter() - t0), 6)
        p_tok = _sanitize_token_count(resp.usage.prompt_tokens)
        c_tok = _sanitize_token_count(resp.usage.completion_tokens)

        actual_cost = estimate_call_cost_usd(route_target, p_tok, c_tok)
        hypothetical_gemini_cost = estimate_call_cost_usd("frontier", p_tok, c_tok)

        decision = RoutingDecision(
            route_target=route_target,
            initial_target=initial_target,
            selected_provider=resp.provider,
            selected_model=resp.model,
            decision_reason=reason,
            privacy_forced=privacy_forced,
            fallback_used=fallback_used,
            warning_flag=warning_flag,
            latency_s=latency_s,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            estimated_cost_usd=actual_cost,
            hypothetical_frontier_cost_usd=hypothetical_gemini_cost,
        )
        self.decisions.append(decision)
        return resp
