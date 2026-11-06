"""Routing policies (Static heuristic & Cascade confidence-aware) with strict privacy gate."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from patchpilot.router.features import TaskFeatures

RouteTarget = Literal["local", "frontier"]
PolicyMode = Literal["static", "cascade", "hybrid"]


class PolicyEvaluation(BaseModel):
    """Output of a routing policy evaluation before provider execution."""

    target: RouteTarget
    reason: str
    policy_mode: PolicyMode
    privacy_forced: bool = False
    complexity_score: float = 0.0


@runtime_checkable
class RoutingPolicy(Protocol):
    """Protocol for static, cascade, or hybrid routing policies."""

    def evaluate(self, features: TaskFeatures) -> PolicyEvaluation:
        """Choose `'local'` (Ollama) or `'frontier'` (Gemini) for `features`."""
        ...


class StaticRulePolicy:
    """Static rule-based policy using vulnerability/error heuristics and AST thresholds."""

    def __init__(
        self,
        *,
        ast_node_threshold: int = 80,
        ast_depth_threshold: int = 10,
        multi_file_threshold: int = 3,
        token_threshold: int = 1200,
    ) -> None:
        self.ast_node_threshold = ast_node_threshold
        self.ast_depth_threshold = ast_depth_threshold
        self.multi_file_threshold = multi_file_threshold
        self.token_threshold = token_threshold

    def evaluate(self, features: TaskFeatures) -> PolicyEvaluation:
        # 1. Strict Privacy Gate (SPEC §9): never evaluate frontier rules if data_class != "public"
        if features.data_class != "public":
            return PolicyEvaluation(
                target="local",
                reason=f"privacy_gate:data_class={features.data_class}",
                policy_mode="static",
                privacy_forced=True,
                complexity_score=0.0,
            )

        # 2. Known local-model weak spots (from baseline analysis: none_handling & sql_injection)
        for pat in features.matched_weak_patterns:
            if pat.startswith("none_handling:profile_display_name") or pat.startswith(
                "none_handling:unconditional_strip_on_nullable"
            ):
                return PolicyEvaluation(
                    target="frontier",
                    reason=f"static_weak_spot:{pat}",
                    policy_mode="static",
                    complexity_score=0.95,
                )
            if pat.startswith("sql_injection:login_query") or pat.startswith(
                "sql_injection:complex_union_auth"
            ):
                return PolicyEvaluation(
                    target="frontier",
                    reason=f"static_weak_spot:{pat}",
                    policy_mode="static",
                    complexity_score=0.90,
                )

        if features.predicted_taxonomy in ("none_handling", "sql_injection") and (
            features.matched_weak_patterns
            or features.failing_error_type in ("AttributeError", "TypeError", "AssertionError")
        ):
            return PolicyEvaluation(
                target="frontier",
                reason=f"static_taxonomy_escalation:{features.predicted_taxonomy}",
                policy_mode="static",
                complexity_score=0.85,
            )

        # 3. Structural AST / repository complexity thresholds
        if features.ast_node_count >= self.ast_node_threshold:
            return PolicyEvaluation(
                target="frontier",
                reason=(
                    f"static_ast_complexity:nodes={features.ast_node_count}"
                    f">={self.ast_node_threshold}"
                ),
                policy_mode="static",
                complexity_score=0.80,
            )

        if features.max_ast_depth >= self.ast_depth_threshold:
            return PolicyEvaluation(
                target="frontier",
                reason=(
                    f"static_ast_depth:depth={features.max_ast_depth}>={self.ast_depth_threshold}"
                ),
                policy_mode="static",
                complexity_score=0.75,
            )

        if features.non_test_file_count >= self.multi_file_threshold:
            return PolicyEvaluation(
                target="frontier",
                reason=(
                    f"static_multi_file:files={features.non_test_file_count}"
                    f">={self.multi_file_threshold}"
                ),
                policy_mode="static",
                complexity_score=0.70,
            )

        if features.estimated_prompt_tokens >= self.token_threshold:
            return PolicyEvaluation(
                target="frontier",
                reason=(
                    f"static_token_budget:tokens={features.estimated_prompt_tokens}"
                    f">={self.token_threshold}"
                ),
                policy_mode="static",
                complexity_score=0.65,
            )

        return PolicyEvaluation(
            target="local",
            reason="static_local_default:low_complexity",
            policy_mode="static",
            complexity_score=0.20,
        )


class CascadeConfidencePolicy:
    """Cascade policy that escalates retries and low-confidence tasks to the frontier model."""

    def __init__(
        self,
        *,
        static_policy: StaticRulePolicy | None = None,
        escalate_on_retry: bool = True,
        confidence_threshold: float = 0.60,
    ) -> None:
        self.static_policy = static_policy or StaticRulePolicy()
        self.escalate_on_retry = escalate_on_retry
        self.confidence_threshold = confidence_threshold

    def evaluate(self, features: TaskFeatures) -> PolicyEvaluation:
        # 1. Strict Privacy Gate (SPEC §9): never evaluate frontier rules if data_class != "public"
        if features.data_class != "public":
            return PolicyEvaluation(
                target="local",
                reason=f"privacy_gate:data_class={features.data_class}",
                policy_mode="cascade",
                privacy_forced=True,
                complexity_score=0.0,
            )

        # 2. Cascade retry escalation: if a previous local attempt failed verification, escalate
        if self.escalate_on_retry and features.is_retry:
            return PolicyEvaluation(
                target="frontier",
                reason=f"cascade_retry_escalation:attempt={features.retry_attempt}",
                policy_mode="cascade",
                complexity_score=0.95,
            )

        # 3. Evaluate static rule signals
        static_eval = self.static_policy.evaluate(features)
        if static_eval.target == "frontier":
            return PolicyEvaluation(
                target="frontier",
                reason=static_eval.reason,
                policy_mode="cascade",
                privacy_forced=False,
                complexity_score=static_eval.complexity_score,
            )

        # 4. Confidence estimation for local model
        local_confidence = 0.85
        if features.low_confidence_signal:
            local_confidence -= 0.35
        if features.non_test_file_count > 1:
            local_confidence -= 0.15
        if features.ast_node_count > 50:
            local_confidence -= 0.10

        if local_confidence < self.confidence_threshold:
            return PolicyEvaluation(
                target="frontier",
                reason=f"cascade_low_confidence:conf={local_confidence:.2f}<{self.confidence_threshold:.2f}",
                policy_mode="cascade",
                complexity_score=round(1.0 - local_confidence, 2),
            )

        return PolicyEvaluation(
            target="local",
            reason=f"cascade_local_confident:conf={local_confidence:.2f}",
            policy_mode="cascade",
            complexity_score=round(1.0 - local_confidence, 2),
        )
