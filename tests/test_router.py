"""Phase 6 unit, fallback, pipeline integration, and smoke benchmark tests for ModelRouter."""

from __future__ import annotations

import asyncio

import httpx

from patchpilot.config import LangfuseSettings
from patchpilot.pipeline.graph import run_pipeline
from patchpilot.pipeline.models import PipelineState
from patchpilot.providers import (
    GeminiProvider,
    LLMProvider,
    MockProvider,
    OllamaProvider,
    get_provider,
)
from patchpilot.providers.base import LLMRequest, Message
from patchpilot.router import (
    CascadeConfidencePolicy,
    ModelRouter,
    StaticRulePolicy,
    extract_task_features,
    run_router_smoke_benchmark,
)
from patchpilot.sandbox.runner import SandboxResult
from patchpilot.sandbox.tar_and_diff import JUnitReport
from patchpilot.tracing import PatchPilotTracer


def _ollama_response_json(
    text: str = "LOCAL_PATCH", prompt_tok: int = 40, comp_tok: int = 20
) -> dict[str, object]:
    return {
        "model": "qwen2.5-coder:7b",
        "message": {"role": "assistant", "content": text},
        "done": True,
        "prompt_eval_count": prompt_tok,
        "eval_count": comp_tok,
    }


def _gemini_response_json(
    text: str = "FRONTIER_PATCH", prompt_tok: int = 40, comp_tok: int = 20
) -> dict[str, object]:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}],
        "usageMetadata": {
            "promptTokenCount": prompt_tok,
            "candidatesTokenCount": comp_tok,
            "totalTokenCount": prompt_tok + comp_tok,
        },
        "modelVersion": "gemini-3.8-flash",
    }


def _make_router_with_transports(
    ollama_handler: object,
    gemini_handler: object,
    *,
    policy: StaticRulePolicy | CascadeConfidencePolicy | None = None,
) -> ModelRouter:
    async def no_sleep(_: float) -> None:
        return None

    local = OllamaProvider(
        "http://localhost:11434",
        "qwen2.5-coder:7b",
        timeout_s=5.0,
        max_attempts=2,
        retry_base_delay_s=0.01,
        retry_max_delay_s=0.05,
        transport=httpx.MockTransport(ollama_handler),  # type: ignore[arg-type]
        sleep=no_sleep,
    )
    frontier = GeminiProvider(
        "fake-api-key-no-secret",
        "gemini-3.8-flash",
        timeout_s=5.0,
        max_attempts=2,
        retry_base_delay_s=0.01,
        retry_max_delay_s=0.05,
        transport=httpx.MockTransport(gemini_handler),  # type: ignore[arg-type]
        sleep=no_sleep,
    )
    return ModelRouter(local, frontier, policy=policy)


def test_privacy_override_private_data_always_routes_to_ollama() -> None:
    """SPEC §9: private data_class forces local routing even on high-complexity prompts."""
    ollama_calls: list[httpx.Request] = []
    gemini_calls: list[httpx.Request] = []

    router = _make_router_with_transports(
        lambda req: (ollama_calls.append(req), httpx.Response(200, json=_ollama_response_json()))[
            1
        ],
        lambda req: (gemini_calls.append(req), httpx.Response(200, json=_gemini_response_json()))[
            1
        ],
    )
    assert isinstance(router, LLMProvider)
    assert isinstance(get_provider("router"), LLMProvider)

    # Complex SQL injection + None handling prompt that would normally trigger Gemini
    complex_private_prompt = (
        "--- File: user_fmt_101.py ---\n"
        "def format_user_display_name(profile: dict) -> str:\n"
        "    return profile['full_name'].strip()\n"
        "def build_login_query(username: str):\n"
        "    return f\"SELECT id, pass_hash FROM accounts WHERE username = '{username}'\", ()\n"
    )
    req = LLMRequest(
        messages=[Message(role="user", content=complex_private_prompt)],
        data_class="private",  # Default fail-closed private classification
    )

    resp = asyncio.run(router.generate(req))
    assert resp.provider == "ollama"
    assert resp.text == "LOCAL_PATCH"
    assert len(ollama_calls) == 1
    assert len(gemini_calls) == 0

    dec = router.last_decision
    assert dec is not None
    assert dec.route_target == "local"
    assert dec.privacy_forced is True
    assert "privacy_gate" in dec.decision_reason
    assert dec.estimated_cost_usd == 0.0


def test_static_policy_simple_task_routes_to_local_ollama() -> None:
    """Simple public off-by-one / operator task with small AST routes to local Ollama."""
    ollama_calls: list[httpx.Request] = []
    gemini_calls: list[httpx.Request] = []

    router = _make_router_with_transports(
        lambda req: (ollama_calls.append(req), httpx.Response(200, json=_ollama_response_json()))[
            1
        ],
        lambda req: (gemini_calls.append(req), httpx.Response(200, json=_gemini_response_json()))[
            1
        ],
        policy=StaticRulePolicy(ast_node_threshold=80),
    )

    simple_prompt = (
        "--- File: inv_page_101.py ---\n"
        "def paginate_inventory(items: list[str], page: int, page_size: int = 5) -> list[str]:\n"
        "    offset = page * page_size\n"
        "    return items[offset : offset + page_size]\n"
    )
    req = LLMRequest(
        messages=[Message(role="user", content=simple_prompt)],
        data_class="public",
    )
    features = extract_task_features(req)
    assert features.ast_node_count < 80
    assert features.matched_weak_patterns == []

    resp = asyncio.run(router.generate(req))
    assert resp.provider == "ollama"
    assert len(ollama_calls) == 1
    assert len(gemini_calls) == 0
    assert router.last_decision is not None
    assert router.last_decision.route_target == "local"
    assert "low_complexity" in router.last_decision.decision_reason


def test_heuristic_trigger_weak_spots_route_to_gemini() -> None:
    """Known local weak spots (`profile_display_name` and `login_query`) route to Gemini."""
    ollama_calls: list[httpx.Request] = []
    gemini_calls: list[httpx.Request] = []

    router = _make_router_with_transports(
        lambda req: (ollama_calls.append(req), httpx.Response(200, json=_ollama_response_json()))[
            1
        ],
        lambda req: (gemini_calls.append(req), httpx.Response(200, json=_gemini_response_json()))[
            1
        ],
        policy=StaticRulePolicy(),
    )

    # 1. none_handling weak spot: profile_display_name (0.0% local baseline)
    none_prompt = (
        "Failing test output: AttributeError: 'NoneType' object has no attribute 'strip'\n"
        "--- File: user_fmt_202.py ---\n"
        "def format_user_display_name(profile: dict) -> str:\n"
        "    return profile['full_name'].strip()\n"
        "--- Test File: test_user_fmt_202.py ---\n"
        "def test_format_display_name_none():\n"
        "    profile = {'username': 'jdoe', 'full_name': None}\n"
        "    assert format_user_display_name(profile) == 'jdoe'\n"
    )
    req_none = LLMRequest(messages=[Message(role="user", content=none_prompt)], data_class="public")
    resp_none = asyncio.run(router.generate(req_none))
    assert resp_none.provider == "gemini"
    assert router.last_decision is not None
    assert router.last_decision.route_target == "frontier"
    assert "none_handling" in router.last_decision.decision_reason

    # 2. sql_injection weak spot: login_query (37.5% local baseline)
    sql_prompt = (
        "--- File: login_query_303.py ---\n"
        "def build_login_query(username: str) -> tuple[str, tuple]:\n"
        "    query = f\"SELECT id, pass_hash FROM accounts WHERE username = '{username}'\"\n"
        "    return query, ()\n"
    )
    req_sql = LLMRequest(messages=[Message(role="user", content=sql_prompt)], data_class="public")
    resp_sql = asyncio.run(router.generate(req_sql))
    assert resp_sql.provider == "gemini"
    assert router.last_decision is not None
    assert router.last_decision.route_target == "frontier"
    assert "sql_injection" in router.last_decision.decision_reason
    assert len(ollama_calls) == 0
    assert len(gemini_calls) == 2


def test_heuristic_trigger_high_ast_complexity_routes_to_gemini() -> None:
    """Tasks exceeding the AST node complexity threshold escalate to Gemini."""
    ollama_calls: list[httpx.Request] = []
    gemini_calls: list[httpx.Request] = []

    router = _make_router_with_transports(
        lambda req: (ollama_calls.append(req), httpx.Response(200, json=_ollama_response_json()))[
            1
        ],
        lambda req: (gemini_calls.append(req), httpx.Response(200, json=_gemini_response_json()))[
            1
        ],
        policy=StaticRulePolicy(ast_node_threshold=60),
    )

    lines = [
        "--- File: complex_engine.py ---",
        "def compute_matrix(items: list[int]) -> int:",
        "    total = 0",
    ]
    for i in range(20):
        lines.append(f"    val_{i} = (items[{i} % len(items)] + {i}) * 2")
        lines.append(f"    total += val_{i}")
    lines.append("    return total")

    req = LLMRequest(messages=[Message(role="user", content="\n".join(lines))], data_class="public")
    feats = extract_task_features(req)
    assert feats.ast_node_count >= 60

    resp = asyncio.run(router.generate(req))
    assert resp.provider == "gemini"
    assert len(gemini_calls) == 1
    assert len(ollama_calls) == 0
    assert router.last_decision is not None
    assert "static_ast_complexity" in router.last_decision.decision_reason


def test_cascade_policy_escalates_on_retry_feedback() -> None:
    """CascadeConfidencePolicy routes attempt 1 to Ollama and escalates attempt 2 to Gemini."""
    ollama_calls: list[httpx.Request] = []
    gemini_calls: list[httpx.Request] = []

    router = _make_router_with_transports(
        lambda req: (ollama_calls.append(req), httpx.Response(200, json=_ollama_response_json()))[
            1
        ],
        lambda req: (gemini_calls.append(req), httpx.Response(200, json=_gemini_response_json()))[
            1
        ],
        policy=CascadeConfidencePolicy(escalate_on_retry=True),
    )

    attempt_1_prompt = (
        "You are PatchPilot, an autonomous automated code repair agent.\n"
        "--- File: simple_math.py ---\n"
        "def add_bonus(x: int) -> int:\n"
        "    return x - 5\n"
    )
    resp1 = asyncio.run(
        router.generate(
            LLMRequest(
                messages=[Message(role="user", content=attempt_1_prompt)], data_class="public"
            )
        )
    )
    assert resp1.provider == "ollama"
    assert router.last_decision is not None
    assert router.last_decision.route_target == "local"

    attempt_2_prompt = (
        f"{attempt_1_prompt}\n"
        "PREVIOUS ATTEMPT FAILED WITH TEST OUTPUT / ERROR:\n"
        "Attempt 1 verification failed:\n"
        "AssertionError: assert 5 == 15\n"
    )
    resp2 = asyncio.run(
        router.generate(
            LLMRequest(
                messages=[Message(role="user", content=attempt_2_prompt)], data_class="public"
            )
        )
    )
    assert resp2.provider == "gemini"
    assert router.last_decision is not None
    assert router.last_decision.route_target == "frontier"
    assert "cascade_retry_escalation:attempt=2" in router.last_decision.decision_reason


def test_fallback_on_gemini_429_rate_limit_to_ollama_with_warning() -> None:
    """Simulated Gemini HTTP 429 falls back gracefully to Ollama and sets warning_flag."""
    ollama_calls: list[httpx.Request] = []
    gemini_calls: list[httpx.Request] = []

    def gemini_429_handler(req: httpx.Request) -> httpx.Response:
        gemini_calls.append(req)
        return httpx.Response(429, headers={"Retry-After": "0.0"}, json={"error": "quota exceeded"})

    router = _make_router_with_transports(
        lambda req: (
            ollama_calls.append(req),
            httpx.Response(200, json=_ollama_response_json("FALLBACK_OK")),
        )[1],
        gemini_429_handler,
    )

    weak_spot_prompt = (
        "--- File: login_query_999.py ---\n"
        "def build_login_query(username: str):\n"
        "    return f\"SELECT * FROM accounts WHERE username = '{username}'\", ()\n"
    )
    req = LLMRequest(messages=[Message(role="user", content=weak_spot_prompt)], data_class="public")
    resp = asyncio.run(router.generate(req))

    assert resp.provider == "ollama"
    assert resp.text == "FALLBACK_OK"
    assert len(gemini_calls) == 2  # initial + 1 retry before exhaustion
    assert len(ollama_calls) == 1

    dec = router.last_decision
    assert dec is not None
    assert dec.initial_target == "frontier"
    assert dec.route_target == "local"
    assert dec.fallback_used is True
    assert "rate_limit_429" in dec.warning_flag
    assert dec.estimated_cost_usd == 0.0


def test_fallback_on_gemini_network_timeout_to_ollama() -> None:
    """Simulated Gemini network timeout falls back to local Ollama without breaking execution."""
    ollama_calls: list[httpx.Request] = []

    def gemini_timeout_handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated gemini network timeout", request=req)

    router = _make_router_with_transports(
        lambda req: (
            ollama_calls.append(req),
            httpx.Response(200, json=_ollama_response_json("TIMEOUT_RECOVERED")),
        )[1],
        gemini_timeout_handler,
    )

    req = LLMRequest(
        messages=[
            Message(
                role="user",
                content=(
                    "--- File: user_fmt_777.py ---\n"
                    "def format_user_display_name(profile: dict) -> str:\n"
                    "    return profile['full_name'].strip()\n"
                ),
            )
        ],
        data_class="public",
    )
    resp = asyncio.run(router.generate(req))
    assert resp.provider == "ollama"
    assert resp.text == "TIMEOUT_RECOVERED"
    assert router.last_decision is not None
    assert router.last_decision.fallback_used is True
    assert "network_timeout" in router.last_decision.warning_flag


def test_router_pipeline_integration_and_cost_latency_metrics() -> None:
    """End-to-end LangGraph pipeline execution using ModelRouter and trace propagation."""
    files = {
        "user_fmt_500.py": (
            '"""User formatting module 500."""\n\n\n'
            "def format_user_display_name(profile: dict) -> str:\n"
            "    # Missing None check on profile['full_name']\n"
            '    return profile["full_name"].strip()\n'
        ),
        "test_user_fmt_500.py": (
            "from user_fmt_500 import format_user_display_name\n\n"
            "def test_format_display_name_none():\n"
            '    p = {"username": "jdoe", "full_name": None}\n'
            '    assert format_user_display_name(p) == "jdoe"\n'
        ),
    }
    valid_search_replace = (
        "<<<<<<< SEARCH:user_fmt_500.py\n"
        "def format_user_display_name(profile: dict) -> str:\n"
        "    # Missing None check on profile['full_name']\n"
        '    return profile["full_name"].strip()\n'
        "=======\n"
        "def format_user_display_name(profile: dict) -> str:\n"
        '    name = profile.get("full_name")\n'
        "    if name is not None:\n"
        "        return name.strip()\n"
        '    return profile.get("username", "Guest").strip()\n'
        ">>>>>>> REPLACE\n"
    )

    local_mock = MockProvider(canned=valid_search_replace)
    frontier_mock = MockProvider(canned=valid_search_replace)
    router = ModelRouter(local_mock, frontier_mock)

    async def fake_verify(fs: dict[str, str], patch: str) -> SandboxResult:
        return SandboxResult(
            exit_code=0,
            patch_applied=bool(patch.strip()),
            junit=JUnitReport(passed=1, failures=0, errors=0),
        )

    tracer = PatchPilotTracer(
        settings=LangfuseSettings(host="", public_key="", secret_key=""),
        enabled=False,
    )
    state = PipelineState(
        case_id="seeded_train/test_router_case",
        files=files,
        data_class="public",
        provider="router",
    )
    final_state = asyncio.run(
        run_pipeline(
            state,
            provider=router,
            verify_fn=fake_verify,
            tracer=tracer,
        )
    )

    assert final_state.verified is True
    assert final_state.pass_at_1 is True
    assert len(router.decisions) == 2  # analyze + propose
    assert all(c.route_target in ("local", "frontier") for c in final_state.llm_calls)
    assert all(c.decision_reason != "" for c in final_state.llm_calls)

    # Also record a simple local call to verify mixed cost savings calculation
    simple_req = LLMRequest(
        messages=[Message(role="user", content="def f(x): return x + 1")],
        data_class="public",
    )
    asyncio.run(router.generate(simple_req))

    summary = router.get_metrics_summary(baseline_ollama_latency_s=0.75)
    assert summary.total_calls == 3
    assert summary.local_calls == 1
    assert summary.frontier_calls == 2
    assert summary.all_gemini_cost_usd > summary.router_total_cost_usd > 0.0
    assert summary.cost_savings_usd > 0.0
    assert 0.0 < summary.cost_savings_pct < 100.0


def test_router_smoke_benchmark_10_seeded_train_cases() -> None:
    """Smoke benchmark on 10 `seeded_train` cases comparing Ollama vs. Router vs. Gemini."""
    report = asyncio.run(run_router_smoke_benchmark(limit=10))

    assert report.sample_size == 10
    assert len(report.case_ids) == 10
    # Router matches Gemini-only accuracy (10/10 = 100%) while beating Ollama-only on weak spots
    assert report.router.pass_at_1_count == 10
    assert report.gemini_only.pass_at_1_count == 10
    assert report.router.pass_at_1_rate > report.ollama_only.pass_at_1_rate
    # Router uses local Ollama for most calls and escalates only weak-spot cases to Gemini
    assert report.router.local_calls > 0
    assert report.router.frontier_calls > 0
    assert report.router_metrics.cost_savings_pct > 50.0
    assert report.router.estimated_cost_usd < report.gemini_only.estimated_cost_usd
    assert report.router.simulated_latency_s < report.gemini_only.simulated_latency_s
