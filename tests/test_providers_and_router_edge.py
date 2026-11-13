"""Edge-case unit tests for LLM providers and Dynamic Model Router.

Covers:
1. Strict `data_class == "public"` enforcement (`PrivacyViolationError` before any network call
   on `GeminiProvider` and `OpenRouterProvider`; `ModelRouter`, `StaticRulePolicy`, and
   `CascadeConfidencePolicy` forcing non-public requests to local `OllamaProvider` with
   `privacy_forced=True`).
2. Complete credential redaction (`***REDACTED***`) in `__repr__`, `__str__`, logs, and
   exception messages across all providers, router fallback paths, and helper functions.
3. Safe handling of malformed JSON responses, non-dict JSON payloads, empty choices/candidates,
   `finish_reason="length"` or safety/content-filter blocks, negative/`None`/`NaN`/`bool` token
   counts, HTTP 429/5xx retries, and clean fallback from frontier provider to local Ollama.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pytest

from patchpilot.providers import (
    GeminiProvider,
    MockProvider,
    ModelRouter,
    OllamaProvider,
    OpenRouterProvider,
    PrivacyViolationError,
)
from patchpilot.providers.base import (
    REDACTION_MASK,
    LLMRequest,
    LLMResponse,
    Message,
    ProviderError,
    Usage,
    _extract_secrets_for_redaction,
    _parse_retry_after,
    _redact_secrets,
    _sanitize_token_count,
    post_json_with_retry,
)
from patchpilot.router.features import (
    TaskFeatures,
    _compute_ast_stats,
    extract_task_features,
)
from patchpilot.router.metrics import (
    RoutingDecision,
    compute_router_metrics,
    estimate_call_cost_usd,
)
from patchpilot.router.policy import CascadeConfidencePolicy, StaticRulePolicy


async def _no_sleep(_: float) -> None:
    return None


def _ollama_ok(
    text: str = "LOCAL_OK",
    prompt_eval_count: Any = 12,
    eval_count: Any = 8,
) -> dict[str, Any]:
    return {
        "model": "qwen2.5-coder:7b",
        "message": {"role": "assistant", "content": text},
        "done": True,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


# ---------------------------------------------------------------------------
# 1. Strict `data_class == "public"` enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("non_public_class", ["private", "internal", "restricted", ""])
def test_gemini_and_openrouter_reject_non_public_before_network_call(
    non_public_class: str,
) -> None:
    """GeminiProvider and OpenRouterProvider raise PrivacyViolationError with 0 network calls."""
    gemini_calls: list[httpx.Request] = []
    openrouter_calls: list[httpx.Request] = []

    def record_gemini(req: httpx.Request) -> httpx.Response:
        gemini_calls.append(req)
        return httpx.Response(200)

    def record_openrouter(req: httpx.Request) -> httpx.Response:
        openrouter_calls.append(req)
        return httpx.Response(200)

    gemini = GeminiProvider(
        api_key="AIzaSyTopSecretGeminiKey123456",
        default_model="gemini-3.6-flash",
        transport=httpx.MockTransport(record_gemini),
        sleep=_no_sleep,
    )
    openrouter = OpenRouterProvider(
        api_key="sk-or-v1-topsecret-openrouter-key",
        default_model="google/gemma-4-31b-it:free",
        transport=httpx.MockTransport(record_openrouter),
        sleep=_no_sleep,
    )

    req = LLMRequest.model_construct(
        messages=[Message(role="user", content="def secret_logic(): pass")],
        model=None,
        temperature=0.0,
        max_tokens=64,
        data_class=non_public_class,
    )

    with pytest.raises(PrivacyViolationError, match="only 'public' is allowed"):
        asyncio.run(gemini.generate(req))
    with pytest.raises(PrivacyViolationError, match="only 'public' is allowed"):
        asyncio.run(openrouter.generate(req))

    assert gemini_calls == []
    assert openrouter_calls == []


def test_router_and_policies_force_non_public_requests_to_local_ollama() -> None:
    """ModelRouter, StaticRulePolicy, and CascadeConfidencePolicy force non-public to local."""
    ollama_calls: list[httpx.Request] = []
    frontier_calls: list[httpx.Request] = []

    def record_ollama(req: httpx.Request) -> httpx.Response:
        ollama_calls.append(req)
        return httpx.Response(200, json=_ollama_ok())

    def record_frontier(req: httpx.Request) -> httpx.Response:
        frontier_calls.append(req)
        return httpx.Response(200)

    local = OllamaProvider(
        "http://localhost:11434",
        "qwen2.5-coder:7b",
        transport=httpx.MockTransport(record_ollama),
        sleep=_no_sleep,
    )
    frontier = OpenRouterProvider(
        "sk-or-v1-frontier-secret-key",
        "google/gemma-4-31b-it:free",
        transport=httpx.MockTransport(record_frontier),
        sleep=_no_sleep,
    )
    router = ModelRouter(local, frontier)

    # Prompt combines weak spots + retry escalation that would route to frontier if public
    retry_weak_spot_prompt = (
        "--- File: auth_service.py ---\n"
        "def format_user_display_name(profile: dict) -> str:\n"
        "    return profile['full_name'].strip()\n"
        "def build_login_query(username: str):\n"
        "    return f\"SELECT * FROM users WHERE username = '{username}'\"\n"
        "PREVIOUS ATTEMPT FAILED WITH TEST OUTPUT / ERROR:\n"
        "Attempt 1 verification failed:\n"
        "AttributeError: 'NoneType' object has no attribute 'strip'\n"
    )
    private_req = LLMRequest(
        messages=[Message(role="user", content=retry_weak_spot_prompt)],
        data_class="private",
    )

    feats = extract_task_features(private_req)
    assert feats.is_retry is True
    assert len(feats.matched_weak_patterns) > 0

    static_eval = StaticRulePolicy().evaluate(feats)
    assert static_eval.target == "local"
    assert static_eval.privacy_forced is True

    cascade_eval = CascadeConfidencePolicy().evaluate(feats)
    assert cascade_eval.target == "local"
    assert cascade_eval.privacy_forced is True

    resp = asyncio.run(router.generate(private_req))
    assert resp.provider == "ollama"
    assert resp.text == "LOCAL_OK"
    assert len(ollama_calls) == 1
    assert frontier_calls == []
    assert router.last_decision is not None
    assert router.last_decision.route_target == "local"
    assert router.last_decision.privacy_forced is True


# ---------------------------------------------------------------------------
# 2. Complete credential redaction (***REDACTED***) in __repr__, __str__, logs, errors
# ---------------------------------------------------------------------------


def test_credential_redaction_in_repr_and_str_across_all_providers_and_router() -> None:
    """All providers and ModelRouter redact secrets in both __repr__ and __str__."""
    gemini_secret = "AIzaSySecretKeyForGemini987654321"
    openrouter_secret = "sk-or-v1-secretkeyforopenrouter987654"
    ollama_secret = "ollama_url_pass_secret_12345"
    mock_secret = "sk-or-v1-mockcannedsecret99999"

    gemini = GeminiProvider(api_key=gemini_secret, default_model="gemini-3.6-flash")
    openrouter = OpenRouterProvider(
        api_key=openrouter_secret,
        default_model="google/gemma-4-31b-it:free",
    )
    ollama = OllamaProvider(
        base_url=f"http://admin:{ollama_secret}@localhost:11434",
        default_model="qwen2.5-coder:7b",
    )
    mock = MockProvider(canned=f"Bearer {mock_secret}")
    router = ModelRouter(ollama, openrouter)

    for obj, secret in [
        (gemini, gemini_secret),
        (openrouter, openrouter_secret),
        (ollama, ollama_secret),
        (mock, mock_secret),
        (router, openrouter_secret),
        (router, ollama_secret),
    ]:
        r_text = repr(obj)
        s_text = str(obj)
        assert secret not in r_text
        assert secret not in s_text
        assert REDACTION_MASK in r_text
        assert REDACTION_MASK in s_text


def test_redact_secrets_helper_and_error_snippets_redact_all_token_formats() -> None:
    """_redact_secrets and post_json_with_retry redact headers, URLs, and body fields."""
    raw_key = "custom-header-secret-xyz987"
    bearer_tok = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    or_key = "sk-or-v1-abcdef0123456789"
    gem_key = "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz"
    gh_tok = "ghp_1234567890abcdefghijklmnopqrstuv"
    lf_tok = "sk-lf-9876543210abcdef"
    url_pw = "urlpassword999"
    query_sec = "querysecret888"

    url = f"https://user:{url_pw}@api.example.com/v1/chat?api_key={query_sec}"
    headers = {"X-Custom-Key": raw_key, "Authorization": f"Bearer {bearer_tok}"}
    extracted = _extract_secrets_for_redaction(url, headers)

    text = (
        f"Failed for {url} with header={raw_key}, auth=Bearer {bearer_tok}, "
        f"or={or_key}, gem={gem_key}, gh={gh_tok}, lf={lf_tok}, "
        f'json={{"api_key": "inline_json_secret_555"}}'
    )
    redacted = _redact_secrets(text, extracted)

    for secret in (
        raw_key,
        bearer_tok,
        or_key,
        gem_key,
        gh_tok,
        lf_tok,
        url_pw,
        query_sec,
        "inline_json_secret_555",
    ):
        assert secret not in redacted
    assert REDACTION_MASK in redacted


def test_router_fallback_redacts_secrets_in_logs_and_warning_flag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When frontier provider raises ProviderError with a leaked key, logs and warning redact it."""
    leaked_key = "sk-or-v1-leaked-in-exception-999999"

    class _LeakyFrontierProvider:
        name = "openrouter"
        default_model = "google/gemma-4-31b-it:free"
        _api_key = "custom-internal-key-777777"

        async def generate(self, request: LLMRequest) -> LLMResponse:
            raise ProviderError(
                f"openrouter: HTTP 429 rate limit for {leaked_key} and {self._api_key}"
            )

    local = MockProvider(canned="SAFE_LOCAL_FALLBACK")
    router = ModelRouter(local, _LeakyFrontierProvider(), policy=StaticRulePolicy())

    prompt = (
        "--- File: login_query.py ---\n"
        "def build_login_query(username: str):\n"
        "    return f\"SELECT * FROM users WHERE username = '{username}'\"\n"
    )
    req = LLMRequest(messages=[Message(role="user", content=prompt)], data_class="public")

    with caplog.at_level(logging.WARNING, logger="patchpilot.router"):
        resp = asyncio.run(router.generate(req))

    assert resp.text == "SAFE_LOCAL_FALLBACK"
    assert router.last_decision is not None
    assert router.last_decision.fallback_used is True
    assert leaked_key not in router.last_decision.warning_flag
    assert "custom-internal-key-777777" not in router.last_decision.warning_flag
    assert REDACTION_MASK in router.last_decision.warning_flag

    log_text = "\n".join(caplog.messages)
    assert leaked_key not in log_text
    assert "custom-internal-key-777777" not in log_text
    assert REDACTION_MASK in log_text


# ---------------------------------------------------------------------------
# 3. Malformed JSON, non-dict payloads, empty choices/candidates, safety & length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_body", ["{not-json", "", "[]", '"just_a_string"', "42", "null"])
def test_providers_reject_malformed_or_non_dict_json_responses(bad_body: str) -> None:
    """All HTTP providers raise ProviderError on invalid JSON or non-object JSON payloads."""
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            text=bad_body,
            headers={"content-type": "application/json"},
        )
    )
    ollama = OllamaProvider(
        "http://localhost:11434",
        "qwen2.5-coder:7b",
        transport=transport,
        sleep=_no_sleep,
    )
    gemini = GeminiProvider(
        "AIzaSyTestKey123456",
        "gemini-3.6-flash",
        transport=transport,
        sleep=_no_sleep,
    )
    openrouter = OpenRouterProvider(
        "sk-or-v1-testkey123456",
        "google/gemma-4-31b-it:free",
        transport=transport,
        sleep=_no_sleep,
    )

    req = LLMRequest(messages=[Message(role="user", content="hello")], data_class="public")
    for prov in (ollama, gemini, openrouter):
        with pytest.raises(ProviderError):
            asyncio.run(prov.generate(req))


@pytest.mark.parametrize(
    "gemini_payload",
    [
        {},
        {"candidates": []},
        {"candidates": ["not-a-dict"]},
        {"candidates": [{}]},
        {"candidates": [{"content": {}}]},
        {"candidates": [{"content": {"parts": []}}]},
        {"candidates": [{"content": {"parts": [{"not_text": 123}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "   \n\t  "}]}}]},
        {"promptFeedback": {"blockReason": "SAFETY"}},
        {
            "candidates": [
                {
                    "finishReason": "SAFETY",
                    "content": {"role": "model", "parts": [{"text": "unsafe partial"}]},
                }
            ]
        },
        {
            "candidates": [
                {
                    "finishReason": "RECITATION",
                    "content": {"role": "model", "parts": [{"text": "recited"}]},
                }
            ]
        },
        {
            "candidates": [
                {
                    "finishReason": "MAX_TOKENS",
                    "content": {"role": "model", "parts": [{"text": "truncated patch"}]},
                }
            ]
        },
        {"error": {"message": "Upstream error for AIzaSySecretInBody99999"}},
    ],
)
def test_gemini_edge_payloads_raise_provider_error_and_redact(
    gemini_payload: dict[str, Any],
) -> None:
    """GeminiProvider rejects empty candidates, safety blocks, MAX_TOKENS, and payload errors."""
    provider = GeminiProvider(
        api_key="AIzaSySecretInBody99999",
        default_model="gemini-3.6-flash",
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=gemini_payload)),
        sleep=_no_sleep,
    )
    req = LLMRequest(messages=[Message(role="user", content="fix")], data_class="public")
    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(provider.generate(req))
    assert "AIzaSySecretInBody99999" not in str(exc_info.value)


@pytest.mark.parametrize(
    "openrouter_payload",
    [
        {},
        {"choices": []},
        {"choices": ["not-a-dict"]},
        {"choices": [{"finish_reason": "length", "message": {"content": "truncated"}}]},
        {"choices": [{"native_finish_reason": "MAX_TOKENS", "message": {"content": "truncated"}}]},
        {"choices": [{"finish_reason": "content_filter", "message": {"content": "filtered"}}]},
        {"choices": [{"native_finish_reason": "SAFETY", "message": {"content": "blocked"}}]},
        {"choices": [{"error": {"message": "Mid-stream fail sk-or-v1-secret9999"}}]},
        {"choices": [{"message": "not-a-dict"}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": "   \n "}}]},
        {"error": {"message": "Provider error sk-or-v1-secret9999"}},
    ],
)
def test_openrouter_edge_payloads_raise_provider_error_and_redact(
    openrouter_payload: dict[str, Any],
) -> None:
    """OpenRouterProvider rejects empty choices, truncation, safety blocks, and payload errors."""
    provider = OpenRouterProvider(
        api_key="sk-or-v1-secret9999",
        default_model="google/gemma-4-31b-it:free",
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=openrouter_payload)),
        sleep=_no_sleep,
    )
    req = LLMRequest(messages=[Message(role="user", content="fix")], data_class="public")
    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(provider.generate(req))
    assert "sk-or-v1-secret9999" not in str(exc_info.value)


@pytest.mark.parametrize(
    "ollama_payload",
    [
        {},
        {"message": "not-a-dict"},
        {"message": {"content": None}},
        {"message": {"content": "   \t\n "}},
        {"done_reason": "length", "message": {"content": "truncated"}},
        {"error": "model not found"},
    ],
)
def test_ollama_edge_payloads_raise_provider_error(ollama_payload: dict[str, Any]) -> None:
    """OllamaProvider rejects missing message, empty content, length truncation, and error key."""
    provider = OllamaProvider(
        "http://localhost:11434",
        "qwen2.5-coder:7b",
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=ollama_payload)),
        sleep=_no_sleep,
    )
    req = LLMRequest(messages=[Message(role="user", content="fix")], data_class="private")
    with pytest.raises(ProviderError):
        asyncio.run(provider.generate(req))


# ---------------------------------------------------------------------------
# 4. Negative, None, NaN, inf, and boolean token counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_val", "expected"),
    [
        (None, 0),
        (-1, 0),
        (-999, 0),
        (0, 0),
        (15, 15),
        (15.9, 15),
        (True, 0),
        (False, 0),
        (float("nan"), 0),
        (float("inf"), 0),
        (float("-inf"), 0),
        ("42", 42),
        ("-10", 0),
        ("not-a-number", 0),
        (["list"], 0),
    ],
)
def test_sanitize_token_count_and_usage_model(raw_val: Any, expected: int) -> None:
    """_sanitize_token_count and Usage model clamp invalid/negative/None values to >= 0."""
    assert _sanitize_token_count(raw_val) == expected
    usage = Usage(prompt_tokens=raw_val, completion_tokens=raw_val)
    assert usage.prompt_tokens == expected
    assert usage.completion_tokens == expected


def test_providers_and_metrics_clamp_negative_and_none_token_counts() -> None:
    """Ollama, Gemini, OpenRouter, and router metrics safely clamp negative/None token counts."""
    ollama = OllamaProvider(
        "http://localhost:11434",
        "qwen2.5-coder:7b",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json=_ollama_ok("OK_OLLAMA", prompt_eval_count=-1, eval_count=None),
            )
        ),
        sleep=_no_sleep,
    )
    gemini = GeminiProvider(
        "AIzaSyValidKey123456",
        "gemini-3.6-flash",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"parts": [{"text": "OK_GEMINI"}]}}],
                    "usageMetadata": {"promptTokenCount": -50, "candidatesTokenCount": None},
                },
            )
        ),
        sleep=_no_sleep,
    )
    openrouter = OpenRouterProvider(
        "sk-or-v1-validkey123456",
        "google/gemma-4-31b-it:free",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "OK_OR"}}],
                    "usage": {"prompt_tokens": None, "completion_tokens": -25},
                },
            )
        ),
        sleep=_no_sleep,
    )

    req = LLMRequest(messages=[Message(role="user", content="test")], data_class="public")
    for prov in (ollama, gemini, openrouter):
        resp = asyncio.run(prov.generate(req))
        assert resp.usage.prompt_tokens == 0
        assert resp.usage.completion_tokens == 0

    assert estimate_call_cost_usd("frontier", None, -100) == 0.0
    dec = RoutingDecision(
        route_target="frontier",
        initial_target="frontier",
        selected_provider="gemini",
        selected_model="gemini-3.6-flash",
        decision_reason="test",
        prompt_tokens=-10,
        completion_tokens=None,  # type: ignore[arg-type]
    )
    assert dec.prompt_tokens == 0
    assert dec.completion_tokens == 0
    summary = compute_router_metrics([dec])
    assert summary.total_calls == 1
    assert summary.router_total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# 5. HTTP 429/5xx retries, Retry-After edge cases, and clean frontier fallback
# ---------------------------------------------------------------------------


def test_parse_retry_after_handles_edge_values() -> None:
    """_parse_retry_after handles numbers, HTTP dates, negative numbers, NaN, inf, and junk."""
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("") is None
    assert _parse_retry_after("   ") is None
    assert _parse_retry_after("2.5") == 2.5
    assert _parse_retry_after("0") == 0.0
    assert _parse_retry_after("-5") is None
    assert _parse_retry_after("nan") is None
    assert _parse_retry_after("inf") is None
    assert _parse_retry_after("-inf") is None
    assert _parse_retry_after("not-a-date-or-float") is None
    # Past HTTP-date clamps to 0.0
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0


def test_post_json_with_retry_invalid_max_attempts_raises() -> None:
    """post_json_with_retry raises ProviderError immediately when max_attempts < 1."""
    with pytest.raises(ProviderError, match="max_attempts must be >= 1"):
        asyncio.run(
            post_json_with_retry(
                provider_name="ollama",
                url="http://localhost:11434/api/chat",
                payload={},
                timeout_s=1.0,
                max_attempts=0,
                retry_base_delay_s=0.01,
                retry_max_delay_s=0.05,
                sleep=_no_sleep,
            )
        )


@pytest.mark.parametrize(
    ("frontier_Response_fn", "expected_warning_prefix"),
    [
        (
            lambda req: httpx.Response(429, json={"error": "rate limit"}),
            "frontier_fallback:rate_limit_429:",
        ),
        (
            lambda req: httpx.Response(503, json={"error": "service unavailable"}),
            "frontier_fallback:provider_error:",
        ),
        (
            lambda req: httpx.Response(
                200,
                json={"choices": [{"finish_reason": "length", "message": {"content": "cut"}}]},
            ),
            "frontier_fallback:provider_error:",
        ),
        (
            lambda req: httpx.Response(
                200,
                json={"choices": [{"finish_reason": "safety", "message": {"content": "blocked"}}]},
            ),
            "frontier_fallback:provider_error:",
        ),
        (
            lambda req: httpx.Response(200, text="{malformed json"),
            "frontier_fallback:provider_error:",
        ),
    ],
)
def test_router_clean_fallback_to_local_ollama_across_frontier_failures(
    frontier_Response_fn: Any,
    expected_warning_prefix: str,
) -> None:
    """ModelRouter cleanly falls back to local Ollama when frontier fails for any reason."""
    local = OllamaProvider(
        "http://localhost:11434",
        "qwen2.5-coder:7b",
        max_attempts=1,
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=_ollama_ok("RECOVERED"))
        ),
        sleep=_no_sleep,
    )
    frontier = OpenRouterProvider(
        "sk-or-v1-secret-frontier-key",
        "google/gemma-4-31b-it:free",
        max_attempts=1,
        transport=httpx.MockTransport(frontier_Response_fn),
        sleep=_no_sleep,
    )
    router = ModelRouter(local, frontier, policy=StaticRulePolicy())

    prompt = (
        "--- File: user_fmt.py ---\n"
        "def format_user_display_name(profile: dict) -> str:\n"
        "    return profile['full_name'].strip()\n"
    )
    req = LLMRequest(messages=[Message(role="user", content=prompt)], data_class="public")
    resp = asyncio.run(router.generate(req))

    assert resp.provider == "ollama"
    assert resp.text == "RECOVERED"
    assert router.last_decision is not None
    assert router.last_decision.initial_target == "frontier"
    assert router.last_decision.route_target == "local"
    assert router.last_decision.fallback_used is True
    assert router.last_decision.warning_flag.startswith(expected_warning_prefix)
    assert "sk-or-v1-secret-frontier-key" not in router.last_decision.warning_flag


# ---------------------------------------------------------------------------
# 6. Feature extraction edge cases (NUL bytes, syntax errors, deep ASTs)
# ---------------------------------------------------------------------------


def test_feature_extractor_handles_nul_bytes_syntax_errors_and_deep_ast() -> None:
    """_compute_ast_stats and extract_task_features handle invalid/deep source safely."""
    assert _compute_ast_stats("") == (0, 0, 0)
    assert _compute_ast_stats("def broken(:\n    pass") == (0, 0, 0)
    assert _compute_ast_stats("x = 1\x00y = 2") == (0, 0, 0)

    # Deeply nested binary expression (depth > 80) handled iteratively without RecursionError
    deep_expr = "x = " + " + ".join(["1"] * 85)
    nodes, depth, funcs = _compute_ast_stats(deep_expr)
    assert nodes > 85
    assert depth > 80
    assert funcs == 0

    req = LLMRequest(
        messages=[Message(role="user", content=f"--- File: deep.py ---\n{deep_expr}\n")],
        data_class="public",
    )
    feats: TaskFeatures = extract_task_features(req)
    assert feats.max_ast_depth > 80
    assert StaticRulePolicy(ast_depth_threshold=10).evaluate(feats).target == "frontier"
