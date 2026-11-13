"""Tests for OpenRouter/retry error guards, embedder selection, and router eval telemetry."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from patchpilot.config import LangfuseSettings, Settings
from patchpilot.evals.runner import (
    EvalCaseRunResult,
    EvalRunReport,
    _select_embedder,
    build_eval_summary,
    format_eval_summary,
    run_eval_pool,
)
from patchpilot.pipeline.models import PipelineState
from patchpilot.pipeline.nodes import compute_prompt_version_hash
from patchpilot.providers.base import (
    LLMRequest,
    Message,
    PrivacyViolationError,
    ProviderError,
    post_json_with_retry,
)
from patchpilot.providers.mock import MockProvider
from patchpilot.providers.openrouter import OpenRouterProvider
from patchpilot.retrieval import HashEmbedder, OllamaEmbedder
from patchpilot.router.policy import StaticRulePolicy
from patchpilot.router.router import ModelRouter
from patchpilot.sandbox.runner import JUnitReport, SandboxResult
from patchpilot.tracing import LLMCallMetric


async def _no_sleep(_: float) -> None:
    return None


# ---------------------------------------------------------------------------
# 1. OpenRouter finish_reason & empty-content guards + privacy/key redaction
# ---------------------------------------------------------------------------


def test_openrouter_rejects_length_truncation_finish_reason() -> None:
    """OpenRouterProvider raises ProviderError when choices[0].finish_reason == 'length'."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-len-1",
                "model": "google/gemma-4-31b-it:free",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "partial truncated output"},
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 100},
            },
        )

    provider = OpenRouterProvider(
        api_key="sk-or-v1-secret-test-key",
        default_model="google/gemma-4-31b-it:free",
        transport=httpx.MockTransport(handler),
        sleep=_no_sleep,
    )
    req = LLMRequest(messages=[Message(role="user", content="fix bug")], data_class="public")

    with pytest.raises(ProviderError, match="finish_reason='length'"):
        asyncio.run(provider.generate(req))


@pytest.mark.parametrize("empty_content", ["", "   ", "\n\t  \n"])
def test_openrouter_rejects_empty_content_after_stripping(empty_content: str) -> None:
    """OpenRouterProvider raises ProviderError when content is empty after stripping."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-empty-1",
                "model": "google/gemma-4-31b-it:free",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": empty_content},
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 0},
            },
        )

    provider = OpenRouterProvider(
        api_key="sk-or-v1-secret-test-key",
        default_model="google/gemma-4-31b-it:free",
        transport=httpx.MockTransport(handler),
        sleep=_no_sleep,
    )
    req = LLMRequest(messages=[Message(role="user", content="fix bug")], data_class="public")

    with pytest.raises(ProviderError, match="empty response content"):
        asyncio.run(provider.generate(req))


def test_openrouter_preserves_privacy_violation_and_repr_redaction() -> None:
    """Private data_class raises PrivacyViolationError and repr redacts API key."""
    calls: list[httpx.Request] = []
    provider = OpenRouterProvider(
        api_key="sk-or-v1-top-secret-987654",
        default_model="google/gemma-4-31b-it:free",
        transport=httpx.MockTransport(lambda req: (calls.append(req), httpx.Response(200))[1]),
        sleep=_no_sleep,
    )
    assert "sk-or-v1-top-secret-987654" not in repr(provider)
    assert "api_key='***'" in repr(provider)

    private_req = LLMRequest(messages=[Message(role="user", content="secret code")])
    with pytest.raises(PrivacyViolationError, match="public"):
        asyncio.run(provider.generate(private_req))
    assert calls == []


# ---------------------------------------------------------------------------
# 2. post_json_with_retry HTTP error body snippet + secret redaction
# ---------------------------------------------------------------------------


def test_post_json_with_retry_includes_truncated_redacted_body_on_400_and_429() -> None:
    """400 and exhausted 429 errors include truncated body snippets with secrets redacted."""
    secret_bearer = "sk-or-v1-supersecretkeyvalue12345"
    secret_goog = "AIzaSySecretGeminiKeyValue99999"
    secret_url_tok = "url-secret-token-7777"

    # Case 1: 400 echoing Authorization header secret and long body
    long_reason = "x" * 350
    body_400 = (
        f'{{"error":{{"message":"Invalid key Bearer {secret_bearer} '
        f'and url token {secret_url_tok}: {long_reason}"}}}}'
    )

    def handler_400(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=body_400)

    with pytest.raises(ProviderError) as exc_400:
        asyncio.run(
            post_json_with_retry(
                provider_name="openrouter",
                url=f"https://openrouter.ai/api/v1/chat/completions?token={secret_url_tok}",
                payload={"model": "test"},
                headers={"Authorization": f"Bearer {secret_bearer}"},
                timeout_s=5.0,
                max_attempts=2,
                retry_base_delay_s=0.01,
                retry_max_delay_s=0.05,
                transport=httpx.MockTransport(handler_400),
                sleep=_no_sleep,
            )
        )

    msg_400 = str(exc_400.value)
    assert "non-retryable HTTP 400" in msg_400
    assert "Invalid key" in msg_400
    assert msg_400.endswith("...")
    assert secret_bearer not in msg_400
    assert secret_url_tok not in msg_400
    assert "***" in msg_400

    # Case 2: 429 exhausted retries echoing x-goog-api-key secret
    def handler_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "0"},
            json={
                "error": {
                    "code": 429,
                    "message": f"Upstream :free pool rate limit for key {secret_goog}",
                }
            },
        )

    with pytest.raises(ProviderError) as exc_429:
        asyncio.run(
            post_json_with_retry(
                provider_name="gemini",
                url="https://generativelanguage.googleapis.com/v1beta/models/m:generateContent",
                payload={"contents": []},
                headers={"x-goog-api-key": secret_goog},
                timeout_s=5.0,
                max_attempts=2,
                retry_base_delay_s=0.01,
                retry_max_delay_s=0.05,
                transport=httpx.MockTransport(handler_429),
                sleep=_no_sleep,
            )
        )

    msg_429 = str(exc_429.value)
    assert "HTTP 429 after 2 attempt(s)" in msg_429
    assert "Upstream :free pool rate limit" in msg_429
    assert secret_goog not in msg_429
    assert "***" in msg_429


# ---------------------------------------------------------------------------
# 3. Embedder selection in evals/runner.py + Qdrant-down fallback
# ---------------------------------------------------------------------------


def test_select_embedder_uses_ollama_for_live_providers_and_hash_only_for_mock() -> None:
    """OllamaEmbedder is used for ollama, openrouter, and router; HashEmbedder only for mock."""
    settings = Settings(_env_file=None, ollama_base_url="http://localhost:11434")

    assert isinstance(_select_embedder("mock", settings), HashEmbedder)
    for live_provider in ("ollama", "openrouter", "router"):
        emb = _select_embedder(live_provider, settings)
        assert isinstance(emb, OllamaEmbedder)
        assert emb.base_url == "http://localhost:11434"


def test_run_eval_pool_passes_expected_embedder_and_handles_qdrant_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_eval_pool wires OllamaEmbedder for openrouter/router/ollama and HashEmbedder for mock."""
    recorded_embedders: dict[str, object] = {}

    async def fake_verify(fs: dict[str, str], patch: str) -> SandboxResult:
        return SandboxResult(
            exit_code=0,
            patch_applied=True,
            junit=JUnitReport(passed=1, failures=0, errors=0),
        )

    from patchpilot.evals import runner as eval_runner

    orig_run_pipeline = eval_runner.run_pipeline

    async def spy_run_pipeline(
        initial_state: PipelineState,
        *,
        provider: Any,
        sandbox_runner: Any = None,
        verify_fn: Any = None,
        qdrant: Any = None,
        embedder: Any = None,
        tracer: Any = None,
    ) -> PipelineState:
        recorded_embedders[initial_state.provider] = embedder
        # Point Qdrant at an unused closed port to verify graceful Qdrant-down fallback
        closed_qdrant = eval_runner.QdrantStore(base_url="http://127.0.0.1:59999", timeout_s=0.1)
        return await orig_run_pipeline(
            initial_state,
            provider=provider,
            sandbox_runner=sandbox_runner,
            verify_fn=verify_fn,
            qdrant=closed_qdrant,
            embedder=embedder,
            tracer=tracer,
        )

    monkeypatch.setattr(eval_runner, "run_pipeline", spy_run_pipeline)

    canned_patch = (
        "<<<<<<< SEARCH:off_by_one.py\n    return n - 1\n=======\n    return n + 1\n>>>>>>> REPLACE"
    )
    dataset_root = tmp_path / "datasets"
    case_repo = dataset_root / "mini_pool" / "0001" / "repo"
    case_repo.mkdir(parents=True)
    (case_repo / "off_by_one.py").write_text(
        "def inc(n: int) -> int:\n    return n - 1\n",
        encoding="utf-8",
    )
    (case_repo / "test_off_by_one.py").write_text(
        "from off_by_one import inc\ndef test_inc():\n    assert inc(1) == 2\n",
        encoding="utf-8",
    )
    (dataset_root / "mini_pool" / "0001" / "patch.diff").write_text(
        canned_patch,
        encoding="utf-8",
    )

    async def _run_all() -> None:
        for prov_name in ("mock", "ollama", "openrouter", "router"):
            reports, _ = await run_eval_pool(
                pool="mini_pool",
                provider_name=prov_name,
                runs=1,
                seed=1337,
                limit=1,
                datasets_dir=dataset_root,
                results_dir=tmp_path / f"results_{prov_name}",
                provider_override=None
                if prov_name == "mock"
                else MockProvider(canned=canned_patch),
                verify_fn=fake_verify,
                langfuse_settings=LangfuseSettings(_env_file=None),
            )
            assert reports[0].total_cases == 1

    asyncio.run(_run_all())

    assert isinstance(recorded_embedders["mock"], HashEmbedder)
    assert isinstance(recorded_embedders["ollama"], OllamaEmbedder)
    assert isinstance(recorded_embedders["openrouter"], OllamaEmbedder)
    assert isinstance(recorded_embedders["router"], OllamaEmbedder)


# ---------------------------------------------------------------------------
# 4. EvalRunReport & format_eval_summary aggregation of router/fallback telemetry
# ---------------------------------------------------------------------------


def test_eval_report_backward_compatible_defaults_and_summary_aggregation() -> None:
    """Legacy JSON without router fields deserializes cleanly; summary aggregates telemetry."""
    prompt_hash = compute_prompt_version_hash()
    legacy_payload = {
        "run_id": "legacy_run",
        "run_index": 1,
        "timestamp": "2026-11-13T00:00:00Z",
        "pool": "seeded_train",
        "provider": "ollama",
        "model_name": "qwen2.5-coder:7b",
        "seed": 1337,
        "prompt_version_hash": prompt_hash,
        "total_cases": 1,
        "pass_at_1_count": 1,
        "pass_at_1_rate": 1.0,
        "pass_at_1_ci_95": [0.206549, 1.0],
        "pass_at_3_count": 1,
        "pass_at_3_rate": 1.0,
        "pass_at_3_ci_95": [0.206549, 1.0],
        "total_prompt_tokens": 100,
        "total_completion_tokens": 20,
        "total_tokens": 120,
        "total_time_s": 1.5,
        "cases": [
            {
                "case_id": "0001",
                "pool": "seeded_train",
                "run_index": 1,
                "seed": 1337,
                "provider": "ollama",
                "model_name": "qwen2.5-coder:7b",
                "prompt_version_hash": prompt_hash,
                "attempts": 1,
                "verified": True,
                "pass_at_1": True,
                "pass_at_3": True,
            }
        ],
    }
    legacy_report = EvalRunReport.model_validate_json(json.dumps(legacy_payload))
    assert legacy_report.local_calls == 0
    assert legacy_report.frontier_calls == 0
    assert legacy_report.fallback_count == 0
    assert legacy_report.warning_flags == []

    # Build two runs with LLMCallMetric router/fallback metadata
    run1 = EvalRunReport(
        run_id="router_run",
        run_index=1,
        timestamp="2026-11-13T00:00:00Z",
        pool="seeded_train",
        provider="router",
        model_name="router(qwen2.5-coder:7b+google/gemma-4-31b-it:free)",
        seed=1337,
        prompt_version_hash=prompt_hash,
        total_cases=1,
        pass_at_1_count=1,
        pass_at_1_rate=1.0,
        pass_at_1_ci_95=(0.2, 1.0),
        pass_at_3_count=1,
        pass_at_3_rate=1.0,
        pass_at_3_ci_95=(0.2, 1.0),
        total_prompt_tokens=200,
        total_completion_tokens=40,
        total_tokens=240,
        total_time_s=2.0,
        cases=[
            EvalCaseRunResult(
                case_id="0001",
                pool="seeded_train",
                run_index=1,
                seed=1337,
                provider="router",
                model_name="qwen2.5-coder:7b",
                prompt_version_hash=prompt_hash,
                attempts=1,
                verified=True,
                pass_at_1=True,
                pass_at_3=True,
                llm_calls=[
                    LLMCallMetric(
                        node="analyze",
                        model="google/gemma-4-31b-it:free",
                        provider="openrouter",
                        route_target="frontier",
                    ),
                    LLMCallMetric(
                        node="propose",
                        model="qwen2.5-coder:7b",
                        provider="ollama",
                        route_target="local",
                        fallback_used=True,
                        warning_flag="frontier_fallback:rate_limit_429:openrouter: HTTP 429",
                    ),
                ],
            )
        ],
    )
    run2 = EvalRunReport(
        run_id="router_run",
        run_index=2,
        timestamp="2026-11-13T00:01:00Z",
        pool="seeded_train",
        provider="router",
        model_name="router(qwen2.5-coder:7b+google/gemma-4-31b-it:free)",
        seed=1337,
        prompt_version_hash=prompt_hash,
        total_cases=1,
        pass_at_1_count=1,
        pass_at_1_rate=1.0,
        pass_at_1_ci_95=(0.2, 1.0),
        pass_at_3_count=1,
        pass_at_3_rate=1.0,
        pass_at_3_ci_95=(0.2, 1.0),
        total_prompt_tokens=200,
        total_completion_tokens=40,
        total_tokens=240,
        total_time_s=2.0,
        local_calls=1,
        frontier_calls=1,
        fallback_count=1,
        warning_flags=["frontier_fallback:rate_limit_429:openrouter: HTTP 429"],
        cases=[
            EvalCaseRunResult(
                case_id="0001",
                pool="seeded_train",
                run_index=2,
                seed=1337,
                provider="router",
                model_name="qwen2.5-coder:7b",
                prompt_version_hash=prompt_hash,
                attempts=1,
                verified=True,
                pass_at_1=True,
                pass_at_3=True,
            )
        ],
    )

    summary = build_eval_summary([run1, run2])
    assert summary.local_calls == 2
    assert summary.frontier_calls == 2
    assert summary.fallback_count == 2
    assert summary.warning_flags == ["frontier_fallback:rate_limit_429:openrouter: HTTP 429"]

    text = format_eval_summary(summary)
    assert "Routing telemetry: local_calls=2 | frontier_calls=2 | fallback_count=2" in text
    assert "Warning flags (1):" in text
    assert "frontier_fallback:rate_limit_429:openrouter: HTTP 429" in text


def test_run_eval_pool_records_openrouter_429_fallback_telemetry_in_report_and_summary(
    tmp_path: Path,
) -> None:
    """End-to-end eval run with ModelRouter hitting OpenRouter 429 records fallback telemetry."""
    secret_key = "sk-or-v1-router-eval-secret-key"
    canned_patch = (
        "<<<<<<< SEARCH:user_fmt.py\n"
        "    return profile['full_name'].strip()\n"
        "=======\n"
        "    name = profile.get('full_name')\n"
        "    return name.strip() if name is not None else profile.get('username', '')\n"
        ">>>>>>> REPLACE"
    )

    def openrouter_429_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "0"},
            json={
                "error": {
                    "message": f"Rate limit exceeded on :free upstream for Bearer {secret_key}"
                }
            },
        )

    frontier = OpenRouterProvider(
        api_key=secret_key,
        default_model="google/gemma-4-31b-it:free",
        max_attempts=1,
        transport=httpx.MockTransport(openrouter_429_handler),
        sleep=_no_sleep,
    )
    local = MockProvider(canned=canned_patch)
    # StaticRulePolicy routes none_handling (`profile_display_name` / `full_name`) to frontier,
    # triggering the 429 fallback to local on both analyze and propose.
    router = ModelRouter(local, frontier, policy=StaticRulePolicy())

    dataset_root = tmp_path / "datasets"
    case_dir = dataset_root / "seeded_train" / "0001"
    repo_dir = case_dir / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "user_fmt.py").write_text(
        "def format_user_display_name(profile: dict) -> str:\n"
        "    return profile['full_name'].strip()\n",
        encoding="utf-8",
    )
    (repo_dir / "test_user_fmt.py").write_text(
        "from user_fmt import format_user_display_name\n"
        "def test_none():\n"
        "    assert format_user_display_name({'username': 'jdoe', 'full_name': None}) == 'jdoe'\n",
        encoding="utf-8",
    )
    (case_dir / "meta.json").write_text(
        json.dumps({"taxonomy": "none_handling", "template": "profile_display_name"}),
        encoding="utf-8",
    )

    async def fast_verify(fs: dict[str, str], patch: str) -> SandboxResult:
        return SandboxResult(
            exit_code=0,
            patch_applied=True,
            junit=JUnitReport(passed=1, failures=0, errors=0),
        )

    reports, summary = asyncio.run(
        run_eval_pool(
            pool="seeded_train",
            provider_name="router",
            runs=1,
            seed=1337,
            limit=1,
            datasets_dir=dataset_root,
            results_dir=tmp_path / "results",
            provider_override=router,
            verify_fn=fast_verify,
            langfuse_settings=LangfuseSettings(_env_file=None),
        )
    )

    rep = reports[0]
    assert rep.local_calls == 2
    assert rep.frontier_calls == 0
    assert rep.fallback_count == 2
    assert len(rep.warning_flags) == 1
    assert "rate_limit_429" in rep.warning_flags[0]
    assert "Rate limit exceeded on :free upstream" in rep.warning_flags[0]
    assert secret_key not in rep.warning_flags[0]

    assert summary.local_calls == 2
    assert summary.frontier_calls == 0
    assert summary.fallback_count == 2
    assert summary.warning_flags == rep.warning_flags

    summary_text = format_eval_summary(summary)
    assert "Routing telemetry: local_calls=2 | frontier_calls=0 | fallback_count=2" in summary_text
    assert "rate_limit_429" in summary_text
    assert secret_key not in summary_text
