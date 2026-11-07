"""Phase 1 provider contract, privacy-guard, retry, and opt-in live tests."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from patchpilot.config import ProviderName, Settings
from patchpilot.providers import (
    GeminiProvider,
    LLMProvider,
    MockProvider,
    OllamaProvider,
    OpenRouterProvider,
    get_provider,
)
from patchpilot.providers.base import (
    LLMRequest,
    LLMResponse,
    Message,
    PrivacyViolationError,
    ProviderError,
)

PROVIDERS: tuple[ProviderName, ...] = ("mock", "ollama", "gemini", "openrouter")


def _ollama_ok_payload(
    text: str = "FIXED_PATCH",
    model: str = "qwen2.5-coder:7b",
) -> dict[str, object]:
    return {
        "model": model,
        "message": {"role": "assistant", "content": text},
        "done": True,
        "prompt_eval_count": 11,
        "eval_count": 7,
    }


def _gemini_ok_payload(
    text: str = "FIXED_PATCH",
    model: str = "gemini-test-model",
) -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": text}],
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 7,
            "totalTokenCount": 18,
        },
        "modelVersion": model,
    }


def _openrouter_ok_payload(
    text: str = "FIXED_PATCH",
    model: str = "google/gemma-4-31b-it:free",
) -> dict[str, object]:
    return {
        "id": "gen-test-123",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        },
    }


def _build_provider(
    name: ProviderName,
    handler: Callable[[httpx.Request], httpx.Response],
    sleeps: list[float],
    *,
    max_attempts: int = 3,
) -> LLMProvider:
    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    transport = httpx.MockTransport(handler)
    if name == "mock":
        return MockProvider(canned="FIXED_PATCH")
    if name == "ollama":
        return OllamaProvider(
            "http://localhost:11434",
            "qwen2.5-coder:7b",
            timeout_s=5.0,
            max_attempts=max_attempts,
            retry_base_delay_s=0.2,
            retry_max_delay_s=2.0,
            transport=transport,
            sleep=fake_sleep,
        )
    if name == "gemini":
        return GeminiProvider(
            "test-secret-api-key-do-not-leak",
            "gemini-test-model",
            timeout_s=5.0,
            max_attempts=max_attempts,
            retry_base_delay_s=0.2,
            retry_max_delay_s=2.0,
            transport=transport,
            sleep=fake_sleep,
        )
    if name == "openrouter":
        return OpenRouterProvider(
            "test-secret-api-key-do-not-leak",
            "google/gemma-4-31b-it:free",
            timeout_s=5.0,
            max_attempts=max_attempts,
            retry_base_delay_s=0.2,
            retry_max_delay_s=2.0,
            transport=transport,
            sleep=fake_sleep,
        )
    raise AssertionError(f"unexpected provider {name}")


@pytest.mark.parametrize("provider_name", PROVIDERS)
def test_contract_basic_generation(provider_name: ProviderName) -> None:
    calls: list[httpx.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if provider_name == "ollama":
            assert request.url.path == "/api/chat"
            return httpx.Response(200, json=_ollama_ok_payload())
        if provider_name == "openrouter":
            assert request.url.path == "/api/v1/chat/completions"
            assert "test-secret-api-key-do-not-leak" not in str(request.url)
            assert request.headers.get("authorization") == "Bearer test-secret-api-key-do-not-leak"
            return httpx.Response(200, json=_openrouter_ok_payload())
        assert "test-secret-api-key-do-not-leak" not in str(request.url)
        assert request.headers.get("x-goog-api-key") == "test-secret-api-key-do-not-leak"
        return httpx.Response(200, json=_gemini_ok_payload())

    provider = _build_provider(provider_name, handler, sleeps)
    assert isinstance(provider, LLMProvider)

    req = LLMRequest(
        messages=[Message(role="user", content="fix off by one bug")],
        data_class="public",
    )
    resp = asyncio.run(provider.generate(req))
    assert isinstance(resp, LLMResponse)
    assert resp.provider == provider_name
    assert resp.text == "FIXED_PATCH"
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0
    if provider_name != "mock":
        assert len(calls) == 1
        assert sleeps == []


@pytest.mark.parametrize("provider_name", PROVIDERS)
def test_contract_model_override_and_parameters(provider_name: ProviderName) -> None:
    captured_bodies: list[dict[str, object]] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_bodies.append(body)
        if provider_name == "ollama":
            return httpx.Response(200, json=_ollama_ok_payload(model="custom-model-v1"))
        if provider_name == "openrouter":
            return httpx.Response(200, json=_openrouter_ok_payload(model="custom-model-v1"))
        return httpx.Response(200, json=_gemini_ok_payload(model="custom-model-v1"))

    provider = _build_provider(provider_name, handler, sleeps)
    req = LLMRequest(
        messages=[
            Message(role="system", content="Return unified diff only."),
            Message(role="user", content="def add(a, b): return a - b"),
        ],
        model="custom-model-v1",
        temperature=0.25,
        max_tokens=64,
        data_class="public",
    )
    resp = asyncio.run(provider.generate(req))
    assert resp.model == "custom-model-v1"

    if provider_name == "mock":
        assert isinstance(provider, MockProvider)
        assert provider.calls[-1].model == "custom-model-v1"
        assert provider.calls[-1].max_tokens == 64
    elif provider_name == "ollama":
        body = captured_bodies[0]
        assert body["model"] == "custom-model-v1"
        assert body["stream"] is False
        assert body["options"] == {"temperature": 0.25, "num_predict": 64}
    elif provider_name == "gemini":
        body = captured_bodies[0]
        assert body["systemInstruction"] == {"parts": [{"text": "Return unified diff only."}]}
        assert body["generationConfig"] == {"temperature": 0.25, "maxOutputTokens": 64}
    elif provider_name == "openrouter":
        body = captured_bodies[0]
        assert body["model"] == "custom-model-v1"
        assert body["temperature"] == 0.25
        assert body["max_tokens"] == 64


@pytest.mark.parametrize("provider_name", PROVIDERS)
def test_contract_empty_messages_raises_provider_error(provider_name: ProviderName) -> None:
    calls: list[httpx.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    provider = _build_provider(provider_name, handler, sleeps)
    req = LLMRequest(messages=[], data_class="public")
    with pytest.raises(ProviderError):
        asyncio.run(provider.generate(req))
    assert calls == []


@pytest.mark.parametrize("provider_name", PROVIDERS)
def test_contract_retry_on_5xx_and_timeout(provider_name: ProviderName) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("simulated timeout", request=request)
        if attempts == 2:
            return httpx.Response(503, json={"error": "unavailable"})
        if provider_name == "ollama":
            return httpx.Response(200, json=_ollama_ok_payload())
        if provider_name == "openrouter":
            return httpx.Response(200, json=_openrouter_ok_payload())
        return httpx.Response(200, json=_gemini_ok_payload())

    provider = _build_provider(provider_name, handler, sleeps, max_attempts=3)
    req = LLMRequest(
        messages=[Message(role="user", content="check retry")],
        data_class="public",
    )
    resp = asyncio.run(provider.generate(req))
    assert resp.text == "FIXED_PATCH"

    if provider_name != "mock":
        assert attempts == 3
        assert len(sleeps) == 2
        assert 0.1 <= sleeps[0] <= 0.2
        assert 0.2 <= sleeps[1] <= 0.4


@pytest.mark.parametrize("provider_name", PROVIDERS)
def test_contract_retry_on_429_respects_retry_after(provider_name: ProviderName) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "1.75"},
                json={"error": "rate limited"},
            )
        if provider_name == "ollama":
            return httpx.Response(200, json=_ollama_ok_payload())
        if provider_name == "openrouter":
            return httpx.Response(200, json=_openrouter_ok_payload())
        return httpx.Response(200, json=_gemini_ok_payload())

    provider = _build_provider(provider_name, handler, sleeps, max_attempts=3)
    req = LLMRequest(
        messages=[Message(role="user", content="check 429 retry-after")],
        data_class="public",
    )
    resp = asyncio.run(provider.generate(req))
    assert resp.text == "FIXED_PATCH"

    if provider_name != "mock":
        assert attempts == 2
        assert sleeps == [1.75]


@pytest.mark.parametrize("provider_name", PROVIDERS)
def test_contract_no_retry_on_400(provider_name: ProviderName) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"error": "bad request"})

    provider = _build_provider(provider_name, handler, sleeps, max_attempts=3)
    if provider_name == "mock":
        bad_req = LLMRequest(messages=[], data_class="public")
        with pytest.raises(ProviderError):
            asyncio.run(provider.generate(bad_req))
    else:
        req = LLMRequest(
            messages=[Message(role="user", content="trigger 400")],
            data_class="public",
        )
        with pytest.raises(ProviderError) as exc_info:
            asyncio.run(provider.generate(req))
        assert "400" in str(exc_info.value)
        assert "test-secret-api-key-do-not-leak" not in str(exc_info.value)
        assert attempts == 1
        assert sleeps == []


def test_gemini_rejects_private_data_class() -> None:
    calls: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda req: (calls.append(req), httpx.Response(200))[1])
    provider = GeminiProvider(
        api_key="test-secret-key",
        default_model="gemini-test-model",
        transport=transport,
    )
    # Default data_class is "private" (fail-closed per SPEC §9 / D-20)
    private_req = LLMRequest(messages=[Message(role="user", content="private repo snippet")])
    assert private_req.data_class == "private"
    with pytest.raises(ProviderError, match="public"):
        asyncio.run(provider.generate(private_req))
    assert calls == []


def test_gemini_empty_model_fails_fast() -> None:
    calls: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda req: (calls.append(req), httpx.Response(200))[1])

    settings = Settings(gemini_api_key="test-secret-key", gemini_model="")
    provider = get_provider("gemini", settings=settings)
    assert isinstance(provider, GeminiProvider)
    assert "test-secret-key" not in repr(provider)

    provider_with_transport = GeminiProvider(
        api_key=settings.gemini_api_key,
        default_model=settings.gemini_model,
        transport=transport,
    )
    req = LLMRequest(
        messages=[Message(role="user", content="synthetic prompt")],
        data_class="public",
    )
    with pytest.raises(ProviderError, match="PP_GEMINI_MODEL"):
        asyncio.run(provider_with_transport.generate(req))

    empty_key_provider = GeminiProvider(
        api_key="   ",
        default_model="gemini-test-model",
        transport=transport,
    )
    with pytest.raises(ProviderError, match="PP_GEMINI_API_KEY"):
        asyncio.run(empty_key_provider.generate(req))

    assert calls == []


def test_settings_reads_gemini_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # 1. Reads from environment variable
    monkeypatch.setenv("PP_GEMINI_MODEL", "gemini-env-override")
    monkeypatch.setenv("PP_GEMINI_API_KEY", "env-key-123")
    settings = Settings()
    assert settings.gemini_model == "gemini-env-override"
    assert settings.gemini_api_key == "env-key-123"

    # 2. Reads from a temporary .env file
    monkeypatch.delenv("PP_GEMINI_MODEL", raising=False)
    monkeypatch.delenv("PP_GEMINI_API_KEY", raising=False)
    fake_env = tmp_path / ".env"
    fake_env.write_text(
        "PP_GEMINI_MODEL=gemini-file-model\nPP_GEMINI_API_KEY=file-key-456\n",
        encoding="utf-8",
    )
    file_settings = Settings(_env_file=fake_env)
    assert file_settings.gemini_model == "gemini-file-model"
    assert file_settings.gemini_api_key == "file-key-456"


def test_openrouter_rejects_private_data_class() -> None:
    calls: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda req: (calls.append(req), httpx.Response(200))[1])
    provider = OpenRouterProvider(
        api_key="test-secret-key",
        default_model="google/gemma-4-31b-it:free",
        transport=transport,
    )
    # Default data_class is "private" (fail-closed per SPEC §9 / D-14 / D-42)
    private_req = LLMRequest(messages=[Message(role="user", content="private repo snippet")])
    assert private_req.data_class == "private"
    with pytest.raises(PrivacyViolationError, match="public"):
        asyncio.run(provider.generate(private_req))
    assert calls == []


def test_openrouter_empty_model_and_key_fails_fast() -> None:
    calls: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda req: (calls.append(req), httpx.Response(200))[1])

    settings = Settings(openrouter_api_key="test-secret-key", openrouter_model="")
    provider = get_provider("openrouter", settings=settings)
    assert isinstance(provider, OpenRouterProvider)
    assert "test-secret-key" not in repr(provider)

    provider_with_transport = OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        default_model=settings.openrouter_model,
        transport=transport,
    )
    req = LLMRequest(
        messages=[Message(role="user", content="synthetic prompt")],
        data_class="public",
    )
    with pytest.raises(ProviderError, match="PP_OPENROUTER_MODEL"):
        asyncio.run(provider_with_transport.generate(req))

    empty_key_provider = OpenRouterProvider(
        api_key="   ",
        default_model="google/gemma-4-31b-it:free",
        transport=transport,
    )
    with pytest.raises(ProviderError, match="PP_OPENROUTER_API_KEY"):
        asyncio.run(empty_key_provider.generate(req))

    assert calls == []


def test_openrouter_redacts_api_key_in_repr() -> None:
    provider = OpenRouterProvider(
        api_key="sk-or-v1-secret-key-123456",
        default_model="google/gemma-4-31b-it:free",
    )
    assert "sk-or-v1-secret-key-123456" not in repr(provider)
    assert "api_key='***'" in repr(provider)


def test_settings_reads_openrouter_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # 1. Reads from environment variable
    monkeypatch.setenv("PP_OPENROUTER_MODEL", "google/gemma-4-31b-it")
    monkeypatch.setenv("PP_OPENROUTER_API_KEY", "env-key-or-123")
    monkeypatch.setenv("PP_FRONTIER_PROVIDER", "openrouter")
    settings = Settings()
    assert settings.openrouter_model == "google/gemma-4-31b-it"
    assert settings.openrouter_api_key == "env-key-or-123"
    assert settings.frontier_provider == "openrouter"

    # 2. Reads from a temporary .env file
    monkeypatch.delenv("PP_OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("PP_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PP_FRONTIER_PROVIDER", raising=False)
    fake_env = tmp_path / ".env"
    fake_env.write_text(
        "PP_OPENROUTER_MODEL=custom/gemma-file\nPP_OPENROUTER_API_KEY=file-key-or-456\nPP_FRONTIER_PROVIDER=openrouter\n",
        encoding="utf-8",
    )
    file_settings = Settings(_env_file=fake_env)
    assert file_settings.openrouter_model == "custom/gemma-file"
    assert file_settings.openrouter_api_key == "file-key-or-456"
    assert file_settings.frontier_provider == "openrouter"


@pytest.mark.live
def test_live_openrouter() -> None:
    if os.environ.get("PP_LIVE") != "1":
        pytest.skip("set PP_LIVE=1 to run live provider tests")
    settings = Settings()
    if not settings.openrouter_api_key:
        pytest.skip("PP_OPENROUTER_API_KEY not configured for live test")
    provider = OpenRouterProvider(
        settings.openrouter_api_key,
        settings.openrouter_model,
        settings.request_timeout_s,
    )
    req = LLMRequest(
        messages=[Message(role="user", content="Reply with the single word OK.")],
        max_tokens=16,
        data_class="public",
    )
    try:
        resp = asyncio.run(provider.generate(req))
        assert resp.provider == "openrouter"
        assert resp.text.strip() != ""
    except ProviderError as exc:
        if "429" in str(exc):
            pytest.skip(f"OpenRouter free pool temporarily rate-limited: {exc}")
        raise


@pytest.mark.live
def test_live_ollama() -> None:
    if os.environ.get("PP_LIVE") != "1":
        pytest.skip("set PP_LIVE=1 to run live provider tests")
    settings = Settings()
    provider = OllamaProvider(
        settings.ollama_base_url,
        "qwen2.5-coder:7b",
        settings.request_timeout_s,
    )
    req = LLMRequest(
        messages=[Message(role="user", content="Reply with the single word OK.")],
        max_tokens=16,
    )
    resp = asyncio.run(provider.generate(req))
    assert resp.provider == "ollama"
    assert resp.text.strip() != ""


@pytest.mark.live
def test_live_gemini() -> None:
    if os.environ.get("PP_LIVE") != "1":
        pytest.skip("set PP_LIVE=1 to run live provider tests")
    settings = Settings()
    provider = GeminiProvider(
        settings.gemini_api_key,
        settings.gemini_model,
        settings.request_timeout_s,
    )
    req = LLMRequest(
        messages=[Message(role="user", content="Reply with the single word OK.")],
        max_tokens=16,
        data_class="public",
    )
    resp = asyncio.run(provider.generate(req))
    assert resp.provider == "gemini"
    assert resp.text.strip() != ""
