"""OpenRouter API provider for open-weights & frontier models (e.g. Gemma 4).

Adheres to SPEC §9 & DECISIONS.md D-04, D-14, D-42.
Never send non-public code (SPEC §9).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from patchpilot.config import Settings
from patchpilot.providers.base import (
    LLMRequest,
    LLMResponse,
    PrivacyViolationError,
    ProviderError,
    SleepFn,
    Usage,
    post_json_with_retry,
)

OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider:
    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        default_model: str,
        timeout_s: float | None = None,
        *,
        base_url: str = OPENROUTER_API_BASE_URL,
        max_attempts: int | None = None,
        retry_base_delay_s: float | None = None,
        retry_max_delay_s: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        cfg = Settings()
        self._api_key = api_key
        self.default_model = default_model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s if timeout_s is not None else cfg.request_timeout_s
        self.max_attempts = max_attempts if max_attempts is not None else cfg.max_attempts
        self.retry_base_delay_s = (
            retry_base_delay_s if retry_base_delay_s is not None else cfg.retry_base_delay_s
        )
        self.retry_max_delay_s = (
            retry_max_delay_s if retry_max_delay_s is not None else cfg.retry_max_delay_s
        )
        self._transport = transport
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"OpenRouterProvider(default_model={self.default_model!r}, api_key='***')"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if request.data_class != "public":
            raise PrivacyViolationError(
                f"openrouter: refusing request with data_class={request.data_class!r}; "
                "only 'public' is allowed (SPEC §9)"
            )

        model = (request.model or self.default_model).strip()
        if not model:
            raise ProviderError("openrouter: model name is empty; set PP_OPENROUTER_MODEL in .env")

        api_key = self._api_key.strip()
        if not api_key:
            raise ProviderError("openrouter: API key is empty; set PP_OPENROUTER_API_KEY in .env")

        if not request.messages:
            raise ProviderError("openrouter: request.messages must not be empty")

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "PatchPilot",
            "Content-Type": "application/json",
        }

        data = await post_json_with_retry(
            provider_name=self.name,
            url=url,
            payload=payload,
            headers=headers,
            timeout_s=self.timeout_s,
            max_attempts=self.max_attempts,
            retry_base_delay_s=self.retry_base_delay_s,
            retry_max_delay_s=self.retry_max_delay_s,
            transport=self._transport,
            sleep=self._sleep,
        )

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("openrouter: malformed response (missing or empty 'choices')")

        choice0 = choices[0]
        if not isinstance(choice0, dict):
            raise ProviderError("openrouter: malformed response ('choices[0]' is not an object)")

        finish_reason = choice0.get("finish_reason")
        if finish_reason == "length":
            raise ProviderError(
                "openrouter: response truncated (finish_reason='length'); increase max_tokens"
            )

        message_obj = choice0.get("message")
        if not isinstance(message_obj, dict):
            raise ProviderError("openrouter: malformed response (missing 'choices[0].message')")

        content = message_obj.get("content")
        if not isinstance(content, str):
            raise ProviderError(
                "openrouter: malformed response (missing string content in message)"
            )
        if not content.strip():
            raise ProviderError(
                "openrouter: empty response content in 'choices[0].message.content'"
            )

        raw_usage = data.get("usage")
        prompt_tokens = 0
        completion_tokens = 0
        if isinstance(raw_usage, dict):
            raw_prompt = raw_usage.get("prompt_tokens", 0)
            raw_comp = raw_usage.get("completion_tokens", 0)
            prompt_tokens = int(raw_prompt) if isinstance(raw_prompt, int | float) else 0
            completion_tokens = int(raw_comp) if isinstance(raw_comp, int | float) else 0

        resp_model = data.get("model")
        effective_model = resp_model if isinstance(resp_model, str) and resp_model else model

        return LLMResponse(
            text=content,
            model=effective_model,
            provider=self.name,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )
