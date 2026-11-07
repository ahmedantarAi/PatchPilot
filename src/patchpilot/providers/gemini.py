"""Gemini API provider, free tier. Never send non-public code (SPEC §9)."""

from __future__ import annotations

import asyncio

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

GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com"


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        default_model: str,
        timeout_s: float | None = None,
        *,
        base_url: str = GEMINI_API_BASE_URL,
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
        return f"GeminiProvider(default_model={self.default_model!r}, api_key='***')"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if request.data_class != "public":
            raise PrivacyViolationError(
                f"gemini: refusing request with data_class={request.data_class!r}; "
                "only 'public' is allowed (SPEC §9)"
            )

        model = (request.model or self.default_model).strip()
        if not model:
            raise ProviderError("gemini: model name is empty; set PP_GEMINI_MODEL in .env")

        api_key = self._api_key.strip()
        if not api_key:
            raise ProviderError("gemini: API key is empty; set PP_GEMINI_API_KEY in .env")

        if not request.messages:
            raise ProviderError("gemini: request.messages must not be empty")

        system_parts: list[dict[str, str]] = []
        contents: list[dict[str, object]] = []
        for msg in request.messages:
            if msg.role == "system":
                system_parts.append({"text": msg.content})
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        if not contents and system_parts:
            contents.append({"role": "user", "parts": system_parts})
            system_parts = []

        gen_config: dict[str, object] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            gen_config["maxOutputTokens"] = request.max_tokens

        payload: dict[str, object] = {
            "contents": contents,
            "generationConfig": gen_config,
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = f"{self.base_url}/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": api_key}

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

        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderError("gemini: malformed response (missing non-empty 'candidates')")
        first = candidates[0]
        if not isinstance(first, dict):
            raise ProviderError("gemini: malformed response (candidate is not an object)")
        content_obj = first.get("content")
        if not isinstance(content_obj, dict):
            raise ProviderError("gemini: malformed response (missing 'candidate.content')")
        parts = content_obj.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ProviderError("gemini: malformed response (missing 'candidate.content.parts')")

        text_chunks: list[str] = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_chunks.append(part["text"])
        if not text_chunks:
            raise ProviderError("gemini: malformed response (no text parts in candidate)")

        usage_meta = data.get("usageMetadata")
        prompt_tokens = 0
        completion_tokens = 0
        if isinstance(usage_meta, dict):
            raw_prompt = usage_meta.get("promptTokenCount", 0)
            raw_completion = usage_meta.get("candidatesTokenCount", 0)
            if isinstance(raw_prompt, int | float):
                prompt_tokens = int(raw_prompt)
            if isinstance(raw_completion, int | float):
                completion_tokens = int(raw_completion)

        model_version = data.get("modelVersion")
        return LLMResponse(
            text="".join(text_chunks),
            model=model_version if isinstance(model_version, str) and model_version else model,
            provider=self.name,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )
