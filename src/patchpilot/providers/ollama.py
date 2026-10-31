"""Ollama provider. Ollama runs natively on the Windows host, not in Docker."""

from __future__ import annotations

import asyncio

import httpx

from patchpilot.config import Settings
from patchpilot.providers.base import (
    LLMRequest,
    LLMResponse,
    ProviderError,
    SleepFn,
    Usage,
    post_json_with_retry,
)


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        default_model: str,
        timeout_s: float | None = None,
        *,
        max_attempts: int | None = None,
        retry_base_delay_s: float | None = None,
        retry_max_delay_s: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        cfg = Settings()
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
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

    async def generate(self, request: LLMRequest) -> LLMResponse:
        model = (request.model or self.default_model).strip()
        if not model:
            raise ProviderError("ollama: model name must not be empty")
        if not self.base_url:
            raise ProviderError("ollama: base_url must not be empty")
        if not request.messages:
            raise ProviderError("ollama: request.messages must not be empty")

        options: dict[str, object] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": options,
        }

        data = await post_json_with_retry(
            provider_name=self.name,
            url=f"{self.base_url}/api/chat",
            payload=payload,
            timeout_s=self.timeout_s,
            max_attempts=self.max_attempts,
            retry_base_delay_s=self.retry_base_delay_s,
            retry_max_delay_s=self.retry_max_delay_s,
            transport=self._transport,
            sleep=self._sleep,
        )

        msg = data.get("message")
        if not isinstance(msg, dict):
            raise ProviderError("ollama: malformed response (missing 'message' object)")
        content = msg.get("content")
        if not isinstance(content, str):
            raise ProviderError("ollama: malformed response (missing string 'message.content')")

        raw_prompt = data.get("prompt_eval_count", 0)
        raw_completion = data.get("eval_count", 0)
        prompt_tokens = int(raw_prompt) if isinstance(raw_prompt, int | float) else 0
        completion_tokens = int(raw_completion) if isinstance(raw_completion, int | float) else 0
        resp_model = data.get("model")

        return LLMResponse(
            text=content,
            model=resp_model if isinstance(resp_model, str) and resp_model else model,
            provider=self.name,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )
