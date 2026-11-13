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
    _extract_secrets_for_redaction,
    _redact_secrets,
    _sanitize_token_count,
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

    def __repr__(self) -> str:
        secrets = _extract_secrets_for_redaction(self.base_url, None)
        safe_url = _redact_secrets(self.base_url, secrets)
        safe_model = _redact_secrets(self.default_model, secrets)
        return f"OllamaProvider(base_url={safe_url!r}, default_model={safe_model!r})"

    def __str__(self) -> str:
        return self.__repr__()

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

        url = f"{self.base_url}/api/chat"
        data = await post_json_with_retry(
            provider_name=self.name,
            url=url,
            payload=payload,
            timeout_s=self.timeout_s,
            max_attempts=self.max_attempts,
            retry_base_delay_s=self.retry_base_delay_s,
            retry_max_delay_s=self.retry_max_delay_s,
            transport=self._transport,
            sleep=self._sleep,
        )

        if "error" in data and data["error"]:
            secrets = _extract_secrets_for_redaction(url, None)
            safe_err = _redact_secrets(str(data["error"]), secrets)
            raise ProviderError(f"ollama: error in response payload: {safe_err}")

        done_reason = data.get("done_reason")
        if isinstance(done_reason, str) and done_reason.strip().lower() in ("length", "max_tokens"):
            raise ProviderError(
                "ollama: response truncated (finish_reason='length'); increase max_tokens"
            )

        msg = data.get("message")
        if not isinstance(msg, dict):
            raise ProviderError("ollama: malformed response (missing 'message' object)")
        content = msg.get("content")
        if not isinstance(content, str):
            raise ProviderError("ollama: malformed response (missing string 'message.content')")
        if not content.strip():
            raise ProviderError("ollama: empty response content in 'message.content'")

        prompt_tokens = _sanitize_token_count(data.get("prompt_eval_count", 0))
        completion_tokens = _sanitize_token_count(data.get("eval_count", 0))
        resp_model = data.get("model")

        return LLMResponse(
            text=content,
            model=resp_model if isinstance(resp_model, str) and resp_model else model,
            provider=self.name,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )
