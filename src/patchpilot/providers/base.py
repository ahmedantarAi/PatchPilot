"""Provider interface: every LLM backend implements `LLMProvider`."""

from __future__ import annotations

import asyncio
import email.utils
import random
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlsplit

import httpx
from pydantic import BaseModel, Field

DataClass = Literal["private", "public"]
SleepFn = Callable[[float], Awaitable[None]]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    messages: list[Message]
    model: str | None = None  # None -> provider default
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    data_class: DataClass = "private"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    model: str
    provider: str
    usage: Usage = Field(default_factory=Usage)


class ProviderError(RuntimeError):
    """Raised for provider failures (network, quota, malformed response)."""


class PrivacyViolationError(ProviderError):
    """Raised when private code is sent to an external/untrusted provider (SPEC §9)."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Return a completion for `request` or raise `ProviderError`."""
        ...


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    stripped = value.strip()
    try:
        seconds = float(stripped)
        if seconds >= 0.0:
            return seconds
    except ValueError:
        pass
    try:
        parsed_dt = email.utils.parsedate_to_datetime(stripped)
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=UTC)
        delta = (parsed_dt - datetime.now(UTC)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError, IndexError):
        return None


_MAX_ERROR_BODY_CHARS = 300
_SENSITIVE_NAME_TOKENS = (
    "authorization",
    "key",
    "token",
    "secret",
    "auth",
    "password",
    "credential",
)
_BEARER_PATTERN = re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._\-+/=]{4,}")
_OPENROUTER_KEY_PATTERN = re.compile(r"\bsk-or-v1-[A-Za-z0-9_\-]{4,}")
_GENERIC_SK_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}")
_GEMINI_KEY_PATTERN = re.compile(r"\bAIza[0-9A-Za-z_\-]{12,}")
_URL_PARAM_SECRET_PATTERN = re.compile(
    r"(?i)([?&](?:api_?key|key|token|access_token|secret|password)=)[^&\s\"'<>]+"
)
_BODY_FIELD_SECRET_PATTERN = re.compile(
    r"(?i)((?:\"(?:api_?key|apikey|x-goog-api-key|x-api-key|authorization|access_token|"
    r"secret_key|password)\"\s*:\s*\")|(?:\b(?:api_?key|apikey|x-goog-api-key|x-api-key|"
    r"access_token|secret_key|password)=))([^\"'\s,&]+)"
)
_URL_USERINFO_PATTERN = re.compile(r"(https?://[^:/@\s]+:)[^@\s]+(@)")


def _extract_secrets_for_redaction(
    url: str,
    headers: dict[str, str] | None,
) -> list[str]:
    secrets: set[str] = set()
    if headers:
        for key, value in headers.items():
            lower_key = key.lower()
            if any(tok in lower_key for tok in _SENSITIVE_NAME_TOKENS):
                stripped = value.strip()
                if len(stripped) >= 4:
                    secrets.add(stripped)
                parts = stripped.split(None, 1)
                if len(parts) == 2 and len(parts[1].strip()) >= 4:
                    secrets.add(parts[1].strip())

    try:
        parsed = urlsplit(url)
        if parsed.password and len(parsed.password) >= 4:
            secrets.add(parsed.password)
        for q_key, q_val in parse_qsl(parsed.query, keep_blank_values=False):
            if any(tok in q_key.lower() for tok in _SENSITIVE_NAME_TOKENS):
                q_stripped = q_val.strip()
                if len(q_stripped) >= 4:
                    secrets.add(q_stripped)
    except ValueError:
        pass

    return sorted(secrets, key=len, reverse=True)


def _redact_secrets(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        redacted = redacted.replace(secret, "***")
    redacted = _BEARER_PATTERN.sub(r"\1***", redacted)
    redacted = _OPENROUTER_KEY_PATTERN.sub("***", redacted)
    redacted = _GENERIC_SK_KEY_PATTERN.sub("***", redacted)
    redacted = _GEMINI_KEY_PATTERN.sub("***", redacted)
    redacted = _URL_PARAM_SECRET_PATTERN.sub(r"\1***", redacted)
    redacted = _BODY_FIELD_SECRET_PATTERN.sub(r"\1***", redacted)
    redacted = _URL_USERINFO_PATTERN.sub(r"\1***\2", redacted)
    return redacted


def _format_error_body_snippet(
    response: httpx.Response,
    *,
    url: str,
    headers: dict[str, str] | None,
    max_chars: int = _MAX_ERROR_BODY_CHARS,
) -> str:
    try:
        raw_text = response.text
    except (httpx.HTTPError, UnicodeDecodeError, AttributeError):
        return ""

    collapsed = " ".join(raw_text.split())
    if not collapsed:
        return ""

    secrets = _extract_secrets_for_redaction(url, headers)
    redacted = _redact_secrets(collapsed, secrets)
    if len(redacted) > max_chars:
        return redacted[:max_chars].rstrip() + "..."
    return redacted


def _compute_backoff(attempt: int, base_delay_s: float, max_delay_s: float) -> float:
    exp_delay = min(max_delay_s, base_delay_s * float(1 << (attempt - 1)))
    return float(exp_delay * random.uniform(0.5, 1.0))


async def post_json_with_retry(
    *,
    provider_name: str,
    url: str,
    payload: dict[str, object],
    headers: dict[str, str] | None = None,
    timeout_s: float,
    max_attempts: int,
    retry_base_delay_s: float,
    retry_max_delay_s: float,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: SleepFn = asyncio.sleep,
) -> dict[str, object]:
    if max_attempts < 1:
        raise ProviderError(f"{provider_name}: max_attempts must be >= 1")

    async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                response = await client.post(url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < max_attempts:
                    delay = _compute_backoff(attempt, retry_base_delay_s, retry_max_delay_s)
                    await sleep(delay)
                    continue
                raise ProviderError(
                    f"{provider_name}: transport error ({type(exc).__name__}) "
                    f"after {max_attempts} attempt(s)"
                ) from None
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"{provider_name}: HTTP client error ({type(exc).__name__})"
                ) from None

            status = response.status_code
            if status == 200:
                try:
                    data = response.json()
                except ValueError:
                    raise ProviderError(
                        f"{provider_name}: invalid JSON in HTTP 200 response"
                    ) from None
                if not isinstance(data, dict):
                    raise ProviderError(f"{provider_name}: expected JSON object in response")
                return data

            if status == 429 or 500 <= status < 600:
                if attempt < max_attempts:
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    delay = (
                        retry_after
                        if retry_after is not None
                        else _compute_backoff(attempt, retry_base_delay_s, retry_max_delay_s)
                    )
                    await sleep(delay)
                    continue
                snippet = _format_error_body_snippet(response, url=url, headers=headers)
                detail = f": {snippet}" if snippet else ""
                raise ProviderError(
                    f"{provider_name}: HTTP {status} after {max_attempts} attempt(s){detail}"
                )

            snippet = _format_error_body_snippet(response, url=url, headers=headers)
            detail = f": {snippet}" if snippet else ""
            raise ProviderError(
                f"{provider_name}: non-retryable HTTP {status} on attempt {attempt}{detail}"
            )

    raise ProviderError(f"{provider_name}: exhausted retry loop")
