"""Provider interface: every LLM backend implements `LLMProvider`."""

from __future__ import annotations

import asyncio
import email.utils
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable

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
                raise ProviderError(
                    f"{provider_name}: HTTP {status} after {max_attempts} attempt(s)"
                )

            raise ProviderError(
                f"{provider_name}: non-retryable HTTP {status} on attempt {attempt}"
            )

    raise ProviderError(f"{provider_name}: exhausted retry loop")
