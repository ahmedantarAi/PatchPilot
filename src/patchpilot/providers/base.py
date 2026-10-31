"""Provider interface: every LLM backend implements `LLMProvider`."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    messages: list[Message]
    model: str | None = None  # None -> provider default
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)


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
