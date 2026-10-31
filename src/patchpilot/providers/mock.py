"""Deterministic provider for tests and CI. Makes no network calls."""

from __future__ import annotations

from patchpilot.providers.base import LLMRequest, LLMResponse, ProviderError, Usage


class MockProvider:
    name = "mock"

    def __init__(self, canned: str = "MOCK_RESPONSE") -> None:
        self._canned = canned
        self.calls: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not request.messages:
            raise ProviderError("mock: request.messages must not be empty")
        self.calls.append(request)
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        return LLMResponse(
            text=self._canned,
            model=request.model or "mock-1",
            provider=self.name,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=len(self._canned.split())),
        )
