"""Deterministic provider for tests and CI. Makes no network calls."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from patchpilot.providers.base import LLMRequest, LLMResponse, ProviderError, Usage

CannedType = str | Sequence[str] | Callable[[LLMRequest], str]


class MockProvider:
    name = "mock"

    def __init__(self, canned: CannedType = "MOCK_RESPONSE") -> None:
        self._canned = canned
        self._call_index = 0
        self.calls: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not request.messages:
            raise ProviderError("mock: request.messages must not be empty")
        self.calls.append(request)

        if callable(self._canned):
            text = self._canned(request)
        elif isinstance(self._canned, str):
            text = self._canned
        else:
            seq = list(self._canned)
            if seq:
                idx = min(self._call_index, len(seq) - 1)
                text = seq[idx]
                self._call_index += 1
            else:
                text = "MOCK_RESPONSE"

        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        return LLMResponse(
            text=text,
            model=request.model or "mock-1",
            provider=self.name,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=len(text.split())),
        )
