"""Gemini API provider (stub), free tier. Must respect free-tier rate limits."""

from __future__ import annotations

from patchpilot.providers.base import LLMRequest, LLMResponse


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, default_model: str, timeout_s: float = 60.0) -> None:
        self._api_key = api_key
        self.default_model = default_model
        self.timeout_s = timeout_s

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("GeminiProvider.generate: see PLAN.md, phase 1")
