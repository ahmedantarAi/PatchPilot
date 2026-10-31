"""Ollama provider (stub). Ollama runs natively on the Windows host, not in Docker."""

from __future__ import annotations

from patchpilot.providers.base import LLMRequest, LLMResponse


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, default_model: str, timeout_s: float = 60.0) -> None:
        self.base_url = base_url
        self.default_model = default_model
        self.timeout_s = timeout_s

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("OllamaProvider.generate: see PLAN.md, phase 1")
