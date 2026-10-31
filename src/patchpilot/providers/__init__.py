"""LLM provider registry."""

from __future__ import annotations

from patchpilot.config import ProviderName, Settings
from patchpilot.providers.base import LLMProvider
from patchpilot.providers.gemini import GeminiProvider
from patchpilot.providers.mock import MockProvider
from patchpilot.providers.ollama import OllamaProvider


def get_provider(name: ProviderName | None = None, settings: Settings | None = None) -> LLMProvider:
    s = settings or Settings()
    chosen = name or s.llm_provider
    if chosen == "mock":
        return MockProvider()
    if chosen == "ollama":
        return OllamaProvider(s.ollama_base_url, s.ollama_model, s.request_timeout_s)
    if chosen == "gemini":
        return GeminiProvider(s.gemini_api_key, s.gemini_model, s.request_timeout_s)
    raise ValueError(f"unknown provider: {chosen!r}")


__all__ = ["get_provider", "LLMProvider", "MockProvider", "OllamaProvider", "GeminiProvider"]
