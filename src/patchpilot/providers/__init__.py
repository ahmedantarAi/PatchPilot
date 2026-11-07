"""LLM provider registry."""

from __future__ import annotations

from patchpilot.config import ProviderName, Settings
from patchpilot.providers.base import LLMProvider, PrivacyViolationError
from patchpilot.providers.gemini import GeminiProvider
from patchpilot.providers.mock import MockProvider
from patchpilot.providers.ollama import OllamaProvider
from patchpilot.providers.openrouter import OpenRouterProvider
from patchpilot.router.router import ModelRouter


def get_provider(name: ProviderName | None = None, settings: Settings | None = None) -> LLMProvider:
    s = settings or Settings()
    chosen = name or s.llm_provider
    if chosen == "mock":
        return MockProvider()
    if chosen == "ollama":
        return OllamaProvider(
            s.ollama_base_url,
            s.ollama_model,
            s.request_timeout_s,
            max_attempts=s.max_attempts,
            retry_base_delay_s=s.retry_base_delay_s,
            retry_max_delay_s=s.retry_max_delay_s,
        )
    if chosen == "gemini":
        return GeminiProvider(
            s.gemini_api_key,
            s.gemini_model,
            s.request_timeout_s,
            max_attempts=s.max_attempts,
            retry_base_delay_s=s.retry_base_delay_s,
            retry_max_delay_s=s.retry_max_delay_s,
        )
    if chosen == "openrouter":
        return OpenRouterProvider(
            s.openrouter_api_key,
            s.openrouter_model,
            s.request_timeout_s,
            base_url=s.openrouter_base_url,
            max_attempts=s.max_attempts,
            retry_base_delay_s=s.retry_base_delay_s,
            retry_max_delay_s=s.retry_max_delay_s,
        )
    if chosen == "router":
        local = OllamaProvider(
            s.ollama_base_url,
            s.ollama_model,
            s.request_timeout_s,
            max_attempts=s.max_attempts,
            retry_base_delay_s=s.retry_base_delay_s,
            retry_max_delay_s=s.retry_max_delay_s,
        )
        frontier: LLMProvider
        if s.frontier_provider == "gemini":
            frontier = GeminiProvider(
                s.gemini_api_key,
                s.gemini_model,
                s.request_timeout_s,
                max_attempts=s.max_attempts,
                retry_base_delay_s=s.retry_base_delay_s,
                retry_max_delay_s=s.retry_max_delay_s,
            )
        else:
            frontier = OpenRouterProvider(
                s.openrouter_api_key,
                s.openrouter_model,
                s.request_timeout_s,
                base_url=s.openrouter_base_url,
                max_attempts=s.max_attempts,
                retry_base_delay_s=s.retry_base_delay_s,
                retry_max_delay_s=s.retry_max_delay_s,
            )
        return ModelRouter(local, frontier)
    raise ValueError(f"unknown provider: {chosen!r}")


__all__ = [
    "get_provider",
    "LLMProvider",
    "MockProvider",
    "OllamaProvider",
    "GeminiProvider",
    "OpenRouterProvider",
    "ModelRouter",
    "PrivacyViolationError",
]
