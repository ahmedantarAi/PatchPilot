"""Environment-driven settings (see .env.example)."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["ollama", "gemini", "mock"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PP_", env_file=".env", extra="ignore")

    llm_provider: ProviderName = "mock"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    request_timeout_s: float = 60.0
