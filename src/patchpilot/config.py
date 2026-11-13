"""Environment-driven settings (see .env.example)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["ollama", "gemini", "mock", "router", "openrouter"]
FrontierProviderName = Literal["gemini", "openrouter"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _find_duplicate_keys(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    keys: list[str] = []
    duplicates: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            if not key:
                continue
            if key in keys and key not in duplicates:
                duplicates.append(key)
            keys.append(key)
    return duplicates


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PP_",
        env_file=(ENV_FILE, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: ProviderName = "mock"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    gemini_api_key: str = ""
    gemini_model: str = ""  # required when provider=gemini; set PP_GEMINI_MODEL in .env
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemma-4-31b-it:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    frontier_provider: FrontierProviderName = "openrouter"
    request_timeout_s: float = 60.0
    max_attempts: int = 3
    retry_base_delay_s: float = 0.5
    retry_max_delay_s: float = 8.0

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_llm_provider(cls, val: Any) -> Any:
        if isinstance(val, str):
            stripped = val.strip()
            return stripped if stripped else "mock"
        return val

    @field_validator("frontier_provider", mode="before")
    @classmethod
    def _normalize_frontier_provider(cls, val: Any) -> Any:
        if isinstance(val, str):
            stripped = val.strip()
            return stripped if stripped else "openrouter"
        return val

    def __init__(self, **values: Any) -> None:
        if "_env_file" in values:
            env_files = values["_env_file"]
            if env_files is None:
                targets: list[object] = []
            elif isinstance(env_files, tuple | list):
                targets = list(env_files)
            else:
                targets = [env_files]
        else:
            config_env = self.model_config.get("env_file")
            targets = (
                list(config_env)
                if isinstance(config_env, tuple | list)
                else [config_env]
                if config_env
                else []
            )

        for target in targets:
            if isinstance(target, str | Path):
                p = Path(target)
                dups = _find_duplicate_keys(p)
                if dups:
                    raise ValueError(f"duplicate variable {dups[0]!r} found in {p}")

        super().__init__(**values)

    def __repr__(self) -> str:
        return (
            f"Settings(llm_provider={self.llm_provider!r}, "
            f"ollama_base_url={self.ollama_base_url!r}, "
            f"ollama_model={self.ollama_model!r}, "
            "gemini_api_key='***REDACTED***', "
            f"gemini_model={self.gemini_model!r}, "
            "openrouter_api_key='***REDACTED***', "
            f"openrouter_model={self.openrouter_model!r}, "
            f"openrouter_base_url={self.openrouter_base_url!r}, "
            f"frontier_provider={self.frontier_provider!r}, "
            f"request_timeout_s={self.request_timeout_s!r}, "
            f"max_attempts={self.max_attempts!r}, "
            f"retry_base_delay_s={self.retry_base_delay_s!r}, "
            f"retry_max_delay_s={self.retry_max_delay_s!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


class LangfuseSettings(BaseSettings):
    """Environment-driven settings for optional Langfuse v4 observability."""

    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_",
        env_file=(ENV_FILE, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "http://localhost:3000"
    public_key: str = ""
    secret_key: str = ""
    init_project_public_key: str = ""
    init_project_secret_key: str = ""

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)

    @property
    def effective_public_key(self) -> str:
        return (self.public_key or self.init_project_public_key).strip()

    @property
    def effective_secret_key(self) -> str:
        return (self.secret_key or self.init_project_secret_key).strip()

    @property
    def is_configured(self) -> bool:
        pub = self.effective_public_key
        sec = self.effective_secret_key
        host = self.host.strip()
        if not host or not pub or not sec:
            return False
        if "changeme" in pub.lower() or "changeme" in sec.lower():
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"LangfuseSettings(host={self.host!r}, "
            f"public_key={self.effective_public_key!r}, "
            "secret_key='***REDACTED***')"
        )

    def __str__(self) -> str:
        return self.__repr__()
