"""Environment-driven settings (see .env.example)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["ollama", "gemini", "mock"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _find_duplicate_keys(path: Path) -> list[str]:
    if not path.is_file():
        return []
    keys: list[str] = []
    duplicates: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
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
    request_timeout_s: float = 60.0
    max_attempts: int = 3
    retry_base_delay_s: float = 0.5
    retry_max_delay_s: float = 8.0

    def __init__(self, **values: Any) -> None:
        env_files = values.get("_env_file")
        if env_files is None:
            config_env = self.model_config.get("env_file")
            targets: list[object] = (
                list(config_env)
                if isinstance(config_env, tuple | list)
                else [config_env]
                if config_env
                else []
            )
        else:
            targets = list(env_files) if isinstance(env_files, tuple | list) else [env_files]

        for target in targets:
            if isinstance(target, str | Path):
                p = Path(target)
                dups = _find_duplicate_keys(p)
                if dups:
                    raise ValueError(f"duplicate variable {dups[0]!r} found in {p}")

        super().__init__(**values)


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
        if "CHANGEME" in pub or "CHANGEME" in sec:
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
