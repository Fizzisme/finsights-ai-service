"""Server-side settings for the first LLM integration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_timeout_seconds: float = Field(default=60, gt=0, le=300)

    agent_max_steps: int = Field(default=8, ge=1, le=20)
    agent_timeout_seconds: float = Field(default=60, gt=0, le=300)

    mysql_source_readonly_dsn: SecretStr = SecretStr("")


@lru_cache
def get_settings() -> Settings:
    return Settings()
