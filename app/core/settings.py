"""Server-side settings for the first LLM integration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_timeout_seconds: float = Field(default=60, gt=0, le=300)

    agent_max_steps: int = Field(default=8, ge=1, le=20)
    agent_max_tool_calls: int = Field(default=6, ge=1, le=20)
    agent_timeout_seconds: float = Field(default=60, gt=0, le=300)
    verifier_max_retries: int = Field(default=1, ge=0, le=3)

    # Per-user token quota of chat sessions (see app/storage/mysql/usage_store.py); 0 = off.
    # Measured 1.4k (a count) to 12k (lists of articles) cache-weighted tokens per turn,
    # about 6k on average: ~25 turns per 5 hours, ~250 per week.
    quota_five_hour_tokens: int = Field(default=150_000, ge=0)
    quota_weekly_tokens: int = Field(default=1_500_000, ge=0)

    # Text embeddings for retrieval (app/integrations/voyage/embeddings.py).
    voyage_api_key: SecretStr = SecretStr("")
    # Jev, a TypeSafe System One decision model (app/integrations/jev/client.py).
    typesafe_api_key: SecretStr = SecretStr("")
    embedding_model: str = "voyage-4"
    embedding_dimensions: int = Field(default=1024, ge=256, le=2048)

    mysql_source_readonly_dsn: SecretStr = SecretStr("")
    # Node 1 finsights_ai: chat sessions and messages (read/write account).
    mysql_chat_dsn: SecretStr = SecretStr("")

    # finsights-coredb creates MySQL accounts with REQUIRE SSL. Each node's certificate is
    # issued to MySQL_Endpoint_Server (not the IP), verified against that node's CA.
    # Turn off only for a local MySQL without TLS.
    mysql_tls: bool = True
    mysql_tls_server_name: str = "MySQL_Endpoint_Server"
    mysql_source_tls_ca: Path = _ROOT / "certs" / "mysql-node-2-ca.pem"
    mysql_chat_tls_ca: Path = _ROOT / "certs" / "mysql-node-1-ca.pem"


@lru_cache
def get_settings() -> Settings:
    return Settings()
