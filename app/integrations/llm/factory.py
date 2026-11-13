from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.integrations.llm.catalog import Catalog, Tier, load_catalog
from app.integrations.llm.config import ModelConfig, ModelConfigError, resolve_api_key


class LLMConfigurationError(ModelConfigError):
    pass


def build_chat_model(config: ModelConfig) -> BaseChatModel:
    api_key = SecretStr(resolve_api_key(config.api_key_ref))
    # Models that reject an effort setting must not receive one
    effort = config.effort if config.capabilities.reasoning_effort else None

    if config.dialect == "anthropic":
        # Pyright only sees pydantic alias names here (model_name, max_tokens_to_sample, stop)
        return ChatAnthropic(
            model_name=config.model,
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.read_timeout,
            max_tokens_to_sample=config.max_output_tokens,
            # max_attempts counts the first try; LangChain's max_retries does not
            max_retries=config.max_attempts - 1,
            reasoning_effort=effort,  # pyright: ignore[reportCallIssue]
            stop=None,
        )

    if config.dialect in ("openai_responses", "openai_compat"):
        return ChatOpenAI(
            model=config.model,
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.read_timeout,
            max_retries=config.max_attempts - 1,
            max_completion_tokens=config.max_output_tokens,
            reasoning_effort=effort,
            use_responses_api=config.dialect == "openai_responses",
        )

    raise LLMConfigurationError(f"Unsupported dialect: {config.dialect}")


@lru_cache
def _catalog() -> Catalog:
    return load_catalog()


def get_chat_model(profile: str, tier: Tier = "standard") -> BaseChatModel:
    return build_chat_model(_catalog().select(profile, tier))