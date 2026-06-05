"""Chat-model factory.

The rest of the application depends only on LangChain's :class:`BaseChatModel`
interface and never imports a concrete provider. Swapping Groq for Anthropic,
OpenAI, or a local Ollama model is therefore a configuration change, not a code
change — the single seam where a provider is chosen lives here.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from agentic_research_agent.config.settings import LLMProvider, Settings
from agentic_research_agent.core.exceptions import (
    ConfigurationError,
    LLMProviderError,
)
from agentic_research_agent.core.logging import get_logger

logger = get_logger(__name__)


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Construct a chat model for the provider named in ``settings``.

    Provider SDKs are imported lazily inside each branch so that installing,
    say, only ``langchain-groq`` does not force the others to be importable.

    Raises:
        ConfigurationError: A required API key is missing.
        LLMProviderError: The provider is unknown/unsupported.
    """

    provider = settings.llm_provider
    logger.debug("Building chat model: provider=%s model=%s", provider, settings.llm_model)

    if provider is LLMProvider.GROQ:
        return _build_groq(settings)
    if provider is LLMProvider.ANTHROPIC:
        return _build_anthropic(settings)
    if provider is LLMProvider.OPENAI:
        return _build_openai(settings)
    if provider is LLMProvider.OLLAMA:
        return _build_ollama(settings)

    raise LLMProviderError(f"Unsupported LLM provider: {provider!r}")


def _require_key(value: str | None, env_var: str) -> SecretStr:
    if not value:
        raise ConfigurationError(
            f"{env_var} is not set. Add it to your .env file or environment "
            f"to use this provider."
        )
    return SecretStr(value)


def _build_groq(settings: Settings) -> BaseChatModel:
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        api_key=_require_key(settings.groq_api_key, "GROQ_API_KEY"),
    )


def _build_anthropic(settings: Settings) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    # `model`/`max_tokens` are valid runtime fields; the v1 type stubs omit them.
    return ChatAnthropic(  # type: ignore[call-arg]
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        api_key=_require_key(settings.anthropic_api_key, "ANTHROPIC_API_KEY"),
    )


def _build_openai(settings: Settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    # `max_tokens` is a valid runtime field; the v1 type stubs omit it.
    return ChatOpenAI(  # type: ignore[call-arg]
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        api_key=_require_key(settings.openai_api_key, "OPENAI_API_KEY"),
    )


def _build_ollama(settings: Settings) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens,
        base_url=settings.ollama_base_url,
    )
