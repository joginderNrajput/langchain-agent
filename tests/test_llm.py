"""Tests for the LLM provider factory."""

from __future__ import annotations

import pytest

from agentic_research_agent.config.settings import LLMProvider, Settings
from agentic_research_agent.core.exceptions import ConfigurationError
from agentic_research_agent.core.llm import build_chat_model


def test_missing_api_key_raises_configuration_error() -> None:
    settings = Settings(_env_file=None, llm_provider=LLMProvider.GROQ, groq_api_key=None)
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        build_chat_model(settings)


def test_groq_model_built_when_key_present() -> None:
    settings = Settings(_env_file=None, llm_provider=LLMProvider.GROQ, groq_api_key="test-key")
    model = build_chat_model(settings)
    # Duck-typed check: it's a chat model exposing the standard interface.
    assert hasattr(model, "invoke")
    assert hasattr(model, "bind_tools")


def test_gemini_missing_api_key_raises_configuration_error() -> None:
    settings = Settings(_env_file=None, llm_provider=LLMProvider.GEMINI, google_api_key=None)
    with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY"):
        build_chat_model(settings)


def test_gemini_model_built_when_key_present() -> None:
    settings = Settings(_env_file=None, llm_provider=LLMProvider.GEMINI, google_api_key="test-key")
    model = build_chat_model(settings)
    assert hasattr(model, "invoke")
    assert hasattr(model, "bind_tools")
    assert model.model.endswith("gemini-2.5-flash")  # provider default resolved


def test_azure_missing_endpoint_raises_configuration_error() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider=LLMProvider.AZURE,
        azure_openai_api_key="test-key",
        azure_openai_deployment="gpt-4o",
    )
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_ENDPOINT"):
        build_chat_model(settings)


def test_azure_missing_deployment_raises_configuration_error() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider=LLMProvider.AZURE,
        azure_openai_api_key="test-key",
        azure_openai_endpoint="https://example.openai.azure.com/",
    )
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_DEPLOYMENT"):
        build_chat_model(settings)


def test_azure_model_built_when_configured() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider=LLMProvider.AZURE,
        azure_openai_api_key="test-key",
        azure_openai_endpoint="https://example.openai.azure.com/",
        azure_openai_deployment="my-deployment",
    )
    model = build_chat_model(settings)
    assert hasattr(model, "invoke")
    assert hasattr(model, "bind_tools")
    # The deployment name is the effective model for Azure.
    assert settings.effective_llm_model == "my-deployment"
