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
    settings = Settings(
        _env_file=None, llm_provider=LLMProvider.GROQ, groq_api_key="test-key"
    )
    model = build_chat_model(settings)
    # Duck-typed check: it's a chat model exposing the standard interface.
    assert hasattr(model, "invoke")
    assert hasattr(model, "bind_tools")
