"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_research_agent.config.settings import LLMProvider, Settings


def test_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.llm_provider is LLMProvider.GROQ
    assert s.llm_model is None  # unset → resolved per provider
    assert s.effective_llm_model == "llama-3.3-70b-versatile"
    assert 0.0 <= s.llm_temperature <= 2.0
    assert s.retriever_top_k > 0


@pytest.mark.parametrize(
    ("provider", "expected_model"),
    [
        (LLMProvider.GROQ, "llama-3.3-70b-versatile"),
        (LLMProvider.OPENAI, "gpt-4o-mini"),
        (LLMProvider.ANTHROPIC, "claude-sonnet-4-6"),
        (LLMProvider.GEMINI, "gemini-2.5-flash"),
        (LLMProvider.OLLAMA, "llama3.1"),
    ],
)
def test_effective_model_defaults_per_provider(provider: LLMProvider, expected_model: str) -> None:
    # This is the fix for the OpenAI 404: switching provider without setting
    # LLM_MODEL resolves the provider's own default, not a Groq model.
    s = Settings(_env_file=None, llm_provider=provider)
    assert s.effective_llm_model == expected_model


def test_explicit_model_overrides_default() -> None:
    s = Settings(_env_file=None, llm_provider=LLMProvider.OPENAI, llm_model="gpt-4o")
    assert s.effective_llm_model == "gpt-4o"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    s = Settings(_env_file=None)
    assert s.llm_provider is LLMProvider.OPENAI
    assert s.llm_temperature == 0.7


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="VERBOSE")


def test_temperature_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_temperature=5.0)


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(_env_file=None, chunk_size=1000, chunk_overlap=1000)


def test_is_production_flag() -> None:
    assert Settings(_env_file=None, environment="production").is_production
    assert not Settings(_env_file=None, environment="development").is_production
