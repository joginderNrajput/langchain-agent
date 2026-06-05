"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from agentic_research_agent.config.settings import Settings, get_settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    """A Settings instance isolated to a temp dir (no real env/secrets needed)."""

    get_settings.cache_clear()
    return Settings(
        groq_api_key="test-key",
        knowledge_base_dir=tmp_path / "kb",
        vector_store_dir=tmp_path / "vs",
    )
