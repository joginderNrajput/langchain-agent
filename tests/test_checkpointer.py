"""Tests for the checkpointer factory."""

from __future__ import annotations

from contextlib import ExitStack

import pytest

from agentic_research_agent.agents.checkpointer import build_checkpointer
from agentic_research_agent.config.settings import CheckpointerBackend, Settings


def test_memory_backend() -> None:
    settings = Settings(_env_file=None, checkpointer=CheckpointerBackend.MEMORY)
    with ExitStack() as stack:
        saver = build_checkpointer(settings, stack)
        assert saver is not None


def test_sqlite_backend(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        checkpointer=CheckpointerBackend.SQLITE,
        sqlite_path=tmp_path / "cp.sqlite",
    )
    with ExitStack() as stack:
        saver = build_checkpointer(settings, stack)
        assert saver is not None
    assert (tmp_path / "cp.sqlite").exists()


def test_postgres_requires_dsn() -> None:
    # The model validator rejects postgres without a DSN at construction time.
    with pytest.raises(ValueError, match="postgres_dsn"):
        Settings(_env_file=None, checkpointer=CheckpointerBackend.POSTGRES)
