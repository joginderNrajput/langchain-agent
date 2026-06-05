"""Checkpointer factory — pluggable conversation-state persistence.

LangGraph snapshots graph state after each step, keyed by ``thread_id``. The
backend that stores those snapshots is the single thing that decides whether
the agent can scale horizontally and survive restarts:

* ``memory``   — in-process; lost on restart, not shared. Dev/test only.
* ``sqlite``   — durable on one node; good for single-instance deployments.
* ``postgres`` — shared across replicas; required for horizontal scaling.

Concrete savers are imported lazily so that, e.g., the Postgres driver is only
needed when Postgres is actually selected.
"""

from __future__ import annotations

from contextlib import ExitStack

from langgraph.checkpoint.base import BaseCheckpointSaver

from agentic_research_agent.config.settings import CheckpointerBackend, Settings
from agentic_research_agent.core.exceptions import ConfigurationError
from agentic_research_agent.core.logging import get_logger

logger = get_logger(__name__)


def build_checkpointer(settings: Settings, stack: ExitStack) -> BaseCheckpointSaver:
    """Construct the checkpointer named in ``settings``.

    Args:
        settings: Application settings.
        stack: An :class:`~contextlib.ExitStack` owned by the caller (the agent
            service). Backends that hold connections are opened as context
            managers and registered here, so they are closed exactly when the
            owning service is torn down.

    Raises:
        ConfigurationError: The selected backend is misconfigured or its
            optional dependency is not installed.
    """

    backend = settings.checkpointer

    if backend is CheckpointerBackend.MEMORY:
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Using in-memory checkpointer (not durable).")
        return MemorySaver()

    if backend is CheckpointerBackend.SQLITE:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ConfigurationError(
                "checkpointer=sqlite requires 'langgraph-checkpoint-sqlite'."
            ) from exc

        settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Using SQLite checkpointer at %s", settings.sqlite_path)
        return stack.enter_context(SqliteSaver.from_conn_string(str(settings.sqlite_path)))

    if backend is CheckpointerBackend.POSTGRES:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ConfigurationError(
                "checkpointer=postgres requires the 'postgres' extra: "
                "install 'langgraph-checkpoint-postgres' (and psycopg)."
            ) from exc

        if not settings.postgres_dsn:
            raise ConfigurationError("postgres_dsn is required when checkpointer=postgres")
        logger.info("Using Postgres checkpointer.")
        pg_saver = stack.enter_context(PostgresSaver.from_conn_string(settings.postgres_dsn))
        pg_saver.setup()  # idempotent: ensures checkpoint tables exist
        return pg_saver

    raise ConfigurationError(f"Unsupported checkpointer backend: {backend!r}")
