"""Observability configuration.

LangChain and LangGraph emit tracing spans automatically when the LangSmith
environment variables are present. Rather than make callers set those by hand,
we translate our typed settings into the expected environment variables once at
startup. This keeps tracing a single config flag (``LANGSMITH_TRACING=true``)
without scattering env lookups through the code.
"""

from __future__ import annotations

import os

from agentic_research_agent.config.settings import Settings
from agentic_research_agent.core.logging import get_logger

logger = get_logger(__name__)


def configure_tracing(settings: Settings) -> None:
    """Enable LangSmith tracing if configured.

    No-op when tracing is disabled or no API key is set, so local/offline runs
    are never blocked by a missing key.
    """

    if not settings.langsmith_tracing:
        return
    if not settings.langsmith_api_key:
        logger.warning("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is unset; skipping.")
        return

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    logger.info("LangSmith tracing enabled (project=%s).", settings.langsmith_project)
