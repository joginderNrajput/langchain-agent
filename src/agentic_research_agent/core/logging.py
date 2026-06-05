"""Centralized logging configuration.

Uses :mod:`rich` for readable, colorized console output in development. In a
real deployment you would swap the handler for a JSON formatter shipping to
your log aggregator; keeping configuration in one place makes that a one-line
change.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from rich.logging import RichHandler

_LOGGER_NAME = "agentic_research_agent"
_configured = False


class JsonLogFormatter(logging.Formatter):
    """Minimal JSON formatter for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure the package logger exactly once.

    Idempotent: repeated calls (e.g. from both the CLI and the service layer)
    are no-ops after the first, so handlers are never duplicated.
    """

    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)

    if _configured:
        return

    if json_logs:
        handler: logging.Handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
    else:
        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    logger.addHandler(handler)
    logger.propagate = False

    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the package namespace.

    Pass ``__name__`` from a module to get a hierarchical logger such as
    ``agentic_research_agent.tools.web_search``.
    """

    if name is None or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    if name.startswith(_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
