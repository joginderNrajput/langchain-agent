"""HTTP API layer.

A thin FastAPI surface over :class:`~agentic_research_agent.ResearchAgent`. The
agent is built once at application startup and shared across requests; the
routes add only transport concerns — auth, rate limiting, timeouts, health,
metrics, and error mapping.
"""

from agentic_research_agent.api.app import app, create_app

__all__ = ["app", "create_app"]
