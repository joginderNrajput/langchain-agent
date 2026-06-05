"""Reusable FastAPI dependencies: agent access, auth, and rate limiting."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Depends, Header, HTTPException, Request, status

from agentic_research_agent.agents.service import ResearchAgent
from agentic_research_agent.config.settings import Settings
from agentic_research_agent.core.logging import get_logger

logger = get_logger(__name__)


def get_agent(request: Request) -> ResearchAgent:
    """Return the process-wide agent built during application startup."""

    agent = getattr(request.app.state, "agent", None)
    if agent is None:  # pragma: no cover - only if used before lifespan ran
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="agent_not_ready",
        )
    return agent


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


async def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Authenticate the caller via the ``X-API-Key`` header.

    If no keys are configured, authentication is disabled (suitable for local
    development only) — the application logs a warning at startup in that case.
    """

    settings: Settings = request.app.state.settings
    allowed = settings.api_keys
    if not allowed:
        return  # auth disabled
    if x_api_key is None or x_api_key not in allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_or_missing_api_key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


class SlidingWindowRateLimiter:
    """A simple in-process per-identity sliding-window limiter.

    Adequate for a single instance. For multi-replica deployments, move this to
    a shared store (e.g. Redis) so the limit is enforced cluster-wide.
    """

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._window = 60.0
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, identity: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[identity]
            while hits and now - hits[0] > self._window:
                hits.popleft()
            if len(hits) >= self._max:
                return False
            hits.append(now)
            return True


async def enforce_rate_limit(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject callers exceeding the configured per-minute request budget."""

    settings: Settings = request.app.state.settings
    if not settings.rate_limit_enabled:
        return
    limiter: SlidingWindowRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    identity = x_api_key or (request.client.host if request.client else "anonymous")
    if not limiter.check(identity):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate_limit_exceeded",
            headers={"Retry-After": "60"},
        )


# Convenience aggregate for protected routes.
Protected = (Depends(verify_api_key), Depends(enforce_rate_limit))
