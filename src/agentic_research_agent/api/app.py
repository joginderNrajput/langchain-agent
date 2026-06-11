"""FastAPI application.

The agent is constructed **once** in the lifespan handler (it loads an embedding
model and compiles the graph) and shared across all requests. Routes add only
transport concerns: authentication, rate limiting, a hard request timeout,
health/readiness probes, Prometheus metrics, and error mapping.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import iterate_in_threadpool

from agentic_research_agent.agents.base import AgentService
from agentic_research_agent.api import metrics
from agentic_research_agent.api.dependencies import (
    SlidingWindowRateLimiter,
    enforce_rate_limit,
    get_agent,
    verify_api_key,
)
from agentic_research_agent.config.settings import Settings, get_settings
from agentic_research_agent.core.exceptions import AgentError, ConfigurationError
from agentic_research_agent.core.logging import configure_logging, get_logger
from agentic_research_agent.schemas.models import AgentRequest, AgentResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build shared resources on startup; release them on shutdown."""

    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings
    configure_logging(settings.log_level, json_logs=settings.is_production)

    if settings.is_production and not settings.api_keys:
        logger.warning("Running in production with NO API keys configured — API is open.")

    app.state.rate_limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)

    # Allow tests/embedders to inject a pre-built agent; otherwise build one
    # (single ReAct or multi-agent RAG, per AGENT_MODE).
    if getattr(app.state, "agent", None) is None:
        from agentic_research_agent.service_factory import build_agent_service

        logger.info("Building agent service for API process…")
        app.state.agent = build_agent_service(settings)

    try:
        yield
    finally:
        agent = getattr(app.state, "agent", None)
        if agent is not None and hasattr(agent, "close"):
            agent.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory."""

    app = FastAPI(
        title="Agentic Research Agent API",
        version="0.1.0",
        summary="Provider-agnostic research assistant (LangChain + LangGraph).",
        lifespan=lifespan,
    )
    if settings is not None:
        app.state.settings = settings

    resolved = settings or get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_middleware(app)
    _register_error_handlers(app)
    _register_routes(app)
    return app


def _register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Attach a request id and enforce a hard request timeout."""

        request_id = request.headers.get("X-Request-ID", uuid4().hex)
        request.state.request_id = request_id
        timeout = request.app.state.settings.request_timeout_seconds
        try:
            response = await asyncio.wait_for(call_next(request), timeout=timeout)
        except TimeoutError:
            logger.warning("request timed out request_id=%s path=%s", request_id, request.url.path)
            return JSONResponse(
                {"detail": "request_timeout", "request_id": request_id},
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        response.headers["X-Request-ID"] = request_id
        return response


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ConfigurationError)
    async def _config_error(request: Request, exc: ConfigurationError) -> JSONResponse:
        logger.error("configuration error: %s", exc)
        return JSONResponse({"detail": "configuration_error"}, status_code=500)

    @app.exception_handler(AgentError)
    async def _agent_error(request: Request, exc: AgentError) -> JSONResponse:
        # AgentExecutionError and friends → upstream/agent failure.
        return JSONResponse({"detail": "agent_run_failed"}, status_code=502)


def _register_routes(app: FastAPI) -> None:
    @app.get("/health/live", tags=["health"])
    async def live() -> dict:
        """Liveness: the process is running."""

        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def ready(request: Request) -> JSONResponse:
        """Readiness: dependencies are reachable."""

        agent: AgentService | None = getattr(request.app.state, "agent", None)
        if agent is None:
            return JSONResponse({"ready": False, "checks": {}}, status_code=503)
        checks = {
            "knowledge_base": agent.knowledge_base_ready(),
            "llm": agent.llm_reachable(),
        }
        ok = all(checks.values())
        return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)

    @app.get("/metrics", tags=["observability"])
    async def prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/v1/ask",
        response_model=AgentResponse,
        tags=["agent"],
        dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
    )
    async def ask(
        payload: AgentRequest,
        request: Request,
        agent: AgentService = Depends(get_agent),
    ) -> AgentResponse:
        """Answer a single question and return a structured response."""

        _validate_length(payload, request.app.state.settings)
        try:
            # ask() is blocking; off-load to a worker thread to free the loop.
            response = await asyncio.to_thread(agent.ask, payload.question, payload.thread_id)
        except Exception:
            metrics.REQUESTS.labels(endpoint="ask", status="error").inc()
            metrics.RUN_OUTCOMES.labels(outcome="error").inc()
            raise
        metrics.REQUESTS.labels(endpoint="ask", status="ok").inc()
        metrics.RUN_OUTCOMES.labels(outcome="success").inc()
        metrics.RUN_LATENCY.observe(response.duration_ms / 1000.0)
        for call in response.tool_calls:
            metrics.TOOL_CALLS.labels(tool=call.name).inc()
        return response

    @app.post(
        "/v1/stream",
        tags=["agent"],
        dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
    )
    def stream(
        payload: AgentRequest,
        request: Request,
        agent: AgentService = Depends(get_agent),
    ) -> StreamingResponse:
        """Stream the run as Server-Sent Events (one event per step)."""

        _validate_length(payload, request.app.state.settings)

        def event_source():
            for chunk in agent.stream(payload.question, payload.thread_id):
                yield f"data: {chunk}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(
            iterate_in_threadpool(event_source()),
            media_type="text/event-stream",
        )


def _validate_length(payload: AgentRequest, settings: Settings) -> None:
    from fastapi import HTTPException

    if len(payload.question) > settings.max_question_chars:
        raise HTTPException(
            status_code=413,
            detail=f"question exceeds {settings.max_question_chars} characters",
        )


# Module-level ASGI app for `uvicorn agentic_research_agent.api.app:app`.
app = create_app()
