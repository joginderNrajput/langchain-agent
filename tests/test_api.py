"""API tests.

A fake agent is injected into ``app.state`` so the lifespan does not build a
real ResearchAgent — these tests need no API key, no network, and no model
download. They verify transport behavior: auth, rate limiting, input limits,
health, metrics, and the happy path.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from agentic_research_agent.api.app import create_app
from agentic_research_agent.config.settings import Settings
from agentic_research_agent.schemas.models import AgentResponse, ToolCallRecord


class _FakeAgent:
    """Stand-in for ResearchAgent with the interface the API depends on."""

    def __init__(self, *, kb_ready: bool = True) -> None:
        self._kb_ready = kb_ready
        self.closed = False

    def ask(self, question: str, thread_id: str = "default") -> AgentResponse:
        return AgentResponse(
            answer=f"echo: {question}",
            thread_id=thread_id,
            run_id="test-run",
            duration_ms=5,
            tool_calls=[ToolCallRecord(name="calculator", args={"expression": "1+1"})],
        )

    def stream(self, question: str, thread_id: str = "default") -> Iterator[str]:
        yield "[thinking] calling tools: calculator"
        yield f"echo: {question}"

    def knowledge_base_ready(self) -> bool:
        return self._kb_ready

    def llm_reachable(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


def _client(settings: Settings, agent: _FakeAgent | None = None) -> TestClient:
    app = create_app(settings)
    app.state.agent = agent or _FakeAgent()
    return TestClient(app)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        api_keys_raw="secret-key",
        rate_limit_per_minute=1000,
        max_question_chars=100,
    )


def test_liveness(settings: Settings) -> None:
    with _client(settings) as client:
        r = client.get("/health/live")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_readiness_ok(settings: Settings) -> None:
    with _client(settings) as client:
        r = client.get("/health/ready")
        assert r.status_code == 200
        assert r.json()["ready"] is True


def test_readiness_degraded(settings: Settings) -> None:
    with _client(settings, _FakeAgent(kb_ready=False)) as client:
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["checks"]["knowledge_base"] is False


def test_ask_requires_api_key(settings: Settings) -> None:
    with _client(settings) as client:
        r = client.post("/v1/ask", json={"question": "hello"})
        assert r.status_code == 401


def test_ask_happy_path(settings: Settings) -> None:
    with _client(settings) as client:
        r = client.post(
            "/v1/ask",
            json={"question": "what is 1+1?"},
            headers={"X-API-Key": "secret-key"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "echo: what is 1+1?"
        assert body["run_id"] == "test-run"
        assert body["tool_calls"][0]["name"] == "calculator"
        assert r.headers.get("X-Request-ID")


def test_ask_rejects_overlong_question(settings: Settings) -> None:
    with _client(settings) as client:
        r = client.post(
            "/v1/ask",
            json={"question": "x" * 101},
            headers={"X-API-Key": "secret-key"},
        )
        assert r.status_code == 413


def test_rate_limit_enforced() -> None:
    settings = Settings(_env_file=None, api_keys_raw="k", rate_limit_per_minute=2)
    with _client(settings) as client:
        headers = {"X-API-Key": "k"}
        codes = [
            client.post("/v1/ask", json={"question": "hi"}, headers=headers).status_code
            for _ in range(3)
        ]
        assert codes[:2] == [200, 200]
        assert codes[2] == 429


def test_stream_endpoint(settings: Settings) -> None:
    with _client(settings) as client:
        r = client.post(
            "/v1/stream",
            json={"question": "stream please"},
            headers={"X-API-Key": "secret-key"},
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        assert "echo: stream please" in r.text
        assert "[DONE]" in r.text


def test_metrics_endpoint(settings: Settings) -> None:
    with _client(settings) as client:
        client.post("/v1/ask", json={"question": "q"}, headers={"X-API-Key": "secret-key"})
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "research_agent_requests_total" in r.text
