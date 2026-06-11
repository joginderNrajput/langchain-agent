"""The common agent-service interface.

Both :class:`~agentic_research_agent.agents.service.ResearchAgent` and
:class:`~agentic_research_agent.multiagent.service.MultiAgentRAG` satisfy this
Protocol structurally, so the CLI and HTTP API depend on the interface — not a
concrete class — and switch implementations by configuration.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from agentic_research_agent.schemas.models import AgentResponse


@runtime_checkable
class AgentService(Protocol):
    """A question-answering agent with health and lifecycle hooks."""

    def ask(self, question: str, thread_id: str = "default") -> AgentResponse: ...

    def stream(self, question: str, thread_id: str = "default") -> Iterator[str]: ...

    def knowledge_base_ready(self) -> bool: ...

    def llm_reachable(self) -> bool: ...

    def close(self) -> None: ...
