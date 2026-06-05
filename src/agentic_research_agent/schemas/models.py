"""Pydantic data contracts.

These typed models define the boundary of the service. Using them (rather than
passing raw dicts) gives validation, editor autocomplete, and a stable shape to
serialize if you later put the agent behind a FastAPI endpoint or a queue.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """An incoming question for the agent."""

    question: str = Field(..., min_length=1, description="The user's question.")
    thread_id: str = Field(
        default="default",
        description="Conversation id; reuse it to continue a session with memory.",
    )


class ToolCallRecord(BaseModel):
    """A single tool invocation made during a run (for observability)."""

    name: str
    args: dict = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """The agent's answer plus light execution metadata."""

    answer: str
    thread_id: str
    run_id: str = Field(description="Unique id for this agent execution.")
    duration_ms: int = Field(ge=0, description="End-to-end execution duration.")
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
