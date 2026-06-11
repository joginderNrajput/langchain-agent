"""Shared state for the multi-agent RAG graph.

Every node reads and writes this single object. Most channels are "last value
wins" (overwritten each run); ``messages`` uses the ``add_messages`` reducer so
conversation history accumulates per ``thread_id`` when a checkpointer is used.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RetrievedChunk(TypedDict):
    """A retrieved context chunk (serialisable for state/checkpointing)."""

    content: str
    source: str
    score: float


class MultiAgentState(TypedDict, total=False):
    """State threaded through supervisor → retrieval → synthesis → critic."""

    question: str
    route: str  # "retrieve" | "direct"
    queries: list[str]  # expanded search queries
    documents: list[RetrievedChunk]  # retrieved context
    draft: str  # synthesis output
    grounded: bool  # critic verdict
    critique: str  # critic feedback for re-retrieval
    revisions: int  # critic-triggered re-retrieval rounds taken
    answer: str  # final answer
    messages: Annotated[list[AnyMessage], add_messages]
