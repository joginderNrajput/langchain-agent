"""Graph state schema.

LangGraph threads a single state object through every node. Our agent only
needs a running message history; the ``add_messages`` reducer appends new
messages (and reconciles tool-call IDs) instead of overwriting, which is what
makes the ReAct loop accumulate context across turns.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State carried between graph nodes."""

    messages: Annotated[list[AnyMessage], add_messages]
