"""The LangGraph ReAct agent.

This is the classic reason-act loop expressed as an explicit graph (rather than
a prebuilt) so the control flow is visible and reviewable:

    START → agent → (tool calls?) ─yes→ tools → agent
                         │
                         no
                         ↓
                        END

The ``agent`` node calls the LLM; if the LLM requested tools, ``tools_condition``
routes to the ``ToolNode`` which executes them and feeds results back into the
agent. The loop repeats until the LLM answers without requesting a tool.
"""

from __future__ import annotations

import json
import re
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agentic_research_agent.agents.prompts import SYSTEM_PROMPT
from agentic_research_agent.agents.state import AgentState


def build_agent_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the ReAct agent graph.

    Args:
        llm: The chat model. Tools are bound here so the model can emit
            structured tool calls.
        tools: Tools the agent may invoke.
        checkpointer: Optional persistence backend. When provided, runs are
            keyed by ``thread_id`` so conversations retain memory across calls.

    Returns:
        A compiled, invocable graph.
    """

    llm_with_tools = llm.bind_tools(tools)
    tool_names = {tool.name for tool in tools}

    def agent_node(state: AgentState) -> dict:
        """Invoke the LLM with the system prompt prepended to the history."""

        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as exc:
            recovered_response = _coerce_failed_tool_call(exc, tool_names)
            if recovered_response is None:
                raise
            response = recovered_response
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    # tools_condition returns "tools" when the last message has tool calls,
    # otherwise the END sentinel.
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


_FAILED_TOOL_CALL_RE = re.compile(
    r"<function=(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<args>\{.*?\})</function>",
    re.DOTALL,
)


def _coerce_failed_tool_call(
    exc: Exception,
    allowed_tool_names: set[str],
) -> AIMessage | None:
    """Recover from provider pseudo-tool-call errors when possible.

    Some Groq-hosted Llama models occasionally reject their own generated tool
    call with ``tool_use_failed`` while still returning the intended call in a
    ``failed_generation`` payload such as:

        <function=web_search{"query": "latest news"}</function>

    When that payload is well-formed and names one of our tools, translate it
    into a normal LangChain ``AIMessage`` so the graph can execute the tool.
    """

    error_text = str(exc)
    if "tool_use_failed" not in error_text:
        return None

    match = _FAILED_TOOL_CALL_RE.search(error_text)
    if match is None:
        return None

    tool_name = match.group("name")
    if tool_name not in allowed_tool_names:
        return None

    try:
        args = json.loads(match.group("args"))
    except json.JSONDecodeError:
        return None

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": args,
                "id": f"recovered_{uuid4().hex}",
                "type": "tool_call",
            }
        ],
    )
