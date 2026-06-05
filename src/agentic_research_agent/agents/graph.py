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
from agentic_research_agent.core.logging import get_logger

logger = get_logger(__name__)

# How many times to re-invoke the model when it returns an unparseable
# ``tool_use_failed`` error. These failures are sampling artifacts, so a fresh
# generation usually succeeds.
_MAX_LLM_ATTEMPTS = 3


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
        """Invoke the LLM, recovering from provider tool-call hiccups.

        Strategy on failure: first try to salvage the intended tool call from
        the error payload; if that is not possible and the error is a
        ``tool_use_failed`` sampling artifact, re-invoke the model a few times;
        only a genuinely unrelated error (auth, rate limit, context length)
        propagates.
        """

        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        for attempt in range(1, _MAX_LLM_ATTEMPTS + 1):
            try:
                return {"messages": [llm_with_tools.invoke(messages)]}
            except Exception as exc:
                recovered = _coerce_failed_tool_call(exc, tool_names)
                if recovered is not None:
                    return {"messages": [recovered]}
                error_text = str(exc)
                is_tool_use_failure = (
                    "tool_use_failed" in error_text or "failed_generation" in error_text
                )
                if not is_tool_use_failure or attempt == _MAX_LLM_ATTEMPTS:
                    raise
                logger.warning(
                    "tool_use_failed and unparseable; resampling (attempt %d/%d)",
                    attempt,
                    _MAX_LLM_ATTEMPTS,
                )
        # Unreachable: the loop either returns or raises on the final attempt.
        raise RuntimeError("agent_node exhausted retries without returning")

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


# A failed_generation payload names a tool either with a function tag —
#   <function=web_search>{"query": "x"}</function>   (with or without the '>')
#   <function=web_search{"query": "x"}</function>
# — or as a JSON object, optionally wrapped in <tool_call>…</tool_call>:
#   {"name": "web_search", "arguments": {"query": "x"}}
_FUNCTION_NAME_RE = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_]*)")
_JSON_NAME_RE = re.compile(r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"')


def _coerce_failed_tool_call(
    exc: Exception,
    allowed_tool_names: set[str],
) -> AIMessage | None:
    """Recover from provider pseudo-tool-call errors when possible.

    Some Groq-hosted Llama models occasionally reject their own generated tool
    call with ``tool_use_failed`` while still returning the intended call in a
    ``failed_generation`` payload. The payload appears in several shapes across
    model versions, so we parse defensively and recover every well-formed call
    that names one of our tools, translating them into a normal LangChain
    ``AIMessage`` so the graph can execute the tools instead of crashing.
    """

    error_text = str(exc)
    if "tool_use_failed" not in error_text and "failed_generation" not in error_text:
        return None

    tool_calls = _recover_tool_calls(error_text, allowed_tool_names)
    if not tool_calls:
        return None
    return AIMessage(content="", tool_calls=tool_calls)


def _recover_tool_calls(text: str, allowed: set[str]) -> list[dict]:
    """Extract any recoverable tool calls from a failed_generation payload."""

    calls: list[dict] = []
    seen: set[str] = set()

    def add(name: str, args: dict) -> None:
        if name not in allowed or not isinstance(args, dict):
            return
        dedup_key = f"{name}:{json.dumps(args, sort_keys=True)}"
        if dedup_key in seen:
            return
        seen.add(dedup_key)
        calls.append(
            {
                "name": name,
                "args": args,
                "id": f"recovered_{uuid4().hex}",
                "type": "tool_call",
            }
        )

    # Shape 1: <function=NAME …{ …balanced JSON… }
    for match in _FUNCTION_NAME_RE.finditer(text):
        raw = _balanced_json_after(text, match.end())
        if raw is None:
            continue
        args = _safe_json_loads(raw)
        if isinstance(args, dict):
            add(match.group(1), args)

    # Shape 2: JSON objects carrying {"name": …, "arguments"/"parameters": …}
    for match in _JSON_NAME_RE.finditer(text):
        obj_start = text.rfind("{", 0, match.start())
        if obj_start == -1:
            continue
        raw = _balanced_json_after(text, obj_start)
        data = _safe_json_loads(raw) if raw else None
        if not isinstance(data, dict):
            continue
        args = data.get("arguments", data.get("parameters", {}))
        if isinstance(args, str):
            args = _safe_json_loads(args) or {}
        add(match.group(1), args if isinstance(args, dict) else {})

    return calls


def _balanced_json_after(text: str, start: int) -> str | None:
    """Return the first brace-balanced JSON object at/after ``start``.

    Counts braces while respecting string literals/escapes so nested objects
    are captured correctly (a naive non-greedy regex would truncate them).
    """

    brace = text.find("{", start)
    if brace == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(brace, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace : i + 1]
    return None


def _safe_json_loads(raw: str) -> object | None:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
