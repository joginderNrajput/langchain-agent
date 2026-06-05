"""Tests for the LangGraph agent wiring.

These use a lightweight fake model (no network, no API key) to verify the
graph's control flow: the no-tool path ends immediately, and the tool path
executes a real tool and loops back before answering.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agentic_research_agent.agents.graph import build_agent_graph
from agentic_research_agent.tools.calculator import calculator


class _FakeRunnable:
    """Returns preset responses in order, ignoring the input messages."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)

    def invoke(self, _messages, *_args, **_kwargs) -> AIMessage:
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeLLM:
    """Duck-typed stand-in for BaseChatModel used only in tests."""

    def __init__(self, responses: list[AIMessage | Exception]) -> None:
        self._responses = responses

    def bind_tools(self, _tools) -> _FakeRunnable:
        return _FakeRunnable(self._responses)


def test_direct_answer_path_ends_without_tools() -> None:
    llm = _FakeLLM([AIMessage(content="Paris.")])
    graph = build_agent_graph(llm, [calculator])

    result = graph.invoke({"messages": [HumanMessage(content="Capital of France?")]})

    assert result["messages"][-1].content == "Paris."


def test_tool_call_path_executes_tool_then_answers() -> None:
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculator",
                    "args": {"expression": "2 + 2"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="The answer is 4."),
    ]
    graph = build_agent_graph(_FakeLLM(responses), [calculator])

    result = graph.invoke({"messages": [HumanMessage(content="What is 2 + 2?")]})

    contents = [m.content for m in result["messages"]]
    assert "4" in contents  # the ToolMessage carrying the calculator result
    assert result["messages"][-1].content == "The answer is 4."


def test_groq_failed_generation_tool_call_is_recovered() -> None:
    responses = [
        RuntimeError(
            "Error code: 400 - {'error': {'code': 'tool_use_failed', "
            "'failed_generation': '<function=calculator{\"expression\": \"2 + 2\"}"
            "</function>'}}"
        ),
        AIMessage(content="The answer is 4."),
    ]
    graph = build_agent_graph(_FakeLLM(responses), [calculator])

    result = graph.invoke({"messages": [HumanMessage(content="What is 2 + 2?")]})

    contents = [m.content for m in result["messages"]]
    assert "4" in contents
    assert result["messages"][-1].content == "The answer is 4."


def test_non_tool_use_provider_errors_still_raise() -> None:
    graph = build_agent_graph(_FakeLLM([RuntimeError("provider is down")]), [calculator])

    try:
        graph.invoke({"messages": [HumanMessage(content="What is 2 + 2?")]})
    except RuntimeError as exc:
        assert "provider is down" in str(exc)
    else:
        raise AssertionError("Expected provider error to be raised")
