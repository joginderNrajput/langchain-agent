"""Tests for the multi-agent RAG graph routing.

Uses a fake LLM (queued responses by call order) and a fake retriever — no
embeddings, no network, no API key.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agentic_research_agent.multiagent.graph import build_multiagent_graph
from agentic_research_agent.rag.retriever import RetrievedDoc


class _FakeLLM:
    """Returns queued AIMessages in invocation order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def invoke(self, _messages):  # noqa: ANN001
        return AIMessage(content=self._responses.pop(0))


class _FakeRetriever:
    def __init__(self, docs: list[RetrievedDoc]) -> None:
        self._docs = docs

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        return self._docs


class _PassThroughExpander:
    def expand(self, question: str) -> list[str]:
        return [question]


_DOCS = [RetrievedDoc("LangGraph builds stateful agents.", "langgraph.md", 0.9)]


def test_retrieve_path_grounded() -> None:
    llm = _FakeLLM(["RETRIEVE", "LangGraph builds stateful agents [1].", "GROUNDED\nlooks good"])
    graph = build_multiagent_graph(
        llm, _FakeRetriever(_DOCS), _PassThroughExpander(), max_revisions=1
    )
    result = graph.invoke({"question": "What is LangGraph?", "revisions": 0})
    assert "LangGraph" in result["answer"]
    assert len(result["documents"]) == 1
    assert result["grounded"] is True


def test_direct_path_skips_retrieval() -> None:
    llm = _FakeLLM(["DIRECT", "Hello! How can I help?"])
    graph = build_multiagent_graph(
        llm, _FakeRetriever(_DOCS), _PassThroughExpander(), max_revisions=1
    )
    result = graph.invoke({"question": "hi", "revisions": 0})
    assert result["answer"].startswith("Hello")
    assert not result.get("documents")  # retrieval was skipped


def test_critic_triggers_one_revision_then_stops() -> None:
    # not grounded once → re-retrieve → grounded; bounded by max_revisions=1.
    llm = _FakeLLM(
        [
            "RETRIEVE",
            "draft one",
            "NOT_GROUNDED\nmissing detail X",
            "draft two [1]",
            "GROUNDED\nnow supported",
        ]
    )
    graph = build_multiagent_graph(
        llm, _FakeRetriever(_DOCS), _PassThroughExpander(), max_revisions=1
    )
    result = graph.invoke({"question": "explain X", "revisions": 0})
    assert result["answer"] == "draft two [1]"
    assert result["revisions"] == 2  # two critic passes


def test_revision_budget_is_respected() -> None:
    # Always "not grounded"; with max_revisions=0 it must stop after first critic.
    llm = _FakeLLM(["RETRIEVE", "draft", "NOT_GROUNDED\nstill missing"])
    graph = build_multiagent_graph(
        llm, _FakeRetriever(_DOCS), _PassThroughExpander(), max_revisions=0
    )
    result = graph.invoke({"question": "q", "revisions": 0})
    assert result["answer"] == "draft"
    assert result["grounded"] is False  # gave up, returned best effort
