"""Tests for multi-query expansion (with graceful fallback)."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agentic_research_agent.rag.query_transform import QueryTransformer


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self._text = text

    def invoke(self, _messages):  # noqa: ANN001
        return AIMessage(content=self._text)


def test_passthrough_without_llm() -> None:
    qt = QueryTransformer(None, max_queries=3)
    assert qt.expand("what is RAG?") == ["what is RAG?"]


def test_expansion_parses_and_keeps_original_first() -> None:
    llm = _FakeLLM("1. define retrieval augmented generation\n2. how does RAG work")
    qt = QueryTransformer(llm, max_queries=3)
    out = qt.expand("what is RAG?")
    assert out[0] == "what is RAG?"
    assert len(out) == 3
    assert any("retrieval augmented generation" in q for q in out)


def test_expansion_failure_falls_back() -> None:
    class Boom:
        def invoke(self, _messages):  # noqa: ANN001
            raise RuntimeError("provider down")

    qt = QueryTransformer(Boom(), max_queries=3)
    assert qt.expand("q") == ["q"]
