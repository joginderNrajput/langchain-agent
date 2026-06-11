"""Tests for hybrid retrieval (dense + BM25 + RRF) and reranking.

Fusion logic is exercised with an injected fake dense search and an in-memory
corpus — no embeddings, no network.
"""

from __future__ import annotations

from agentic_research_agent.rag.retriever import HybridRetriever, RetrievedDoc


def _docs(*pairs: tuple[str, str]) -> list[RetrievedDoc]:
    return [RetrievedDoc(content=c, source=s) for c, s in pairs]


def test_fusion_unions_and_dedups() -> None:
    corpus = _docs(
        ("python is a language", "py.md"),
        ("bananas are yellow fruit", "fruit.md"),
    )

    def dense(query: str, k: int) -> list[RetrievedDoc]:
        # Dense favours the fruit doc regardless of query (stub).
        return [RetrievedDoc("bananas are yellow fruit", "fruit.md", 0.9)]

    retriever = HybridRetriever(dense, corpus, top_k=5, fetch_k=5)
    results = retriever.retrieve("python language")

    sources = [d.source for d in results]
    assert "py.md" in sources  # found by BM25 (keyword)
    assert "fruit.md" in sources  # found by dense
    assert len(sources) == len(set(sources))  # de-duplicated


def test_bm25_only_when_no_dense() -> None:
    corpus = _docs(("retrieval augmented generation", "rag.md"))

    def dense(query: str, k: int) -> list[RetrievedDoc]:
        return []

    retriever = HybridRetriever(dense, corpus, top_k=3, fetch_k=3)
    results = retriever.retrieve("augmented generation")
    assert [d.source for d in results] == ["rag.md"]


def test_reranker_is_applied() -> None:
    corpus = _docs(("a", "a.md"), ("b", "b.md"))

    def dense(query: str, k: int) -> list[RetrievedDoc]:
        return _docs(("a", "a.md"), ("b", "b.md"))

    class ReverseReranker:
        def rerank(self, query, docs, top_n):  # noqa: ANN001
            return list(reversed(docs))[:top_n]

    retriever = HybridRetriever(dense, corpus, top_k=1, fetch_k=5, reranker=ReverseReranker())
    results = retriever.retrieve("a b")
    assert len(results) == 1  # top_n honoured
