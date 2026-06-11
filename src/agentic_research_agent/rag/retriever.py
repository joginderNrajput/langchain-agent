"""Hybrid retrieval: dense + sparse, fused with Reciprocal Rank Fusion.

Pure-vector search misses exact-term and rare-keyword matches; pure-keyword
search misses paraphrases. Combining a dense retriever (embeddings) with a
sparse one (BM25) and fusing their rankings with RRF is a strong, well-proven
default that beats either alone on most corpora.

The class is decoupled from any concrete store: it takes a ``dense_search``
callable and an in-memory ``corpus`` for BM25, so the fusion logic is unit
testable without embeddings or a network. :func:`build_hybrid_retriever`
adapts a :class:`KnowledgeBase` into those inputs.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentic_research_agent.core.logging import get_logger
from agentic_research_agent.rag.reranker import Reranker

logger = get_logger(__name__)

# Type of the injected dense search: (query, k) -> ranked docs (best first).
DenseSearch = Callable[[str, int], "list[RetrievedDoc]"]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class RetrievedDoc:
    """A retrieved chunk with its source and a fusion/relevance score."""

    content: str
    source: str
    score: float = 0.0

    @property
    def key(self) -> str:
        """Stable identity for fusion/dedup (source + content)."""

        return f"{self.source}::{self.content}"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    """Fuse dense and sparse retrieval, then optionally rerank."""

    def __init__(
        self,
        dense_search: DenseSearch,
        corpus: Sequence[RetrievedDoc],
        *,
        top_k: int = 4,
        fetch_k: int = 10,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
    ) -> None:
        self._dense_search = dense_search
        self._corpus = list(corpus)
        self._top_k = top_k
        self._fetch_k = fetch_k
        self._rrf_k = rrf_k
        self._reranker = reranker

        # Build the BM25 index once over the corpus (lazily imported).
        self._bm25 = None
        if self._corpus:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([_tokenize(doc.content) for doc in self._corpus])

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        """Return the top-k fused (and optionally reranked) documents."""

        dense = self._dense_search(query, self._fetch_k)
        sparse = self._bm25_search(query, self._fetch_k)
        fused = self._reciprocal_rank_fusion([dense, sparse])

        if self._reranker is not None and fused:
            return self._reranker.rerank(query, fused, self._top_k)
        return fused[: self._top_k]

    # -- internals ------------------------------------------------------------

    def _bm25_search(self, query: str, k: int) -> list[RetrievedDoc]:
        if self._bm25 is None:
            return []
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[RetrievedDoc] = []
        for idx in ranked[:k]:
            doc = self._corpus[idx]
            # Require lexical overlap rather than a positive BM25 score: IDF can
            # be non-positive on small corpora, and RRF fuses by rank, not score.
            if not (query_tokens & set(_tokenize(doc.content))):
                continue
            out.append(RetrievedDoc(doc.content, doc.source, float(scores[idx])))
        return out

    def _reciprocal_rank_fusion(self, rankings: list[list[RetrievedDoc]]) -> list[RetrievedDoc]:
        """Fuse multiple ranked lists. RRF score = Σ 1 / (k + rank)."""

        scores: dict[str, float] = {}
        docs: dict[str, RetrievedDoc] = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                scores[doc.key] = scores.get(doc.key, 0.0) + 1.0 / (self._rrf_k + rank)
                docs.setdefault(doc.key, doc)
        ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
        return [
            RetrievedDoc(docs[key].content, docs[key].source, round(scores[key], 6))
            for key in ordered
        ]


def build_hybrid_retriever(
    knowledge_base,  # noqa: ANN001 - avoids a circular import with tools.knowledge_base
    *,
    top_k: int = 4,
    fetch_k: int = 10,
    reranker: Reranker | None = None,
) -> HybridRetriever:
    """Adapt a :class:`KnowledgeBase` into a :class:`HybridRetriever`."""

    def _source(meta: dict) -> str:
        return Path(meta.get("source", "unknown")).name

    def dense_search(query: str, k: int) -> list[RetrievedDoc]:
        pairs = knowledge_base.search_with_scores(query, k)
        # Chroma returns distance (lower = closer); convert to a similarity.
        return [
            RetrievedDoc(doc.page_content, _source(doc.metadata), 1.0 / (1.0 + dist))
            for doc, dist in pairs
        ]

    corpus = [
        RetrievedDoc(doc.page_content, _source(doc.metadata))
        for doc in knowledge_base.iter_chunks()
    ]
    logger.debug("Hybrid retriever corpus size: %d chunks", len(corpus))
    return HybridRetriever(dense_search, corpus, top_k=top_k, fetch_k=fetch_k, reranker=reranker)
