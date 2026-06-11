"""Reranking — the single biggest precision win in a RAG pipeline.

A bi-encoder (embeddings) retrieves fast but coarsely; a cross-encoder scores
each (query, document) pair jointly and far more accurately. We retrieve a wide
candidate set cheaply, then rerank it down to the few chunks actually fed to the
model.

The cross-encoder is optional and lazily loaded (it downloads a model on first
use), so the pipeline runs without it when ``RERANK_ENABLED=false``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agentic_research_agent.core.logging import get_logger

if TYPE_CHECKING:
    from agentic_research_agent.config.settings import Settings
    from agentic_research_agent.rag.retriever import RetrievedDoc

logger = get_logger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Re-orders candidate documents by relevance to the query."""

    def rerank(self, query: str, docs: list[RetrievedDoc], top_n: int) -> list[RetrievedDoc]: ...


class CrossEncoderReranker:
    """Rerank with a sentence-transformers cross-encoder."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None  # lazy: defer the model download

    def _ensure_model(self):  # noqa: ANN202 - third-party return type
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, docs: list[RetrievedDoc], top_n: int) -> list[RetrievedDoc]:
        from agentic_research_agent.rag.retriever import RetrievedDoc

        if not docs:
            return []
        model = self._ensure_model()
        scores = model.predict([(query, doc.content) for doc in docs])
        ranked = sorted(zip(docs, scores, strict=False), key=lambda pair: pair[1], reverse=True)
        return [
            RetrievedDoc(doc.content, doc.source, float(score)) for doc, score in ranked[:top_n]
        ]


def build_reranker(settings: Settings) -> Reranker | None:
    """Return a configured reranker, or ``None`` when reranking is disabled."""

    if not settings.rerank_enabled:
        return None
    return CrossEncoderReranker(settings.rerank_model)
