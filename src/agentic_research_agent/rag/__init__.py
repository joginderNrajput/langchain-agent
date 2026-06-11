"""Advanced retrieval-augmented generation (RAG) building blocks.

This package upgrades plain vector search into a production retrieval pipeline:

    query → (multi-query rewrite) → hybrid search (dense + BM25, RRF-fused)
          → (optional cross-encoder rerank) → top-k cited chunks

Each stage is independently testable and composes behind :class:`HybridRetriever`.
"""

from agentic_research_agent.rag.query_transform import QueryTransformer
from agentic_research_agent.rag.reranker import CrossEncoderReranker, Reranker, build_reranker
from agentic_research_agent.rag.retriever import (
    HybridRetriever,
    RetrievedDoc,
    build_hybrid_retriever,
)

__all__ = [
    "CrossEncoderReranker",
    "HybridRetriever",
    "QueryTransformer",
    "Reranker",
    "RetrievedDoc",
    "build_hybrid_retriever",
    "build_reranker",
]
