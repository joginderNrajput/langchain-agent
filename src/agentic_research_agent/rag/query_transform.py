"""Query transformation — improve recall before retrieval.

A single user phrasing often misses relevant chunks worded differently.
Multi-query expansion asks the LLM for a few alternative phrasings; we retrieve
for each and union the results. The transformer degrades gracefully: with no
LLM (or on any error) it returns just the original query, so retrieval still
works.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_research_agent.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You rewrite a user question into diverse search queries that improve "
    "document retrieval. Return ONLY the queries, one per line, no numbering "
    "or commentary."
)


class QueryTransformer:
    """Expand a question into multiple search queries via the LLM."""

    def __init__(self, llm: BaseChatModel | None, *, max_queries: int = 3) -> None:
        self._llm = llm
        self._max_queries = max(1, max_queries)

    def expand(self, question: str) -> list[str]:
        """Return the original question plus up to ``max_queries-1`` rewrites.

        The original is always first and always included so behaviour never
        regresses below plain single-query retrieval.
        """

        queries = [question.strip()]
        if self._llm is None or self._max_queries == 1:
            return queries

        try:
            response = self._llm.invoke(
                [
                    SystemMessage(content=_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Question: {question}\n"
                            f"Produce {self._max_queries - 1} alternative search queries."
                        )
                    ),
                ]
            )
            text = response.content if isinstance(response.content, str) else str(response.content)
        except Exception as exc:  # noqa: BLE001 - never block retrieval
            logger.warning("query expansion failed, using original only: %s", exc)
            return queries

        for line in text.splitlines():
            cleaned = line.strip().lstrip("0123456789.-) ").strip()
            if cleaned and cleaned.lower() != question.strip().lower():
                queries.append(cleaned)
            if len(queries) >= self._max_queries:
                break
        return queries
