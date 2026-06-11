"""Multi-agent RAG service facade.

:class:`MultiAgentRAG` exposes the same surface as
:class:`~agentic_research_agent.agents.service.ResearchAgent`
(``ask`` / ``stream`` / health / ``close``), so the CLI and HTTP API treat the
two interchangeably — selected by ``AGENT_MODE``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from time import perf_counter
from types import TracebackType
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agentic_research_agent.agents.checkpointer import build_checkpointer
from agentic_research_agent.config.settings import Settings, get_settings
from agentic_research_agent.core.exceptions import AgentExecutionError
from agentic_research_agent.core.llm import build_chat_model
from agentic_research_agent.core.logging import configure_logging, get_logger
from agentic_research_agent.core.observability import configure_tracing
from agentic_research_agent.multiagent.graph import build_multiagent_graph
from agentic_research_agent.rag.query_transform import QueryTransformer
from agentic_research_agent.rag.reranker import build_reranker
from agentic_research_agent.rag.retriever import build_hybrid_retriever
from agentic_research_agent.schemas.models import AgentRequest, AgentResponse, ToolCallRecord
from agentic_research_agent.tools.knowledge_base import KnowledgeBase

logger = get_logger(__name__)


class MultiAgentRAG:
    """Supervisor multi-agent RAG assistant."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        configure_logging(self._settings.log_level, json_logs=self._settings.is_production)
        configure_tracing(self._settings)

        logger.info(
            "Initializing MultiAgentRAG (provider=%s, model=%s, rerank=%s, multiquery=%s)",
            self._settings.llm_provider.value,
            self._settings.effective_llm_model,
            self._settings.rerank_enabled,
            self._settings.multiquery_enabled,
        )

        self._stack = ExitStack()
        self._llm = build_chat_model(self._settings)
        self._knowledge_base = KnowledgeBase(self._settings)
        self._knowledge_base.build()

        reranker = build_reranker(self._settings)
        self._retriever = build_hybrid_retriever(
            self._knowledge_base,
            top_k=self._settings.retriever_top_k,
            fetch_k=self._settings.retriever_fetch_k,
            reranker=reranker,
        )
        self._query_expander = QueryTransformer(
            self._llm if self._settings.multiquery_enabled else None,
            max_queries=self._settings.multiquery_count,
        )
        self._checkpointer = build_checkpointer(self._settings, self._stack)
        self._graph = build_multiagent_graph(
            self._llm,
            self._retriever,
            self._query_expander,
            max_revisions=self._settings.max_revisions,
            checkpointer=self._checkpointer,
        )

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> MultiAgentRAG:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- health ---------------------------------------------------------------

    def knowledge_base_ready(self) -> bool:
        try:
            self._knowledge_base.search("readiness probe")
            return True
        except Exception:  # noqa: BLE001 - readiness must never raise
            logger.warning("knowledge base readiness check failed", exc_info=True)
            return False

    def llm_reachable(self) -> bool:
        return self._llm is not None

    @property
    def settings(self) -> Settings:
        return self._settings

    # -- public API -----------------------------------------------------------

    def ask(self, question: str, thread_id: str = "default") -> AgentResponse:
        request = AgentRequest(question=question, thread_id=thread_id)
        run_id = str(uuid4())
        started = perf_counter()
        logger.info("multiagent_run_started run_id=%s thread_id=%s", run_id, request.thread_id)
        try:
            result = self._graph.invoke(
                self._initial_state(request.question), self._run_config(request.thread_id)
            )
        except Exception as exc:
            duration_ms = int((perf_counter() - started) * 1000)
            logger.exception("multiagent_run_failed run_id=%s duration_ms=%d", run_id, duration_ms)
            raise AgentExecutionError(f"Multi-agent run failed for run_id={run_id}: {exc}") from exc

        duration_ms = int((perf_counter() - started) * 1000)
        answer = result.get("answer") or ""
        tool_calls = _summarize_steps(result)
        logger.info(
            "multiagent_run_finished run_id=%s thread_id=%s duration_ms=%d route=%s docs=%d",
            run_id,
            request.thread_id,
            duration_ms,
            result.get("route"),
            len(result.get("documents") or []),
        )
        return AgentResponse(
            answer=answer,
            thread_id=request.thread_id,
            run_id=run_id,
            duration_ms=duration_ms,
            tool_calls=tool_calls,
        )

    def stream(self, question: str, thread_id: str = "default") -> Iterator[str]:
        request = AgentRequest(question=question, thread_id=thread_id)
        for update in self._graph.stream(
            self._initial_state(request.question),
            self._run_config(request.thread_id),
            stream_mode="updates",
        ):
            for node, payload in update.items():
                yield _step_label(node, payload)
        # Final answer is the last synthesised draft; re-read from the graph state.
        snapshot = self._graph.get_state(self._run_config(request.thread_id)).values
        if snapshot.get("answer"):
            yield str(snapshot["answer"])

    # -- internals ------------------------------------------------------------

    def _initial_state(self, question: str) -> dict:
        return {
            "question": question,
            "messages": [HumanMessage(content=question)],
            "revisions": 0,
            "documents": [],
            "critique": "",
        }

    def _run_config(self, thread_id: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._settings.recursion_limit,
        }


def _summarize_steps(result: dict) -> list[ToolCallRecord]:
    """Represent the agents that ran as tool-call records (for observability)."""

    records: list[ToolCallRecord] = []
    documents = result.get("documents") or []
    if documents:
        records.append(
            ToolCallRecord(
                name="retrieval",
                args={"chunks": len(documents), "revisions": result.get("revisions", 0)},
            )
        )
    return records


def _step_label(node: str, payload: dict) -> str:
    if node == "supervisor":
        return f"[supervisor] route={payload.get('route', '?')}"
    if node == "retrieve":
        return f"[retrieval] {len(payload.get('documents') or [])} chunks"
    if node == "synthesize":
        return "[synthesis] drafted answer"
    if node == "critic":
        return f"[critic] grounded={payload.get('grounded')}"
    return f"[{node}]"
