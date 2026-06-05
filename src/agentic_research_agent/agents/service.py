"""High-level service facade.

:class:`ResearchAgent` is the single entry point the rest of the world uses
(CLI today, an HTTP handler tomorrow). It owns construction of the LLM, tools,
knowledge base, checkpointer, and compiled graph, and exposes a small,
stable API — ``ask`` / ``stream`` — that hides all LangGraph wiring.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from time import perf_counter
from types import TracebackType
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agentic_research_agent.agents.checkpointer import build_checkpointer
from agentic_research_agent.agents.graph import build_agent_graph
from agentic_research_agent.config.settings import Settings, get_settings
from agentic_research_agent.core.exceptions import AgentExecutionError
from agentic_research_agent.core.llm import build_chat_model
from agentic_research_agent.core.logging import configure_logging, get_logger
from agentic_research_agent.core.observability import configure_tracing
from agentic_research_agent.schemas.models import (
    AgentRequest,
    AgentResponse,
    ToolCallRecord,
)
from agentic_research_agent.tools import build_toolset
from agentic_research_agent.tools.knowledge_base import KnowledgeBase

logger = get_logger(__name__)


class ResearchAgent:
    """A ready-to-use research assistant.

    Example:
        >>> agent = ResearchAgent()
        >>> agent.ask("What is 2 ** 10?").answer
        '1024'
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        configure_logging(self._settings.log_level, json_logs=self._settings.is_production)
        configure_tracing(self._settings)

        logger.info(
            "Initializing ResearchAgent (provider=%s, model=%s, checkpointer=%s)",
            self._settings.llm_provider.value,
            self._settings.effective_llm_model,
            self._settings.checkpointer.value,
        )

        # Resources that hold connections (durable checkpointers) are opened
        # against this stack and released by close()/the context manager.
        self._stack = ExitStack()
        self._llm = build_chat_model(self._settings)
        self._knowledge_base = KnowledgeBase(self._settings)
        self._knowledge_base.build()  # load or build the vector store

        self._tools = build_toolset(self._settings, self._knowledge_base)
        self._checkpointer = build_checkpointer(self._settings, self._stack)
        self._graph = build_agent_graph(self._llm, self._tools, self._checkpointer)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Release held resources (e.g. a durable checkpointer connection)."""

        self._stack.close()

    def __enter__(self) -> ResearchAgent:
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
        """Return True if the vector store can serve a query."""

        try:
            self._knowledge_base.search("readiness probe")
            return True
        except Exception:  # noqa: BLE001 - readiness must never raise
            logger.warning("knowledge base readiness check failed", exc_info=True)
            return False

    def llm_reachable(self) -> bool:
        """Return True if the chat model was constructed.

        This is a shallow, zero-cost check (no token-spending call). Construction
        already validates credentials via the provider factory; a deep ping can
        be added if liveness of the provider endpoint must be asserted.
        """

        return self._llm is not None

    @property
    def settings(self) -> Settings:
        return self._settings

    # -- public API -----------------------------------------------------------

    def ask(self, question: str, thread_id: str = "default") -> AgentResponse:
        """Answer a question and return a structured response.

        Args:
            question: The user's question.
            thread_id: Conversation id. Reuse the same id to continue a
                session — the agent remembers prior turns on that thread.
        """

        request = AgentRequest(question=question, thread_id=thread_id)
        run_id = str(uuid4())
        started = perf_counter()
        logger.info("agent_run_started run_id=%s thread_id=%s", run_id, request.thread_id)
        try:
            result = self._graph.invoke(
                {"messages": [HumanMessage(content=request.question)]},
                config=self._run_config(request.thread_id),
            )
        except Exception as exc:
            duration_ms = int((perf_counter() - started) * 1000)
            logger.exception(
                "agent_run_failed run_id=%s thread_id=%s duration_ms=%d",
                run_id,
                request.thread_id,
                duration_ms,
            )
            raise AgentExecutionError(
                f"Agent run failed for run_id={run_id}. "
                f"The model provider rejected or failed the request: {exc}"
            ) from exc
        duration_ms = int((perf_counter() - started) * 1000)
        messages = result["messages"]
        answer = _message_text(messages[-1]) if messages else ""
        tool_calls = _collect_tool_calls(messages)
        logger.info(
            "agent_run_finished run_id=%s thread_id=%s duration_ms=%d tool_calls=%d",
            run_id,
            request.thread_id,
            duration_ms,
            len(tool_calls),
        )
        return AgentResponse(
            answer=answer,
            thread_id=request.thread_id,
            run_id=run_id,
            duration_ms=duration_ms,
            tool_calls=tool_calls,
        )

    def stream(self, question: str, thread_id: str = "default") -> Iterator[str]:
        """Yield a human-readable trace of the run as it executes.

        Emits one line per node step (tool calls and the final answer), useful
        for showing progress in a CLI/UI.
        """

        request = AgentRequest(question=question, thread_id=thread_id)
        for chunk in self._graph.stream(
            {"messages": [HumanMessage(content=request.question)]},
            config=self._run_config(request.thread_id),
            stream_mode="values",
        ):
            last = chunk["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                names = ", ".join(tc["name"] for tc in last.tool_calls)
                yield f"[thinking] calling tools: {names}"
            elif isinstance(last, AIMessage) and last.content:
                text = _message_text(last)
                if text:
                    yield text

    # -- internals ------------------------------------------------------------

    def _run_config(self, thread_id: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._settings.recursion_limit,
        }


def _collect_tool_calls(messages: list) -> list[ToolCallRecord]:
    """Extract every tool call made across the message history."""

    records: list[ToolCallRecord] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            records.append(ToolCallRecord(name=call["name"], args=call.get("args", {})))
    return records


def _message_text(message: object) -> str:
    """Return the plain text of a message.

    Providers differ: some (Groq, OpenAI) put a plain string in ``content``,
    while others (Gemini, Anthropic) return a list of content blocks such as
    ``[{"type": "text", "text": "…"}]``. This normalises both to a clean
    string so the API never leaks raw block dictionaries to callers.
    """

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts).strip()
    return str(content)
