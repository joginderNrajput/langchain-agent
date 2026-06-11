"""The multi-agent RAG graph (LangGraph).

A supervisor routes the question; a retrieval agent gathers cited context via
hybrid search; a synthesis agent drafts a grounded answer; a critic verifies it
and can trigger a bounded re-retrieval round. Control flow:

    START → supervisor ─(retrieve)→ retrieve → synthesize → critic ─(end)→ END
                  │ (direct)                        ▲           │ (not grounded,
                  └────────────────► synthesize ────┘           └─► retrieve
                                       (no docs → END)             ≤ max_revisions)

Nodes call the LLM directly (no tool binding); the retriever and query
transformer are injected so the graph is unit-testable with fakes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_research_agent.agents.service import _message_text
from agentic_research_agent.multiagent.prompts import (
    CRITIC_PROMPT,
    SUPERVISOR_PROMPT,
    SYNTHESIS_DIRECT_PROMPT,
    SYNTHESIS_PROMPT,
)
from agentic_research_agent.multiagent.state import MultiAgentState, RetrievedChunk
from agentic_research_agent.rag.retriever import RetrievedDoc

# Cap how many fused chunks are passed to the model as context.
_MAX_CONTEXT_CHUNKS = 8


class _Retriever(Protocol):
    def retrieve(self, query: str) -> list[RetrievedDoc]: ...


class _QueryExpander(Protocol):
    def expand(self, question: str) -> list[str]: ...


def _format_context(documents: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {d['source']})\n{d['content'].strip()}"
        for i, d in enumerate(documents, start=1)
    )


def build_multiagent_graph(
    llm: BaseChatModel,
    retriever: _Retriever,
    query_expander: _QueryExpander,
    *,
    max_revisions: int = 1,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the supervisor multi-agent RAG graph."""

    def supervisor_node(state: MultiAgentState) -> dict:
        question = state["question"]
        decision = _message_text(
            llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=question)])
        )
        route = "direct" if decision.strip().upper().startswith("DIRECT") else "retrieve"
        # Reset per-run scratch state so a reused thread doesn't leak context.
        return {"route": route, "revisions": 0, "documents": [], "critique": ""}

    def retrieve_node(state: MultiAgentState) -> dict:
        question = state["question"]
        queries = query_expander.expand(question)
        critique = state.get("critique") or ""
        if critique:
            queries = [*queries, f"{question} (focus: {critique})"]

        merged: dict[str, RetrievedDoc] = {}
        for query in queries:
            for doc in retriever.retrieve(query):
                merged.setdefault(doc.key, doc)
        ranked = sorted(merged.values(), key=lambda d: d.score, reverse=True)
        documents = [
            {"content": d.content, "source": d.source, "score": d.score}
            for d in ranked[:_MAX_CONTEXT_CHUNKS]
        ]
        return {"queries": queries, "documents": documents}

    def synthesize_node(state: MultiAgentState) -> dict:
        question = state["question"]
        documents = state.get("documents") or []
        if not documents:
            draft = _message_text(
                llm.invoke(
                    [
                        SystemMessage(content=SYNTHESIS_DIRECT_PROMPT),
                        HumanMessage(content=question),
                    ]
                )
            )
            return {"draft": draft, "answer": draft, "messages": [AIMessage(content=draft)]}

        prompt = f"Context:\n{_format_context(documents)}\n\nQuestion: {question}"
        draft = _message_text(
            llm.invoke([SystemMessage(content=SYNTHESIS_PROMPT), HumanMessage(content=prompt)])
        )
        return {"draft": draft, "answer": draft, "messages": [AIMessage(content=draft)]}

    def critic_node(state: MultiAgentState) -> dict:
        documents = state.get("documents") or []
        prompt = f"CONTEXT:\n{_format_context(documents)}\n\nDRAFT:\n{state.get('draft', '')}"
        verdict = _message_text(
            llm.invoke([SystemMessage(content=CRITIC_PROMPT), HumanMessage(content=prompt)])
        )
        lines = verdict.splitlines()
        grounded = bool(lines) and lines[0].strip().upper().startswith("GROUNDED")
        feedback = "\n".join(lines[1:]).strip()
        return {
            "grounded": grounded,
            "critique": feedback,
            "revisions": state.get("revisions", 0) + 1,
        }

    def route_after_supervisor(state: MultiAgentState) -> str:
        return state.get("route", "retrieve")

    def route_after_synthesis(state: MultiAgentState) -> str:
        return "critic" if state.get("documents") else "end"

    def route_after_critic(state: MultiAgentState) -> str:
        if state.get("grounded") or state.get("revisions", 0) > max_revisions:
            return "end"
        return "retrieve"

    builder: StateGraph = StateGraph(MultiAgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("critic", critic_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor", route_after_supervisor, {"retrieve": "retrieve", "direct": "synthesize"}
    )
    builder.add_edge("retrieve", "synthesize")
    builder.add_conditional_edges(
        "synthesize", route_after_synthesis, {"critic": "critic", "end": END}
    )
    builder.add_conditional_edges(
        "critic", route_after_critic, {"retrieve": "retrieve", "end": END}
    )
    return builder.compile(checkpointer=checkpointer)


# Re-exported for callers that build node-routing tests.
RouteFn = Callable[[MultiAgentState], str]
