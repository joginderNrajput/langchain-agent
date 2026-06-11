"""Multi-agent RAG.

A LangGraph supervisor pipeline that decomposes a question across specialist
agents:

    supervisor (route) → retrieval → synthesis → critic ─┐
            │ direct                                       │ not grounded
            ▼                                              ▼ (≤ max_revisions)
        synthesis ─────────────────────────────────────► retrieval

The retrieval agent uses hybrid search + optional reranking; the critic checks
that the drafted answer is grounded in the retrieved context and can trigger a
bounded re-retrieval round before the answer is returned.
"""

from agentic_research_agent.multiagent.graph import build_multiagent_graph
from agentic_research_agent.multiagent.service import MultiAgentRAG
from agentic_research_agent.multiagent.state import MultiAgentState

__all__ = ["MultiAgentRAG", "MultiAgentState", "build_multiagent_graph"]
