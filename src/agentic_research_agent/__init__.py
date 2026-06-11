"""Agentic Research Agent.

An enterprise-grade, provider-agnostic AI assistant built on LangChain +
LangGraph. It ships two interchangeable agent services:

* ``ResearchAgent`` — a ReAct tool-using agent (web search, knowledge base, calculator).
* ``MultiAgentRAG`` — a supervisor multi-agent RAG pipeline (retrieval → synthesis → critic).

``build_agent_service()`` returns whichever ``AGENT_MODE`` selects.
"""

from __future__ import annotations

__version__ = "0.1.0"

from agentic_research_agent.agents.base import AgentService
from agentic_research_agent.agents.service import ResearchAgent
from agentic_research_agent.service_factory import build_agent_service

__all__ = ["AgentService", "ResearchAgent", "build_agent_service", "__version__"]
