"""Agentic Research Agent.

An enterprise-grade, provider-agnostic AI research assistant built on
LangChain + LangGraph. It answers questions using a ReAct-style reasoning
loop with three tools: web search, an internal knowledge base (RAG), and a
safe calculator.
"""

from __future__ import annotations

__version__ = "0.1.0"

from agentic_research_agent.agents.service import ResearchAgent

__all__ = ["ResearchAgent", "__version__"]
