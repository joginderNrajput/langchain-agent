"""Agent tools.

Each tool is a small, independently-testable capability the agent may call.
:func:`build_toolset` assembles the full list the graph binds to the LLM.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from agentic_research_agent.config.settings import Settings
from agentic_research_agent.tools.calculator import calculator
from agentic_research_agent.tools.knowledge_base import KnowledgeBase
from agentic_research_agent.tools.web_search import build_web_search_tool

__all__ = [
    "KnowledgeBase",
    "build_toolset",
    "build_web_search_tool",
    "calculator",
]


def build_toolset(settings: Settings, knowledge_base: KnowledgeBase) -> list[BaseTool]:
    """Return the ordered list of tools exposed to the agent."""

    return [
        calculator,
        build_web_search_tool(max_results=settings.max_search_results),
        knowledge_base.as_tool(),
    ]
