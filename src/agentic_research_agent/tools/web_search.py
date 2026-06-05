"""Web search tool backed by DuckDuckGo (no API key required).

We expose our own ``@tool`` so we control the name, the description the LLM
sees, and — importantly — failure handling. Public search endpoints are
rate-limited and occasionally flaky; a tool that raised would abort the whole
agent run, so instead we catch errors and return a message the model can reason
about and recover from.

Uses the standalone ``duckduckgo_search`` package directly rather than the
``langchain-community`` wrapper, which is being sunset.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from langchain_core.tools import BaseTool, tool

from agentic_research_agent.core.logging import get_logger

logger = get_logger(__name__)

try:
    DDGS: Any = import_module("ddgs").DDGS
except ImportError:  # pragma: no cover - compatibility for older installs
    DDGS = import_module("duckduckgo_search").DDGS


def build_web_search_tool(max_results: int = 5) -> BaseTool:
    """Create a web-search tool capped at ``max_results`` results."""

    @tool
    def web_search(query: str) -> str:
        """Search the public web for current, real-world information.

        Use this for recent events, facts likely to change over time, or any
        topic not covered by the internal knowledge base. Returns a list of
        result snippets with titles and source links.
        """

        logger.info("web_search query=%r", query)
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001 - surface as recoverable text
            logger.warning("web_search failed: %s", exc)
            return (
                f"Web search is temporarily unavailable ({exc}). "
                f"Try rephrasing or rely on other tools."
            )

        if not results:
            return f"No web results found for {query!r}."

        lines = []
        for i, item in enumerate(results, start=1):
            title = item.get("title", "(no title)")
            body = item.get("body", "")
            href = item.get("href", "")
            lines.append(f"{i}. {title}\n   {body}\n   Source: {href}")
        return "\n".join(lines)

    return web_search
