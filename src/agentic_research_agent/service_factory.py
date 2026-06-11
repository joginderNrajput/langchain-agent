"""Select the agent implementation from configuration.

``AGENT_MODE=single`` builds the ReAct tool-using agent; ``AGENT_MODE=multiagent``
builds the supervisor multi-agent RAG pipeline. Concrete services are imported
lazily so importing this module stays cheap and free of import cycles.
"""

from __future__ import annotations

from agentic_research_agent.agents.base import AgentService
from agentic_research_agent.config.settings import AgentMode, Settings, get_settings
from agentic_research_agent.core.logging import get_logger

logger = get_logger(__name__)


def build_agent_service(settings: Settings | None = None) -> AgentService:
    """Construct the agent service named by ``settings.agent_mode``."""

    settings = settings or get_settings()
    if settings.agent_mode is AgentMode.MULTIAGENT:
        logger.info("Selected agent mode: multiagent (RAG supervisor pipeline)")
        from agentic_research_agent.multiagent.service import MultiAgentRAG

        return MultiAgentRAG(settings)

    logger.info("Selected agent mode: single (ReAct agent)")
    from agentic_research_agent.agents.service import ResearchAgent

    return ResearchAgent(settings)
