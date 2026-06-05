"""Domain-specific exception hierarchy.

A single base (:class:`AgentError`) lets callers catch every error this package
raises with one ``except`` clause, while the subclasses allow precise handling
where it matters (e.g. surfacing a configuration problem differently from a
transient provider failure).
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all errors raised by the agent package."""


class ConfigurationError(AgentError):
    """Raised when configuration is missing or invalid (e.g. no API key)."""


class LLMProviderError(AgentError):
    """Raised when an LLM provider cannot be constructed or invoked."""


class AgentExecutionError(AgentError):
    """Raised when an agent run fails after startup."""


class KnowledgeBaseError(AgentError):
    """Raised when the knowledge base cannot be built or queried."""
