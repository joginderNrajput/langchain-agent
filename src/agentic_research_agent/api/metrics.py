"""Prometheus metrics.

A small, explicit set of metrics covering the golden signals (traffic, errors,
latency) plus agent-specific counters (tokens are out of scope here; tool usage
and run outcomes are tracked). Exposed at ``/metrics`` for scraping.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "research_agent_requests_total",
    "Total API requests.",
    labelnames=("endpoint", "status"),
)

RUN_LATENCY = Histogram(
    "research_agent_run_duration_seconds",
    "End-to-end agent run latency.",
    buckets=(0.5, 1, 2, 4, 8, 16, 32, 64),
)

TOOL_CALLS = Counter(
    "research_agent_tool_calls_total",
    "Tool invocations made by the agent.",
    labelnames=("tool",),
)

RUN_OUTCOMES = Counter(
    "research_agent_runs_total",
    "Agent run outcomes.",
    labelnames=("outcome",),  # success | error
)
