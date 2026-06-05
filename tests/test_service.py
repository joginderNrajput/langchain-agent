"""Tests for service-layer helpers."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agentic_research_agent.agents.service import _message_text


def test_plain_string_content() -> None:
    assert _message_text(AIMessage(content="hello world")) == "hello world"


def test_block_list_content_is_flattened() -> None:
    # Gemini / Anthropic style: content is a list of typed blocks.
    msg = AIMessage(
        content=[
            {"type": "text", "text": "The answer is "},
            {"type": "text", "text": "437.", "extras": {"signature": "abc"}},
        ]
    )
    assert _message_text(msg) == "The answer is 437."


def test_mixed_and_empty_blocks() -> None:
    msg = AIMessage(content=["a", {"type": "text", "text": "b"}, {"type": "image"}])
    assert _message_text(msg) == "ab"
