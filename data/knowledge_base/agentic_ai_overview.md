# Agentic AI — Overview

Agentic AI refers to systems where a Large Language Model (LLM) acts as the
reasoning engine of an *agent* that can plan, choose actions, use tools, and
observe results — repeating this loop until a goal is met. Unlike a single
prompt-response call, an agent decides *what to do next* at each step.

## Core components of an AI agent

1. **Model (the reasoner):** an LLM that interprets the goal and decides the
   next action.
2. **Tools:** external capabilities the agent can call — web search, a
   database, a calculator, an API, or retrieval over documents.
3. **Memory:** short-term (the current conversation) and long-term (persisted
   facts or vector stores) context the agent can draw on.
4. **Orchestration / control loop:** the logic that runs the
   reason → act → observe cycle and decides when to stop.

## The ReAct pattern

ReAct ("Reasoning + Acting") interleaves thinking and tool use. The model
produces a thought, optionally calls a tool, reads the tool's output
(observation), and then thinks again. This continues until it can answer
directly. ReAct is the most common foundation for tool-using agents.

## Retrieval-Augmented Generation (RAG)

RAG grounds an agent's answers in a trusted corpus. Documents are split into
chunks, converted to embeddings (numeric vectors), and stored in a vector
database. At query time the most semantically similar chunks are retrieved and
given to the model as context, reducing hallucination and enabling citations.

## When to use an agent

Use an agent when a task needs multiple steps, external information, or tools —
for example research, data lookup, or multi-stage workflows. For a single,
self-contained transformation (summarize this text), a plain LLM call is
simpler and cheaper.
