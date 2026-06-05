# Agentic Research Agent

An **enterprise-grade, provider-agnostic AI research assistant** built with
**LangChain** and **LangGraph**. It answers questions using a ReAct-style
reasoning loop with three tools:

- 🔎 **Web search** (DuckDuckGo — no API key required)
- 📚 **Knowledge base** — Retrieval-Augmented Generation (RAG) over your own documents
- 🧮 **Calculator** — safe, exact arithmetic

It ships as a clean reference for how to structure a real agent project:
typed configuration, a swappable LLM provider, a layered package, tests, and a
CLI.

---

## Architecture

```
                ┌──────────────────────────────────────────┐
                │                  CLI                       │
                │        (ask · chat · ingest)               │
                └─────────────────────┬──────────────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   ResearchAgent (service) │   ← public facade
                         └────────────┬─────────────┘
                                      │
        ┌─────────────────┬───────────┼────────────────┬─────────────────┐
        │                 │           │                │                 │
   ┌────▼────┐      ┌──────▼─────┐ ┌───▼────┐     ┌──────▼──────┐   ┌──────▼──────┐
   │  LLM    │      │  LangGraph │ │ Tools  │     │  Knowledge  │   │   Config    │
   │ factory │      │   (graph)  │ │        │     │  base (RAG) │   │ (settings)  │
   └─────────┘      └────────────┘ └────────┘     └─────────────┘   └─────────────┘

   LangGraph control flow:   START → agent → (tool calls?) ─yes→ tools → agent → … → END
```

### Project layout

```
src/agentic_research_agent/
├── config/        # pydantic-settings: typed, env-driven configuration
├── core/          # LLM provider factory, logging, exception hierarchy
├── tools/         # calculator, web_search, knowledge_base (RAG)
├── agents/        # graph (LangGraph), state, prompts, service facade
├── schemas/       # pydantic request/response contracts
└── cli.py         # Typer CLI (console script: `research-agent`)
data/
├── knowledge_base/   # source documents (RAG corpus)
└── vector_store/     # generated Chroma index (gitignored)
tests/                # pytest suite (no network / no API key required)
```

**Design principles**

- **Separation of concerns** — config, infrastructure, tools, and orchestration
  are distinct layers with one-directional dependencies.
- **Provider-agnostic** — the app depends on LangChain's `BaseChatModel`; the
  only place a concrete provider is chosen is [`core/llm.py`](src/agentic_research_agent/core/llm.py).
- **Configuration over code** — every knob lives in [`config/settings.py`](src/agentic_research_agent/config/settings.py)
  and is set via environment variables; nothing secret is hard-coded.
- **Explicit control flow** — the agent loop is a hand-written LangGraph graph
  (not a black box), so it's reviewable and testable.

---

## Quickstart

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure your API key

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY=...   (free key: https://console.groq.com/keys)
```

### 3. Build the knowledge base

The first run downloads the local embedding model (~80 MB) once, then indexes
the documents in `data/knowledge_base/`.

```bash
uv run research-agent ingest
```

### 4. Ask a question

```bash
uv run research-agent ask "What is the ReAct pattern in agentic AI?"
```

### 5. Or chat interactively (with memory)

```bash
uv run research-agent chat
```

---

## CLI reference

| Command                         | Description                                        |
| ------------------------------- | -------------------------------------------------- |
| `research-agent ask "<q>"`      | Answer a single question and exit.                 |
| `research-agent chat`           | Interactive REPL; remembers the conversation.      |
| `research-agent ingest [-f]`    | Build/rebuild the knowledge base (`-f` to force).  |

Use `-t/--thread <id>` on `ask`/`chat` to keep separate conversation memories.

## Use it as a library

```python
from agentic_research_agent import ResearchAgent

agent = ResearchAgent()
response = agent.ask("Compute sqrt(144) and explain RAG in one sentence.")
print(response.answer)
print("tools used:", [t.name for t in response.tool_calls])
```

## Switching LLM provider

Set two env vars — no code change:

```bash
# Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=...

# Local Ollama
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
```

## Development

```bash
uv run pytest          # run the test suite
uv run ruff check .    # lint
uv run ruff format .   # format
uv run mypy            # type-check
```

The test suite runs fully offline — the graph tests use a fake model, so no API
key or network access is needed.

## Adding your own documents

Drop `.md`, `.txt`, or `.pdf` files into `data/knowledge_base/`, then rebuild:

```bash
uv run research-agent ingest --force
```
