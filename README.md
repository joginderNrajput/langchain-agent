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

---

## Run as an HTTP API (production)

The agent also ships as a FastAPI service. The agent is built once at startup
and shared across requests; routes add auth, rate limiting, a request timeout,
health probes, and Prometheus metrics.

```bash
# configure (set GROQ_API_KEY, and API_KEYS to require an X-API-Key header)
cp .env.example .env

# run locally
uv run research-agent-api          # → http://localhost:8000  (docs at /docs)

# or multi-worker via gunicorn (production)
gunicorn agentic_research_agent.api.app:app \
  -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 --timeout 120
```

| Endpoint | Method | Description |
| --- | --- | --- |
| `/v1/ask` | POST | Answer a question → `AgentResponse` (auth + rate-limited). |
| `/v1/stream` | POST | Stream the run as Server-Sent Events. |
| `/health/live` | GET | Liveness probe. |
| `/health/ready` | GET | Readiness (vector store + LLM reachable). |
| `/metrics` | GET | Prometheus metrics. |

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"question": "What is 23 * 19?", "thread_id": "demo"}'
```

### Durable conversation memory

Set `CHECKPOINTER` to persist threads across restarts / replicas:

```bash
CHECKPOINTER=sqlite                              # single node, durable
CHECKPOINTER=postgres POSTGRES_DSN=postgresql://…  # multi-replica (needs the postgres extra)
```

Install the Postgres backend with `uv sync --extra postgres`.

### Containers

```bash
docker build -t agentic-research-agent .
docker compose up --build        # API + Postgres (pgvector image) stack
```

See [docs/production-deployment-guide.html](docs/production-deployment-guide.html)
for deployment strategies, scaling, security, and the full operational guide.

## Multi-agent RAG mode

The project ships two interchangeable agent services, selected by `AGENT_MODE`:

| Mode | What it is |
| --- | --- |
| `single` (default) | A ReAct tool-using agent (web search · knowledge base · calculator). |
| `multiagent` | A **supervisor multi-agent RAG** pipeline: `supervisor → retrieval → synthesis → critic`, with hybrid (dense + BM25) retrieval, multi-query expansion, optional cross-encoder reranking, and a grounding critic that can trigger bounded re-retrieval. |

```bash
# run the multi-agent RAG pipeline (CLI, chat, or API all honour AGENT_MODE)
AGENT_MODE=multiagent uv run research-agent ask "What is RAG? Cite sources."
AGENT_MODE=multiagent uv run research-agent-api
```

Both modes share the same API, auth, checkpointer, metrics, and deployment.
See [docs/multi-agent-rag-guide.html](docs/multi-agent-rag-guide.html) for the
architecture, retrieval pipeline, and tuning knobs, and
[docs/multi-agent-rag-operations.html](docs/multi-agent-rag-operations.html)
for the hands-on **run · deploy · scale** runbook (every command, Docker/K8s
manifests, and scaling strategies).

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
