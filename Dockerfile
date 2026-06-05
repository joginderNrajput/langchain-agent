# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-stage build for the Agentic Research Agent API.
#   Stage 1 (builder): resolve and install dependencies into a venv with uv.
#   Stage 2 (runtime):  copy the venv + source, run as a non-root user.
# ---------------------------------------------------------------------------

FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# uv: fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install deps first (cached layer) using only the lock + manifest.
# The 'postgres' extra adds the durable multi-replica checkpointer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra postgres --no-install-project

# Now copy source and install the project itself.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra postgres


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    PATH="/app/.venv/bin:$PATH"

# curl is used by the container HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app

# Bring in the resolved environment and application code.
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY data/knowledge_base ./data/knowledge_base

# Pre-warm the embedding model into the image so the first request is not slow.
# (Downloads the model defined by EMBEDDING_MODEL; safe to remove to slim build.)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && mkdir -p data/vector_store && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

# Multiple workers behind uvicorn workers; tune -w to the host's CPUs.
CMD ["gunicorn", "agentic_research_agent.api.app:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "4", "-b", "0.0.0.0:8000", "--timeout", "120"]
