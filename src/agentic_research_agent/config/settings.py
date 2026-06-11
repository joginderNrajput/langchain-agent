"""Application settings.

All runtime configuration is centralized here and sourced from environment
variables (and an optional ``.env`` file). This keeps secrets and deployment
knobs out of the code and lets the same image run unchanged across
dev / staging / prod by varying only the environment.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, derived relative to this file so paths work regardless of
# the current working directory:  src/agentic_research_agent/config/settings.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


class LLMProvider(StrEnum):
    """Supported chat-model providers."""

    GROQ = "groq"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class AgentMode(StrEnum):
    """Which agent the service exposes.

    ``single`` is the ReAct tool-using agent; ``multiagent`` is the supervisor
    multi-agent RAG pipeline (retrieval → synthesis → critic).
    """

    SINGLE = "single"
    MULTIAGENT = "multiagent"


class CheckpointerBackend(StrEnum):
    """Where LangGraph conversation state is persisted.

    ``memory`` is process-local (dev/test). ``sqlite`` survives restarts on a
    single node. ``postgres`` is shared across replicas — required for any
    horizontally-scaled deployment.
    """

    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRES = "postgres"


# Per-provider default model, used when LLM_MODEL is not set. Model names are
# provider-specific, so this is what lets you switch providers without also
# remembering to change the model. Override any of these via LLM_MODEL.
DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-6",
    LLMProvider.GEMINI: "gemini-2.5-flash",
    LLMProvider.OLLAMA: "llama3.1",
}


class Settings(BaseSettings):
    """Strongly-typed application configuration.

    Field names map to upper-cased environment variables, e.g. ``llm_provider``
    is read from ``LLM_PROVIDER``. Values are validated on load, so a
    misconfigured deployment fails fast with a clear error instead of
    surfacing a cryptic runtime exception later.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ----- Application -------------------------------------------------------
    app_name: str = "agentic-research-agent"
    environment: str = Field(
        default="development",
        description="Deployment environment: development | staging | production.",
    )
    log_level: str = Field(default="INFO", description="Root log level.")

    # ----- LLM ---------------------------------------------------------------
    llm_provider: LLMProvider = Field(default=LLMProvider.GROQ)
    llm_model: str | None = Field(
        default=None,
        description=(
            "Model name. Leave unset to use the selected provider's default "
            "(see DEFAULT_MODELS); set to override."
        ),
    )
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0)
    llm_timeout_seconds: int = Field(default=60, gt=0)
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        description="Provider-level retries on transient (429/5xx) errors.",
    )

    # ----- Provider credentials ---------------------------------------------
    groq_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    # Gemini: accept either GOOGLE_API_KEY or GEMINI_API_KEY.
    google_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    # Azure OpenAI: the deployment name routes the request (it stands in for the
    # model). All four are required when llm_provider=azure.
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = Field(
        default=None, description="https://<resource>.openai.azure.com/"
    )
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_deployment: str | None = Field(
        default=None, description="Azure deployment name (used as the model)."
    )
    ollama_base_url: str = "http://localhost:11434"

    # ----- Retrieval-Augmented Generation (RAG) ------------------------------
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model for embeddings.",
    )
    # Optional HF token: silences the "unauthenticated requests" notice and
    # raises download rate limits. Read from HF_TOKEN or HUGGINGFACE_API_KEY.
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "HUGGINGFACE_API_KEY"),
    )
    knowledge_base_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    retriever_top_k: int = Field(default=4, gt=0)

    # ----- Agent behaviour ---------------------------------------------------
    agent_mode: AgentMode = Field(
        default=AgentMode.SINGLE,
        description="single (ReAct tools) | multiagent (supervisor RAG pipeline).",
    )
    max_search_results: int = Field(default=5, gt=0)
    recursion_limit: int = Field(
        default=25,
        gt=0,
        description="Max LangGraph super-steps before aborting a run.",
    )

    # ----- Advanced RAG (multi-agent) ----------------------------------------
    retriever_fetch_k: int = Field(
        default=10, gt=0, description="Candidates pulled per retriever before fusion."
    )
    multiquery_enabled: bool = Field(
        default=True, description="Expand the question into several search queries."
    )
    multiquery_count: int = Field(default=3, gt=0)
    rerank_enabled: bool = Field(
        default=False,
        description="Cross-encoder rerank of candidates (downloads a model).",
    )
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_revisions: int = Field(
        default=1,
        ge=0,
        description="Max critic-triggered re-retrieval rounds before answering.",
    )

    # ----- Conversation state (checkpointer) ---------------------------------
    checkpointer: CheckpointerBackend = Field(
        default=CheckpointerBackend.MEMORY,
        description="memory | sqlite | postgres. Use postgres for multi-replica.",
    )
    sqlite_path: Path = PROJECT_ROOT / "data" / "checkpoints.sqlite"
    postgres_dsn: str | None = Field(
        default=None,
        description="postgresql://… DSN, required when checkpointer=postgres.",
    )

    # ----- HTTP API / server -------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, gt=0, le=65535)
    api_workers: int = Field(default=1, gt=0)
    request_timeout_seconds: int = Field(
        default=120, gt=0, description="Hard ceiling for a single API request."
    )
    max_question_chars: int = Field(
        default=4000, gt=0, description="Reject questions longer than this at the edge."
    )
    # Comma-separated; exposed as lists via the api_keys / cors_origins properties.
    api_keys_raw: str | None = Field(default=None, validation_alias="API_KEYS")
    cors_origins_raw: str = Field(default="*", validation_alias="CORS_ORIGINS")

    # ----- Rate limiting -----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = Field(default=60, gt=0)

    # ----- Observability / tracing -------------------------------------------
    langsmith_tracing: bool = Field(
        default=False, description="Enable LangSmith tracing of LLM/tool calls."
    )
    langsmith_api_key: str | None = None
    langsmith_project: str = "agentic-research-agent"

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        value = value.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if value not in valid:
            raise ValueError(f"log_level must be one of {sorted(valid)}, got {value!r}")
        return value

    @model_validator(mode="after")
    def _validate_retrieval_settings(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @model_validator(mode="after")
    def _validate_checkpointer(self) -> Settings:
        if self.checkpointer is CheckpointerBackend.POSTGRES and not self.postgres_dsn:
            raise ValueError("postgres_dsn is required when checkpointer=postgres")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def effective_llm_model(self) -> str:
        """The model to use: an explicit ``llm_model`` or the provider default.

        Azure has no fixed model name — the deployment name does the routing —
        so it resolves to the configured deployment.
        """

        if self.llm_model:
            return self.llm_model
        if self.llm_provider is LLMProvider.AZURE:
            return self.azure_openai_deployment or ""
        return DEFAULT_MODELS.get(self.llm_provider, "")

    @property
    def api_keys(self) -> list[str]:
        """Configured API keys (comma-separated in ``API_KEYS``).

        Empty list means authentication is disabled — acceptable for local dev,
        but the API logs a warning so it is never silently open in production.
        """

        if not self.api_keys_raw:
            return []
        return [key.strip() for key in self.api_keys_raw.split(",") if key.strip()]

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` instance.

    Cached so settings are parsed and validated exactly once. Call
    ``get_settings.cache_clear()`` in tests when you need a fresh load.
    """

    return Settings()
