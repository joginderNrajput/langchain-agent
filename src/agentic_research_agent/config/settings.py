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
    OLLAMA = "ollama"


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
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model name for the selected provider.",
    )
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0)
    llm_timeout_seconds: int = Field(default=60, gt=0)

    # ----- Provider credentials ---------------------------------------------
    groq_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
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
    max_search_results: int = Field(default=5, gt=0)
    recursion_limit: int = Field(
        default=25,
        gt=0,
        description="Max LangGraph super-steps before aborting a run.",
    )

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

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` instance.

    Cached so settings are parsed and validated exactly once. Call
    ``get_settings.cache_clear()`` in tests when you need a fresh load.
    """

    return Settings()
