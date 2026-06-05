"""Internal knowledge base — Retrieval-Augmented Generation (RAG).

Documents under ``settings.knowledge_base_dir`` are chunked, embedded with a
local sentence-transformers model (no API key, runs offline), and stored in a
persistent Chroma vector store. The agent queries this store through a tool to
ground its answers in the organisation's own documents.

Embeddings are computed locally, so the first build downloads the embedding
model (~80 MB) once and caches it; subsequent runs reuse the persisted store.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_research_agent.config.settings import Settings
from agentic_research_agent.core.exceptions import KnowledgeBaseError
from agentic_research_agent.core.logging import get_logger

logger = get_logger(__name__)

# Plain-text document extensions we know how to load without extra parsers.
_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}
_COLLECTION_NAME = "knowledge_base"


class KnowledgeBase:
    """Builds and serves a persistent vector store over local documents."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._store: Chroma | None = None

    # -- lifecycle ------------------------------------------------------------

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazily construct the embedding model (defers the model download)."""

        if self._embeddings is None:
            logger.debug("Loading embedding model: %s", self._settings.embedding_model)
            self._configure_hf_auth()
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self._settings.embedding_model
            )
        return self._embeddings

    def _configure_hf_auth(self) -> None:
        """Authenticate with the HF Hub if a token is configured.

        When a token is present it is exported so model downloads are
        authenticated (higher rate limits). When absent, we quiet the
        "unauthenticated requests" notice — the public model downloads fine
        without a token, so the warning is noise for our use case.
        """

        if self._settings.hf_token:
            os.environ.setdefault("HF_TOKEN", self._settings.hf_token)
        else:
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    def build(self, *, force: bool = False) -> None:
        """Build the vector store, reusing a persisted one when present.

        Args:
            force: Rebuild from scratch even if a persisted store exists. Use
                after editing the documents under the knowledge base directory.
        """

        persist_dir = self._settings.vector_store_dir
        already_built = (persist_dir / "chroma.sqlite3").exists()

        if already_built and not force:
            logger.info("Loading existing vector store from %s", persist_dir)
            self._store = Chroma(
                collection_name=_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(persist_dir),
            )
            return

        documents = self._load_documents()
        if not documents:
            raise KnowledgeBaseError(
                f"No documents found in {self._settings.knowledge_base_dir}. "
                f"Add .md/.txt files and re-run ingestion."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        logger.info("Indexing %d chunks from %d documents", len(chunks), len(documents))

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=_COLLECTION_NAME,
            persist_directory=str(persist_dir),
        )

    # -- querying -------------------------------------------------------------

    def _ensure_built(self) -> Chroma:
        if self._store is None:
            self.build()
        assert self._store is not None  # for type-checkers
        return self._store

    def search(self, query: str) -> list[Document]:
        """Return the top-k most relevant chunks for ``query``."""

        store = self._ensure_built()
        return store.similarity_search(query, k=self._settings.retriever_top_k)

    def as_tool(self) -> BaseTool:
        """Expose the knowledge base as a LangChain tool for the agent."""

        @tool
        def knowledge_base_search(query: str) -> str:
            """Search the internal knowledge base of curated documents.

            Prefer this over web search for organisation-specific or
            foundational/reference material. Returns the most relevant excerpts
            with their source filenames so answers can be cited.
            """

            logger.info("knowledge_base_search query=%r", query)
            docs = self.search(query)
            if not docs:
                return f"No relevant documents found for {query!r}."
            blocks = []
            for i, doc in enumerate(docs, start=1):
                source = Path(doc.metadata.get("source", "unknown")).name
                blocks.append(f"[{i}] (source: {source})\n{doc.page_content.strip()}")
            return "\n\n".join(blocks)

        return knowledge_base_search

    # -- internal -------------------------------------------------------------

    def _load_documents(self) -> list[Document]:
        """Load supported documents from the knowledge base directory."""

        kb_dir = self._settings.knowledge_base_dir
        if not kb_dir.exists():
            raise KnowledgeBaseError(f"Knowledge base directory not found: {kb_dir}")

        documents: list[Document] = []
        for path in sorted(kb_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in _TEXT_SUFFIXES:
                documents.extend(self._load_text(path))
            elif path.suffix.lower() == ".pdf":
                documents.extend(self._load_pdf(path))
            else:
                logger.debug("Skipping unsupported file: %s", path.name)
        return documents

    @staticmethod
    def _load_text(path: Path) -> list[Document]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": str(path)})]

    @staticmethod
    def _load_pdf(path: Path) -> list[Document]:
        # Imported lazily so PDF support is optional.
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return [Document(page_content=text, metadata={"source": str(path)})]
