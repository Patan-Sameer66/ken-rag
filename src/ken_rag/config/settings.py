"""Frozen Settings dataclass for ken-rag.

Settings is assembled by the loader and passed read-only through the DI graph.
Use Settings.default(root) to get a baseline with all defaults filled in.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ken_rag.config.defaults import (
    BATCH_SIZE,
    DEFAULT_K,
    DEFAULT_LLM,
    EMBED_DIM,
    EMBEDDER_NAME,
    NUM_CTX,
    OLLAMA_URL,
    TIMEOUT_S,
)
from ken_rag.config.paths import db_path


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings. Constructed once at startup."""

    # Retrieval
    k: int

    # Generation
    num_ctx: int

    # Embedding
    embed_dim: int
    embedder_name: str

    # LLM
    llm_name: str

    # Batching
    batch_size: int

    # Store
    db_path: Path

    # Ollama
    ollama_url: str
    timeout_s: int

    @classmethod
    def default(cls, root: Path) -> "Settings":
        """Construct Settings filled entirely with compile-time defaults."""
        return cls(
            k=DEFAULT_K,
            num_ctx=NUM_CTX,
            embed_dim=EMBED_DIM,
            embedder_name=EMBEDDER_NAME,
            llm_name=DEFAULT_LLM,
            batch_size=BATCH_SIZE,
            db_path=db_path(root),
            ollama_url=OLLAMA_URL,
            timeout_s=TIMEOUT_S,
        )
