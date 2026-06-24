"""Structural Protocols for ken-rag.

Concrete implementations depend only on these protocols — never on each other.
All protocols are @runtime_checkable so tests can use isinstance checks.
The DI assembly point is cli/context.py:build_context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from ken_rag.domain.models import (
    Answer,
    CandidateFile,
    Chunk,
    EmbeddedChunk,
    FileChange,
    FileRecord,
    ParsedDocument,
    RetrievedChunk,
)


@runtime_checkable
class Parser(Protocol):
    """Converts a file on disk into a ParsedDocument."""

    def parse(self, path: Path) -> ParsedDocument:
        """Read and parse the file at *path*, returning its text and metadata."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a ParsedDocument into Chunks."""

    def chunk(
        self,
        doc: ParsedDocument,
        file_path: str,
        content_hash: str,
    ) -> list[Chunk]:
        """Return an ordered list of Chunks for *doc*."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Converts texts into dense embedding vectors."""

    @property
    def model_name(self) -> str:
        """The name of the embedding model (e.g. 'nomic-embed-text')."""
        ...

    @property
    def dim(self) -> int:
        """Expected vector dimension (e.g. 768)."""
        ...

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        """Embed a list of document texts (applies nomic search_document: prefix)."""
        ...

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed a single query string (applies nomic search_query: prefix)."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores and retrieves embedded chunks."""

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        """Insert or update embedded chunks in the store."""
        ...

    def search(self, vector: tuple[float, ...], k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks by vector similarity."""
        ...

    def search_text(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks by full-text search."""
        ...

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks for *file_path*. Returns the number deleted."""
        ...

    def list_files(self) -> list[FileRecord]:
        """Return a FileRecord for every indexed file."""
        ...

    def get_file(self, file_path: str) -> FileRecord | None:
        """Return the FileRecord for *file_path*, or None if not indexed."""
        ...

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        ...


@runtime_checkable
class MetadataStore(Protocol):
    """Key-value store for global index metadata (embedder name, schema version, …)."""

    def get(self, key: str) -> str | None:
        """Return the value for *key*, or None if absent."""
        ...

    def set(self, key: str, value: str) -> None:
        """Set *key* to *value*, creating or overwriting."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """High-level retrieval interface used by the query pipeline."""

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks relevant to *query*."""
        ...


@runtime_checkable
class Generator(Protocol):
    """Streams generated tokens from a language model."""

    @property
    def model_name(self) -> str:
        """The name of the generation model."""
        ...

    def stream(self, prompt: str, *, num_ctx: int) -> Iterator[str]:
        """Yield token strings for the given prompt."""
        ...


@runtime_checkable
class FileTracker(Protocol):
    """Computes the diff between on-disk candidates and the indexed state."""

    def diff(self, candidates: list[CandidateFile]) -> list[FileChange]:
        """Return FileChange objects for each candidate."""
        ...


@runtime_checkable
class IgnoreFilter(Protocol):
    """Decides whether a path should be excluded from indexing."""

    def is_ignored(self, path: Path) -> bool:
        """Return True if *path* should be skipped."""
        ...


@runtime_checkable
class GitClient(Protocol):
    """Thin wrapper around git shell commands."""

    def is_repo(self) -> bool:
        """Return True if the root is a git repository."""
        ...

    def head_sha(self) -> str | None:
        """Return the current HEAD commit SHA, or None."""
        ...

    def tracked_files(self) -> list[str]:
        """Return paths of all git-tracked files (relative to repo root)."""
        ...
