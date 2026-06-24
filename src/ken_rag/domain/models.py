"""Frozen domain models for ken-rag.

All models use @dataclass(frozen=True, slots=True) — immutable by design.
No I/O or business logic here; this module is a pure data layer.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Optional

from ken_rag.domain.enums import FileChangeState, FileType


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single text chunk ready for embedding."""

    text: str
    file_path: str
    file_type: str
    chunk_index: int
    content_hash: str
    line_start: int
    line_end: int
    symbol_name: Optional[str]
    chunk_kind: str


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A Chunk paired with its embedding vector."""

    chunk: Chunk
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A Chunk returned from retrieval with its relevance score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Citation:
    """A source reference produced for an Answer."""

    file_path: str
    line_start: int
    line_end: int
    symbol_name: Optional[str]


@dataclass(frozen=True, slots=True)
class Answer:
    """The final answer produced by the generation pipeline."""

    text: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True, slots=True)
class FileRecord:
    """Metadata about an indexed file, derived from the chunks table."""

    file_path: str
    content_hash: str
    file_type: str
    chunk_count: int
    indexed_at: datetime.datetime
    git_commit: Optional[str]


@dataclass(frozen=True, slots=True)
class FileChange:
    """Represents the diff state of a single file during incremental re-index."""

    file_path: str
    state: FileChangeState


@dataclass(frozen=True, slots=True)
class CandidateFile:
    """A file discovered on disk, before indexing decisions are made."""

    path: str
    content_hash: str
    file_type: FileType


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """The raw text output of a Parser, before chunking."""

    text: str
    file_type: FileType
    line_count: int
