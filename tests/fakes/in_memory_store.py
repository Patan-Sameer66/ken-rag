"""InMemoryVectorStore — numpy-cosine VectorStore fake for tests.

Implements the full ``VectorStore`` protocol (``ken_rag.domain.protocols``)
without any disk I/O or LanceDB dependency.  Used as:
  1. A fast test double in unit tests.
  2. One leg of the shared VectorStore contract tests in
     ``tests/integration/test_lancedb_store.py``.

Retrieval semantics:
  - ``search``: cosine similarity (higher = better).
  - ``search_text``: naive case-insensitive substring containment, ordered by
    the number of query tokens found in the chunk text (a minimal but correct
    approximation of FTS for test purposes).
"""
from __future__ import annotations

import datetime
import math
import time
from typing import Any

from ken_rag.domain.models import Chunk, EmbeddedChunk, FileRecord, RetrievedChunk


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Return the cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    """In-memory VectorStore backed by plain Python dicts.

    Satisfies ``ken_rag.domain.protocols.VectorStore``.

    Parameters
    ----------
    dim:
        Expected vector dimension.  Not enforced at runtime but kept for
        parity with ``LanceVectorStore.open_or_create(db_path, dim)``.
    """

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim
        # Keyed by chunk id ("file_path::chunk_index")
        self._store: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        """Insert or update embedded chunks (merge on id PK)."""
        now = time.time()
        for ec in chunks:
            c: Chunk = ec.chunk
            chunk_id = f"{c.file_path}::{c.chunk_index}"
            self._store[chunk_id] = {
                "id": chunk_id,
                "vector": ec.vector,
                "chunk": c,
                "indexed_at": now,
            }

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks for *file_path*. Returns the number deleted."""
        to_delete = [
            k for k, v in self._store.items()
            if v["chunk"].file_path == file_path
        ]
        for k in to_delete:
            del self._store[k]
        return len(to_delete)

    def search(self, vector: tuple[float, ...], k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks by cosine similarity (higher = better)."""
        scored = [
            (chunk_id, _cosine_similarity(vector, entry["vector"]), entry["chunk"])
            for chunk_id, entry in self._store.items()
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [
            RetrievedChunk(chunk=chunk, score=score)
            for _, score, chunk in scored[:k]
        ]

    def search_text(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks by substring match (naive FTS approximation).

        Scoring: count of lowercased query tokens found in the lowercased
        chunk text.  Only chunks containing at least one token are returned.
        Ties are broken by insertion order (stable in Python 3.7+).
        """
        tokens = query.lower().split()
        scored: list[tuple[int, Chunk]] = []
        for entry in self._store.values():
            text_lower = entry["chunk"].text.lower()
            hit_count = sum(1 for t in tokens if t in text_lower)
            if hit_count > 0:
                scored.append((hit_count, entry["chunk"]))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            RetrievedChunk(chunk=chunk, score=float(count))
            for count, chunk in scored[:k]
        ]

    def list_files(self) -> list[FileRecord]:
        """Return a FileRecord for every indexed file."""
        # Aggregate per file_path
        per_file: dict[str, list[dict[str, Any]]] = {}
        for entry in self._store.values():
            fp = entry["chunk"].file_path
            per_file.setdefault(fp, []).append(entry)

        records: list[FileRecord] = []
        for fp, entries in per_file.items():
            first_chunk: Chunk = entries[0]["chunk"]
            chunk_count = len(entries)
            max_ts = max(e["indexed_at"] for e in entries)
            indexed_at = datetime.datetime.fromtimestamp(max_ts, tz=datetime.timezone.utc)
            records.append(
                FileRecord(
                    file_path=fp,
                    content_hash=first_chunk.content_hash,
                    file_type=first_chunk.file_type,
                    chunk_count=chunk_count,
                    indexed_at=indexed_at,
                    git_commit=None,
                )
            )
        return records

    def get_file(self, file_path: str) -> FileRecord | None:
        """Return the FileRecord for *file_path*, or None if not indexed."""
        matching = [
            entry for entry in self._store.values()
            if entry["chunk"].file_path == file_path
        ]
        if not matching:
            return None
        first_chunk: Chunk = matching[0]["chunk"]
        max_ts = max(e["indexed_at"] for e in matching)
        indexed_at = datetime.datetime.fromtimestamp(max_ts, tz=datetime.timezone.utc)
        return FileRecord(
            file_path=file_path,
            content_hash=first_chunk.content_hash,
            file_type=first_chunk.file_type,
            chunk_count=len(matching),
            indexed_at=indexed_at,
            git_commit=None,
        )

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        return len(self._store)
