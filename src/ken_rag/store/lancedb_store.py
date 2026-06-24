"""LanceDB-backed VectorStore implementation for ken-rag.

``LanceVectorStore`` satisfies the ``VectorStore`` protocol and is the
primary on-disk store for embedded chunks.

Design decisions:
- PK is ``id = "{file_path}::{chunk_index}"`` — merge_insert on ``id``
  ensures idempotent upserts with no duplicate rows.
- ``delete_by_file`` uses LanceDB's SQL predicate delete for single-pass
  removal of all chunks belonging to a file.
- FTS index on ``chunk_text`` enables KeywordStage (hybrid retrieval).
- Scalar index on ``file_path`` speeds up per-file queries and deletes.
- ``list_files`` derives FileRecords by grouping the chunks table; there is
  no separate file-level table (no second source of truth).
- ``indexed_at`` is stored as a float64 Unix timestamp to keep the schema
  simple across platforms; it is re-hydrated to ``datetime.datetime`` on read.
"""
from __future__ import annotations

import datetime
import time
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import lancedb

from ken_rag.domain.models import Chunk, EmbeddedChunk, FileRecord, RetrievedChunk
from ken_rag.store.schema import chunks_schema
from ken_rag.store.metadata_store import LanceMetadataStore
from ken_rag.store.migrations import ensure_schema


_CHUNKS_TABLE = "chunks"


class LanceVectorStore:
    """Repository-pattern VectorStore backed by a local LanceDB database.

    Satisfies ``ken_rag.domain.protocols.VectorStore``.
    """

    def __init__(self, table: lancedb.table.LanceTable, dim: int) -> None:
        self._table = table
        self._dim = dim

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def open_or_create(cls, db_path: str, dim: int) -> "LanceVectorStore":
        """Open an existing chunks table or create a new one at *db_path*.

        Also stamps the schema version in ``ken_meta`` on first creation.
        """
        db = lancedb.connect(db_path)
        existing = db.list_tables().tables

        if _CHUNKS_TABLE in existing:
            table = db.open_table(_CHUNKS_TABLE)
        else:
            table = db.create_table(_CHUNKS_TABLE, schema=chunks_schema(dim))
            # Create scalar index on file_path for fast per-file filtering
            table.create_scalar_index("file_path")
            # Create FTS index on chunk_text for hybrid retrieval
            table.create_fts_index("chunk_text")

        # Stamp schema version (idempotent)
        meta = LanceMetadataStore.open_or_create(db_path)
        ensure_schema(meta)

        return cls(table, dim)

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        """Insert or update embedded chunks, merging on the ``id`` PK."""
        if not chunks:
            return

        now = time.time()
        rows: dict[str, list[Any]] = {
            "id": [],
            "vector": [],
            "chunk_text": [],
            "file_path": [],
            "file_type": [],
            "chunk_index": [],
            "content_hash": [],
            "line_start": [],
            "line_end": [],
            "symbol_name": [],
            "chunk_kind": [],
            "git_commit": [],
            "indexed_at": [],
        }

        for ec in chunks:
            c: Chunk = ec.chunk
            rows["id"].append(f"{c.file_path}::{c.chunk_index}")
            rows["vector"].append(list(ec.vector))
            rows["chunk_text"].append(c.text)
            rows["file_path"].append(c.file_path)
            rows["file_type"].append(c.file_type)
            rows["chunk_index"].append(c.chunk_index)
            rows["content_hash"].append(c.content_hash)
            rows["line_start"].append(c.line_start)
            rows["line_end"].append(c.line_end)
            rows["symbol_name"].append(c.symbol_name)
            rows["chunk_kind"].append(c.chunk_kind)
            # Chunk model has no git_commit field; store NULL
            rows["git_commit"].append(None)
            rows["indexed_at"].append(now)

        batch = pa.table(
            {
                "id": pa.array(rows["id"], type=pa.string()),
                "vector": pa.array(rows["vector"], type=pa.list_(pa.float32(), self._dim)),
                "chunk_text": pa.array(rows["chunk_text"], type=pa.string()),
                "file_path": pa.array(rows["file_path"], type=pa.string()),
                "file_type": pa.array(rows["file_type"], type=pa.string()),
                "chunk_index": pa.array(rows["chunk_index"], type=pa.int32()),
                "content_hash": pa.array(rows["content_hash"], type=pa.string()),
                "line_start": pa.array(rows["line_start"], type=pa.int32()),
                "line_end": pa.array(rows["line_end"], type=pa.int32()),
                "symbol_name": pa.array(rows["symbol_name"], type=pa.string()),
                "chunk_kind": pa.array(rows["chunk_kind"], type=pa.string()),
                "git_commit": pa.array(rows["git_commit"], type=pa.string()),
                "indexed_at": pa.array(rows["indexed_at"], type=pa.float64()),
            }
        )

        (
            self._table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(batch)
        )

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks for *file_path*. Returns the number deleted."""
        before = self._table.count_rows()
        safe_fp = file_path.replace("'", "''")
        self._table.delete(f"file_path = '{safe_fp}'")
        after = self._table.count_rows()
        return before - after

    def search(self, vector: tuple[float, ...], k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks by vector similarity (cosine distance)."""
        result = (
            self._table.search(list(vector))
            .limit(k)
            .to_arrow()
        )
        return [_row_to_retrieved(result, i, score_col="_distance", lower_is_better=True)
                for i in range(len(result))]

    def search_text(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks by full-text search."""
        result = (
            self._table.search(query, query_type="fts")
            .limit(k)
            .to_arrow()
        )
        return [_row_to_retrieved(result, i, score_col="_score", lower_is_better=False)
                for i in range(len(result))]

    def list_files(self) -> list[FileRecord]:
        """Return a FileRecord for every indexed file, grouped from chunks."""
        arrow = self._table.to_arrow()
        if len(arrow) == 0:
            return []
        return _arrow_to_file_records(arrow)

    def get_file(self, file_path: str) -> FileRecord | None:
        """Return the FileRecord for *file_path*, or None if not indexed."""
        safe_fp = file_path.replace("'", "''")
        result = (
            self._table.search()
            .where(f"file_path = '{safe_fp}'")
            .to_arrow()
        )
        if len(result) == 0:
            return None
        records = _arrow_to_file_records(result)
        return records[0] if records else None

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        return self._table.count_rows()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _row_to_retrieved(
    table: pa.Table,
    i: int,
    *,
    score_col: str,
    lower_is_better: bool,
) -> RetrievedChunk:
    """Convert row *i* of an Arrow result table to a RetrievedChunk."""
    raw_score: float = table[score_col][i].as_py()
    # Normalise: for distance metrics (lower=better) invert so higher=better.
    # Cosine distance of 0 = perfect match → score 1.0; distance 2 → 0.0.
    if lower_is_better:
        score = max(0.0, 1.0 - raw_score / 2.0)
    else:
        score = float(raw_score)

    chunk = Chunk(
        text=table["chunk_text"][i].as_py(),
        file_path=table["file_path"][i].as_py(),
        file_type=table["file_type"][i].as_py(),
        chunk_index=table["chunk_index"][i].as_py(),
        content_hash=table["content_hash"][i].as_py(),
        line_start=table["line_start"][i].as_py(),
        line_end=table["line_end"][i].as_py(),
        symbol_name=table["symbol_name"][i].as_py(),
        chunk_kind=table["chunk_kind"][i].as_py(),
    )
    return RetrievedChunk(chunk=chunk, score=score)


def _arrow_to_file_records(arrow: pa.Table) -> list[FileRecord]:
    """Group Arrow rows by file_path and return one FileRecord per file."""
    unique_fps: list[str] = pc.unique(arrow["file_path"]).to_pylist()
    records: list[FileRecord] = []

    for fp in unique_fps:
        mask = pc.equal(arrow["file_path"], fp)
        subset = arrow.filter(mask)
        # All rows for a file share the same content_hash, file_type, git_commit.
        # Take from the first row.
        first = subset.slice(0, 1)
        content_hash: str = first["content_hash"][0].as_py()
        file_type: str = first["file_type"][0].as_py()
        git_commit: str | None = first["git_commit"][0].as_py()

        chunk_count: int = len(subset)

        # Use the maximum indexed_at across all chunks for this file
        max_ts: float = pc.max(subset["indexed_at"]).as_py()
        indexed_at = datetime.datetime.fromtimestamp(max_ts, tz=datetime.timezone.utc)

        records.append(
            FileRecord(
                file_path=fp,
                content_hash=content_hash,
                file_type=file_type,
                chunk_count=chunk_count,
                indexed_at=indexed_at,
                git_commit=git_commit,
            )
        )

    return records
