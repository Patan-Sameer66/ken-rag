"""PyArrow schema definitions for ken-rag LanceDB tables.

Table ``chunks``:
    id            string       PK = "{file_path}::{chunk_index}"
    vector        fixed_size_list<float32>[dim]
    chunk_text    string       FTS-indexed
    file_path     string       scalar-indexed for fast per-file queries
    file_type     string
    chunk_index   int32
    content_hash  string
    line_start    int32
    line_end      int32
    symbol_name   string       nullable
    chunk_kind    string
    git_commit    string       nullable
    indexed_at    double       Unix timestamp (seconds since epoch)

Table ``ken_meta``:
    key   string   PK
    value string
"""
from __future__ import annotations

import pyarrow as pa


def chunks_schema(dim: int) -> pa.Schema:
    """Return the PyArrow schema for the ``chunks`` table with the given embedding *dim*."""
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dim), nullable=False),
            pa.field("chunk_text", pa.string(), nullable=False),
            pa.field("file_path", pa.string(), nullable=False),
            pa.field("file_type", pa.string(), nullable=False),
            pa.field("chunk_index", pa.int32(), nullable=False),
            pa.field("content_hash", pa.string(), nullable=False),
            pa.field("line_start", pa.int32(), nullable=False),
            pa.field("line_end", pa.int32(), nullable=False),
            pa.field("symbol_name", pa.string(), nullable=True),
            pa.field("chunk_kind", pa.string(), nullable=False),
            pa.field("git_commit", pa.string(), nullable=True),
            pa.field("indexed_at", pa.float64(), nullable=False),
        ]
    )


def ken_meta_schema() -> pa.Schema:
    """Return the PyArrow schema for the ``ken_meta`` key-value table."""
    return pa.schema(
        [
            pa.field("key", pa.string(), nullable=False),
            pa.field("value", pa.string(), nullable=False),
        ]
    )
