"""LanceDB-backed key-value MetadataStore for ken-rag.

Stores global index metadata (embedder name, schema version, …) in a
``ken_meta`` table.  Key uniqueness is enforced via merge_insert upsert.
"""
from __future__ import annotations

import pyarrow as pa
import lancedb

from ken_rag.store.schema import ken_meta_schema


_TABLE_NAME = "ken_meta"


class LanceMetadataStore:
    """Key-value store backed by a LanceDB ``ken_meta`` table.

    Satisfies the ``MetadataStore`` protocol from ``ken_rag.domain.protocols``.
    """

    def __init__(self, table: lancedb.table.LanceTable) -> None:
        self._table = table

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def open_or_create(cls, db_path: str) -> "LanceMetadataStore":
        """Open an existing ``ken_meta`` table or create a new one at *db_path*."""
        db = lancedb.connect(db_path)
        existing = db.list_tables().tables
        if _TABLE_NAME in existing:
            table = db.open_table(_TABLE_NAME)
        else:
            table = db.create_table(_TABLE_NAME, schema=ken_meta_schema())
        return cls(table)

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def get(self, key: str) -> str | None:
        """Return the value for *key*, or None if absent."""
        # Escape single quotes in key for safety
        safe_key = key.replace("'", "''")
        result = self._table.search().where(f"key = '{safe_key}'").to_arrow()
        if len(result) == 0:
            return None
        return result["value"][0].as_py()

    def set(self, key: str, value: str) -> None:
        """Set *key* to *value*, creating or overwriting the existing row."""
        row = pa.table({"key": [key], "value": [value]})
        (
            self._table.merge_insert("key")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(row)
        )
