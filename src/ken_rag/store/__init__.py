"""Store layer for ken-rag.

Exports the primary public API so callers never import internals directly.
"""
from ken_rag.store.lancedb_store import LanceVectorStore
from ken_rag.store.metadata_store import LanceMetadataStore
from ken_rag.store.migrations import CURRENT_SCHEMA_VERSION, ensure_schema

__all__ = [
    "LanceVectorStore",
    "LanceMetadataStore",
    "CURRENT_SCHEMA_VERSION",
    "ensure_schema",
]
