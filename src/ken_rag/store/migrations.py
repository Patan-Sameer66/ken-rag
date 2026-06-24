"""Schema migration support for the ken-rag LanceDB store.

``ensure_schema`` stamps the current schema version into ``ken_meta`` on first
open and is idempotent for subsequent opens on an up-to-date index.

Future migration steps should be added as versioned functions called in order
by ``ensure_schema``.
"""
from __future__ import annotations

from ken_rag.store.metadata_store import LanceMetadataStore


#: The schema version this codebase understands.  Bump when making breaking
#: changes to the ``chunks`` table layout.
CURRENT_SCHEMA_VERSION: int = 1

_KEY = "schema_version"


def ensure_schema(meta: LanceMetadataStore) -> None:
    """Stamp *meta* with the current schema version if not already present.

    Idempotent — safe to call every time a store is opened.  Future versions
    may add migration logic here before updating the stamp.
    """
    existing = meta.get(_KEY)
    if existing is None:
        meta.set(_KEY, str(CURRENT_SCHEMA_VERSION))
        return

    stored_version = int(existing)
    if stored_version < CURRENT_SCHEMA_VERSION:
        # Future: apply incremental migrations here
        meta.set(_KEY, str(CURRENT_SCHEMA_VERSION))
