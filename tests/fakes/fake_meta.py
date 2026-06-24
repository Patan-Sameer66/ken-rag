"""FakeMetadataStore — dict-backed MetadataStore for tests.

Implements ``ken_rag.domain.protocols.MetadataStore`` without LanceDB.
"""
from __future__ import annotations


class FakeMetadataStore:
    """In-memory key-value metadata store."""

    def __init__(self) -> None:
        self._kv: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._kv.get(key)

    def set(self, key: str, value: str) -> None:
        self._kv[key] = value
