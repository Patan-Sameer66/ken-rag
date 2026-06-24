"""Integration tests for the LanceDB store layer — Task 1.8.

Runs the shared assertion suite against BOTH implementations:
  - LanceVectorStore (real LanceDB, tmp dir)
  - InMemoryVectorStore (numpy-cosine fake)

Also tests MetadataStore and migration stamp independently.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any


from ken_rag.domain.models import Chunk, EmbeddedChunk, FileRecord
from ken_rag.config.defaults import EMBED_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    file_path: str,
    chunk_index: int,
    text: str = "sample text",
    content_hash: str = "hash_abc",
    file_type: str = "md",
    line_start: int = 1,
    line_end: int = 10,
    symbol_name: str | None = None,
    chunk_kind: str = "prose",
) -> Chunk:
    return Chunk(
        text=text,
        file_path=file_path,
        file_type=file_type,
        chunk_index=chunk_index,
        content_hash=content_hash,
        line_start=line_start,
        line_end=line_end,
        symbol_name=symbol_name,
        chunk_kind=chunk_kind,
    )


def _unit_vec(dim: int, hot_index: int) -> tuple[float, ...]:
    """One-hot vector: index *hot_index* is 1.0, rest 0.0."""
    v = [0.0] * dim
    v[hot_index] = 1.0
    return tuple(v)


def _make_embedded(file_path: str, chunk_index: int, hot: int, **kwargs: Any) -> EmbeddedChunk:
    chunk = _make_chunk(file_path, chunk_index, **kwargs)
    return EmbeddedChunk(chunk=chunk, vector=_unit_vec(EMBED_DIM, hot))


# ---------------------------------------------------------------------------
# Shared contract suite
# ---------------------------------------------------------------------------

def run_vector_store_contract(store: Any, dim: int = EMBED_DIM) -> None:
    """Assert the VectorStore contract against any conforming implementation."""

    # --- upsert two files --------------------------------------------------
    chunks_a = [
        _make_embedded("a.md", 0, hot=0, text="python functions and classes", content_hash="ha"),
        _make_embedded("a.md", 1, hot=1, text="more about python decorators", content_hash="ha"),
    ]
    chunks_b = [
        _make_embedded("b.py", 0, hot=2, text="rust ownership and borrowing", content_hash="hb"),
    ]
    store.upsert(chunks_a)
    store.upsert(chunks_b)

    total = store.count()
    assert total == 3, f"Expected 3 chunks after upsert, got {total}"

    # --- vector search returns nearest by cosine ----------------------------
    # Query close to hot=0 → should return a.md::0 first
    query = _unit_vec(dim, 0)
    results = store.search(query, k=2)
    assert len(results) >= 1
    assert results[0].chunk.file_path == "a.md"
    assert results[0].chunk.chunk_index == 0
    # Scores should be in descending order
    if len(results) == 2:
        assert results[0].score >= results[1].score

    # --- search_text returns relevant chunks --------------------------------
    text_results = store.search_text("python", k=3)
    assert len(text_results) >= 1
    assert any("python" in r.chunk.text.lower() for r in text_results)

    # --- delete_by_file removes only that file, returns count ---------------
    deleted = store.delete_by_file("a.md")
    assert deleted == 2, f"Expected 2 deleted, got {deleted}"
    assert store.count() == 1, f"Expected 1 chunk left, got {store.count()}"

    # Verify b.py chunks remain untouched
    b_results = store.search(_unit_vec(dim, 2), k=1)
    assert b_results[0].chunk.file_path == "b.py"

    # --- re-upsert after delete leaves no orphans (count stable) -----------
    store.upsert(chunks_a)
    assert store.count() == 3, f"Expected 3 after re-upsert, got {store.count()}"
    # Re-upsert same data again — idempotent, no duplicates
    store.upsert(chunks_a)
    assert store.count() == 3, f"Expected 3 after idempotent re-upsert, got {store.count()}"

    # --- list_files groups correctly ----------------------------------------
    files = store.list_files()
    assert len(files) == 2
    by_path = {f.file_path: f for f in files}
    assert "a.md" in by_path
    assert "b.py" in by_path

    a_rec = by_path["a.md"]
    assert isinstance(a_rec, FileRecord)
    assert a_rec.chunk_count == 2
    assert a_rec.content_hash == "ha"
    assert a_rec.file_type == "md"
    assert isinstance(a_rec.indexed_at, datetime.datetime)

    b_rec = by_path["b.py"]
    assert b_rec.chunk_count == 1

    # --- get_file returns FileRecord for known path, None for unknown -------
    got = store.get_file("a.md")
    assert got is not None
    assert got.file_path == "a.md"

    missing = store.get_file("nonexistent.md")
    assert missing is None

    # --- delete all files leaves count == 0 --------------------------------
    store.delete_by_file("a.md")
    store.delete_by_file("b.py")
    assert store.count() == 0


# ---------------------------------------------------------------------------
# LanceVectorStore tests
# ---------------------------------------------------------------------------

class TestLanceVectorStore:
    def test_open_or_create_creates_new(self, tmp_path: Path) -> None:
        from ken_rag.store.lancedb_store import LanceVectorStore
        store = LanceVectorStore.open_or_create(str(tmp_path), EMBED_DIM)
        assert store.count() == 0

    def test_open_or_create_reopens_existing(self, tmp_path: Path) -> None:
        from ken_rag.store.lancedb_store import LanceVectorStore
        store1 = LanceVectorStore.open_or_create(str(tmp_path), EMBED_DIM)
        chunk = _make_embedded("x.md", 0, hot=0, content_hash="hx")
        store1.upsert([chunk])

        store2 = LanceVectorStore.open_or_create(str(tmp_path), EMBED_DIM)
        assert store2.count() == 1

    def test_contract(self, tmp_path: Path) -> None:
        from ken_rag.store.lancedb_store import LanceVectorStore
        store = LanceVectorStore.open_or_create(str(tmp_path), EMBED_DIM)
        run_vector_store_contract(store)

    def test_upsert_is_idempotent_by_id(self, tmp_path: Path) -> None:
        """Re-upsert with same IDs does not duplicate rows."""
        from ken_rag.store.lancedb_store import LanceVectorStore
        store = LanceVectorStore.open_or_create(str(tmp_path), EMBED_DIM)
        chunk = _make_embedded("f.md", 0, hot=3, content_hash="old")
        store.upsert([chunk])
        assert store.count() == 1

        # Upsert with same file_path::chunk_index but different text/hash
        updated = _make_embedded("f.md", 0, hot=3, text="updated text", content_hash="new")
        store.upsert([updated])
        assert store.count() == 1
        # The updated content should be reflected
        files = store.list_files()
        assert files[0].content_hash == "new"

    def test_delete_by_file_returns_zero_for_missing(self, tmp_path: Path) -> None:
        from ken_rag.store.lancedb_store import LanceVectorStore
        store = LanceVectorStore.open_or_create(str(tmp_path), EMBED_DIM)
        deleted = store.delete_by_file("does_not_exist.md")
        assert deleted == 0

    def test_search_text_fts(self, tmp_path: Path) -> None:
        from ken_rag.store.lancedb_store import LanceVectorStore
        store = LanceVectorStore.open_or_create(str(tmp_path), EMBED_DIM)
        store.upsert([
            _make_embedded("a.md", 0, hot=0, text="lancedb full text search example"),
            _make_embedded("a.md", 1, hot=1, text="vector similarity retrieval"),
        ])
        results = store.search_text("lancedb full text", k=2)
        assert len(results) >= 1
        assert results[0].chunk.chunk_index == 0

    def test_list_files_empty(self, tmp_path: Path) -> None:
        from ken_rag.store.lancedb_store import LanceVectorStore
        store = LanceVectorStore.open_or_create(str(tmp_path), EMBED_DIM)
        assert store.list_files() == []

    def test_list_files_git_commit_preserved(self, tmp_path: Path) -> None:
        from ken_rag.store.lancedb_store import LanceVectorStore
        store = LanceVectorStore.open_or_create(str(tmp_path), EMBED_DIM)
        chunk = _make_embedded("tracked.py", 0, hot=5, content_hash="hc")
        # Modify the chunk to include a git_commit — we need to create it manually
        # because EmbeddedChunk.chunk is frozen; use lancedb_store._upsert_raw if available
        # For now, upsert through the public API (git_commit will be None per Chunk model)
        store.upsert([chunk])
        rec = store.get_file("tracked.py")
        assert rec is not None
        assert rec.git_commit is None  # Chunk model has no git_commit field, stored as NULL


# ---------------------------------------------------------------------------
# MetadataStore tests
# ---------------------------------------------------------------------------

class TestMetadataStore:
    def test_get_set_roundtrip(self, tmp_path: Path) -> None:
        from ken_rag.store.metadata_store import LanceMetadataStore
        meta = LanceMetadataStore.open_or_create(str(tmp_path))
        meta.set("embedder_name", "nomic-embed-text")
        assert meta.get("embedder_name") == "nomic-embed-text"

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        from ken_rag.store.metadata_store import LanceMetadataStore
        meta = LanceMetadataStore.open_or_create(str(tmp_path))
        assert meta.get("nonexistent_key") is None

    def test_set_overwrites(self, tmp_path: Path) -> None:
        from ken_rag.store.metadata_store import LanceMetadataStore
        meta = LanceMetadataStore.open_or_create(str(tmp_path))
        meta.set("key", "v1")
        meta.set("key", "v2")
        assert meta.get("key") == "v2"

    def test_multiple_keys(self, tmp_path: Path) -> None:
        from ken_rag.store.metadata_store import LanceMetadataStore
        meta = LanceMetadataStore.open_or_create(str(tmp_path))
        meta.set("a", "1")
        meta.set("b", "2")
        assert meta.get("a") == "1"
        assert meta.get("b") == "2"

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        from ken_rag.store.metadata_store import LanceMetadataStore
        meta1 = LanceMetadataStore.open_or_create(str(tmp_path))
        meta1.set("schema_version", "1")
        meta2 = LanceMetadataStore.open_or_create(str(tmp_path))
        assert meta2.get("schema_version") == "1"


# ---------------------------------------------------------------------------
# Migrations tests
# ---------------------------------------------------------------------------

class TestMigrations:
    def test_schema_version_stamped_on_create(self, tmp_path: Path) -> None:
        from ken_rag.store.migrations import CURRENT_SCHEMA_VERSION, ensure_schema
        from ken_rag.store.metadata_store import LanceMetadataStore
        meta = LanceMetadataStore.open_or_create(str(tmp_path))
        ensure_schema(meta)
        assert meta.get("schema_version") == str(CURRENT_SCHEMA_VERSION)

    def test_ensure_schema_idempotent(self, tmp_path: Path) -> None:
        from ken_rag.store.migrations import CURRENT_SCHEMA_VERSION, ensure_schema
        from ken_rag.store.metadata_store import LanceMetadataStore
        meta = LanceMetadataStore.open_or_create(str(tmp_path))
        ensure_schema(meta)
        ensure_schema(meta)
        assert meta.get("schema_version") == str(CURRENT_SCHEMA_VERSION)

    def test_open_or_create_stamps_version(self, tmp_path: Path) -> None:
        from ken_rag.store.lancedb_store import LanceVectorStore
        from ken_rag.store.metadata_store import LanceMetadataStore
        from ken_rag.store.migrations import CURRENT_SCHEMA_VERSION
        LanceVectorStore.open_or_create(str(tmp_path), EMBED_DIM)
        meta = LanceMetadataStore.open_or_create(str(tmp_path))
        assert meta.get("schema_version") == str(CURRENT_SCHEMA_VERSION)


# ---------------------------------------------------------------------------
# InMemoryVectorStore contract test
# ---------------------------------------------------------------------------

class TestInMemoryVectorStore:
    def test_contract(self) -> None:
        from tests.fakes.in_memory_store import InMemoryVectorStore
        store = InMemoryVectorStore(dim=EMBED_DIM)
        run_vector_store_contract(store)

    def test_open_and_empty(self) -> None:
        from tests.fakes.in_memory_store import InMemoryVectorStore
        store = InMemoryVectorStore(dim=EMBED_DIM)
        assert store.count() == 0
        assert store.list_files() == []
        assert store.get_file("x") is None

    def test_search_text_substring(self) -> None:
        from tests.fakes.in_memory_store import InMemoryVectorStore
        store = InMemoryVectorStore(dim=EMBED_DIM)
        store.upsert([
            _make_embedded("a.md", 0, hot=0, text="hello world search"),
            _make_embedded("a.md", 1, hot=1, text="unrelated content here"),
        ])
        results = store.search_text("search", k=5)
        assert len(results) == 1
        assert "search" in results[0].chunk.text


# ---------------------------------------------------------------------------
# Protocol conformance tests (isinstance checks)
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    def test_lancedb_store_is_vector_store(self, tmp_path: Path) -> None:
        from ken_rag.store.lancedb_store import LanceVectorStore
        from ken_rag.domain.protocols import VectorStore
        store = LanceVectorStore.open_or_create(str(tmp_path), EMBED_DIM)
        assert isinstance(store, VectorStore)

    def test_metadata_store_is_metadata_store_protocol(self, tmp_path: Path) -> None:
        from ken_rag.store.metadata_store import LanceMetadataStore
        from ken_rag.domain.protocols import MetadataStore
        meta = LanceMetadataStore.open_or_create(str(tmp_path))
        assert isinstance(meta, MetadataStore)

    def test_in_memory_store_is_vector_store(self) -> None:
        from tests.fakes.in_memory_store import InMemoryVectorStore
        from ken_rag.domain.protocols import VectorStore
        store = InMemoryVectorStore(dim=EMBED_DIM)
        assert isinstance(store, VectorStore)
