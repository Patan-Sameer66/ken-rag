"""Integration tests for the hand-written ingest + query pipelines.

All collaborators are fakes — no Ollama daemon, no LanceDB on disk.
Verifies the Phase-1 contract:
  - first ingest records the embedder name in metadata
  - a mismatched embedder name aborts ingest AND query with EmbedderMismatchError
  - re-ingesting an unchanged file does no new work (count stable, no orphans)
  - QueryPipeline.ask returns an Answer with text + citations
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ken_rag.chunking.registry import ChunkerRegistry
from ken_rag.config.settings import Settings
from ken_rag.domain.errors import EmbedderMismatchError
from ken_rag.domain.models import Answer
from ken_rag.parsing.registry import ParserRegistry
from ken_rag.pipeline.ingest import IngestPipeline
from ken_rag.pipeline.query import QueryPipeline
from ken_rag.retrieval.pipeline import RetrievalPipeline
from ken_rag.retrieval.stages import DenseStage, KeywordStage
from tests.fakes.fake_embedder import FakeEmbedder
from tests.fakes.fake_generator import FakeGenerator
from tests.fakes.fake_meta import FakeMetadataStore
from tests.fakes.in_memory_store import InMemoryVectorStore

_DOC = """# Authentication

The require_auth decorator wraps each route and checks the bearer token
before the handler runs. Expired tokens are rejected with a 401 response.

# Tokens

verify_token decodes the JWT, validates its signature, and confirms the
expiry claim has not passed. Invalid tokens raise an AuthError.
"""


def _write_doc(tmp_path: Path, name: str = "auth.md", body: str = _DOC) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _build_ingest(embedder, store, meta) -> IngestPipeline:
    return IngestPipeline(
        parser_registry=ParserRegistry.default(),
        chunker_registry=ChunkerRegistry.default(),
        embedder=embedder,
        vector_store=store,
        metadata_store=meta,
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def test_first_ingest_indexes_file_and_records_embedder(tmp_path):
    _write_doc(tmp_path)
    embedder = FakeEmbedder(model_name="nomic-embed-text")
    store = InMemoryVectorStore()
    meta = FakeMetadataStore()
    ingest = _build_ingest(embedder, store, meta)

    count = ingest.run(tmp_path)

    assert count == 1
    assert store.count() > 0
    assert meta.get("embedder_name") == "nomic-embed-text"


def test_ingest_mismatched_embedder_raises(tmp_path):
    _write_doc(tmp_path)
    store = InMemoryVectorStore()
    meta = FakeMetadataStore()
    meta.set("embedder_name", "some-other-model")
    ingest = _build_ingest(FakeEmbedder(model_name="nomic-embed-text"), store, meta)

    with pytest.raises(EmbedderMismatchError) as exc:
        ingest.run(tmp_path)
    assert exc.value.hint  # fix-it hint present


def test_reingest_unchanged_file_does_no_work(tmp_path):
    _write_doc(tmp_path)
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    meta = FakeMetadataStore()
    ingest = _build_ingest(embedder, store, meta)

    first = ingest.run(tmp_path)
    count_after_first = store.count()
    second = ingest.run(tmp_path)

    assert first == 1
    assert second == 0  # unchanged → skipped
    assert store.count() == count_after_first  # no orphans, no duplicates


def test_reingest_changed_file_replaces_chunks(tmp_path):
    doc = _write_doc(tmp_path)
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    meta = FakeMetadataStore()
    ingest = _build_ingest(embedder, store, meta)

    ingest.run(tmp_path)
    # Change the file → different hash → re-index.
    doc.write_text("# New\n\nCompletely different content here.\n", encoding="utf-8")
    second = ingest.run(tmp_path)

    assert second == 1
    # Only one file's chunks should exist (old ones deleted, not orphaned).
    assert len(store.list_files()) == 1


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def _build_query(embedder, store, meta, tokens) -> QueryPipeline:
    retrieval = RetrievalPipeline(
        stages=[DenseStage(embedder=embedder, store=store), KeywordStage(store=store)]
    )
    generator = FakeGenerator(tokens=tokens, expected_num_ctx=8192)
    settings = Settings.default(Path("."))
    return QueryPipeline(
        embedder=embedder,
        retrieval_pipeline=retrieval,
        generator=generator,
        metadata_store=meta,
        settings=settings,
    )


def test_ask_returns_answer_with_citations(tmp_path):
    _write_doc(tmp_path)
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    meta = FakeMetadataStore()
    _build_ingest(embedder, store, meta).run(tmp_path)

    query = _build_query(embedder, store, meta, tokens=["Auth", " uses", " tokens."])
    answer = query.ask("how does authentication work?")

    assert isinstance(answer, Answer)
    assert answer.text == "Auth uses tokens."
    assert len(answer.citations) >= 1


def test_ask_guards_embedder_mismatch(tmp_path):
    _write_doc(tmp_path)
    embedder = FakeEmbedder(model_name="nomic-embed-text")
    store = InMemoryVectorStore()
    meta = FakeMetadataStore()
    _build_ingest(embedder, store, meta).run(tmp_path)
    # Now query with a different embedder than the index was built with.
    other = FakeEmbedder(model_name="different-model")
    query = _build_query(other, store, meta, tokens=["x"])

    with pytest.raises(EmbedderMismatchError):
        query.ask("anything")
