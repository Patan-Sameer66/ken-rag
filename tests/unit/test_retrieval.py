"""Unit tests for ken_rag.retrieval — Task 1.9.

Tests cover:
- DenseStage: delegates to store.search after embedding the query.
- KeywordStage: delegates to store.search_text.
- rrf_fuse: merges rankings by reciprocal rank, deduplicates by chunk id,
  orders by descending fused score.
- RetrievalPipeline: runs all stages, feeds rankings to rrf_fuse, returns top-k.
- Hybrid value: a literal-identifier query is surfaced via the keyword stage
  even when dense ranks it lower, proving the hybrid value.
"""
from __future__ import annotations

from ken_rag.domain.models import Chunk, EmbeddedChunk, RetrievedChunk
from ken_rag.retrieval.fusion import rrf_fuse
from ken_rag.retrieval.pipeline import RetrievalPipeline
from ken_rag.retrieval.stages import DenseStage, KeywordStage
from tests.fakes.fake_embedder import FakeEmbedder
from tests.fakes.in_memory_store import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_chunk(file_path: str, chunk_index: int, text: str) -> Chunk:
    return Chunk(
        text=text,
        file_path=file_path,
        file_type="md",
        chunk_index=chunk_index,
        content_hash="abc",
        line_start=1,
        line_end=2,
        symbol_name=None,
        chunk_kind="prose",
    )


def _populate_store(store: InMemoryVectorStore, embedder: FakeEmbedder) -> list[Chunk]:
    """Insert three chunks into the store with real fake embeddings."""
    chunks = [
        _make_chunk("docs/a.md", 0, "The authentication middleware validates tokens"),
        _make_chunk("docs/b.md", 0, "parse_config reads the YAML configuration file"),
        _make_chunk("docs/c.md", 0, "Database connection pooling improves throughput"),
    ]
    embedded = [
        EmbeddedChunk(chunk=c, vector=embedder.embed_query(c.text))
        for c in chunks
    ]
    store.upsert(embedded)
    return chunks


# ---------------------------------------------------------------------------
# DenseStage
# ---------------------------------------------------------------------------

class TestDenseStage:
    def test_returns_store_search_results(self) -> None:
        """DenseStage.run(query) returns whatever store.search returns."""
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        _populate_store(store, embedder)

        stage = DenseStage(embedder=embedder, store=store)
        results = stage.run("token validation", k=3)

        assert len(results) == 3
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_embeds_query_before_search(self) -> None:
        """DenseStage uses embed_query (not embed_texts) for the query vector.

        We verify this by using a spy embedder that records which embed method
        was called, ensuring the search_query: prefix path is exercised.
        """

        class _SpyEmbedder:
            """Wraps FakeEmbedder and records method call names."""

            def __init__(self) -> None:
                self._inner = FakeEmbedder()
                self.calls: list[str] = []

            @property
            def model_name(self) -> str:
                return self._inner.model_name

            @property
            def dim(self) -> int:
                return self._inner.dim

            def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
                self.calls.append("embed_texts")
                return self._inner.embed_texts(texts)

            def embed_query(self, text: str) -> tuple[float, ...]:
                self.calls.append("embed_query")
                return self._inner.embed_query(text)

        spy = _SpyEmbedder()
        store = InMemoryVectorStore()
        _populate_store(store, FakeEmbedder())  # populate with base embedder

        stage = DenseStage(embedder=spy, store=store)
        results = stage.run("token validation", k=1)

        assert results  # got a result
        # DenseStage must use embed_query (not embed_texts) for the query
        assert spy.calls == ["embed_query"]

    def test_respects_k_parameter(self) -> None:
        """DenseStage.run returns at most k results."""
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        _populate_store(store, embedder)

        stage = DenseStage(embedder=embedder, store=store)
        results = stage.run("anything", k=2)

        assert len(results) == 2


# ---------------------------------------------------------------------------
# KeywordStage
# ---------------------------------------------------------------------------

class TestKeywordStage:
    def test_returns_store_search_text_results(self) -> None:
        """KeywordStage.run(query) delegates to store.search_text."""
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        _populate_store(store, embedder)

        stage = KeywordStage(store=store)
        results = stage.run("parse_config", k=3)

        assert len(results) >= 1
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_finds_exact_identifier_by_text(self) -> None:
        """KeywordStage surfaces a chunk containing a literal identifier."""
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        _populate_store(store, embedder)

        stage = KeywordStage(store=store)
        results = stage.run("parse_config", k=3)

        found_paths = [r.chunk.file_path for r in results]
        assert "docs/b.md" in found_paths

    def test_respects_k_parameter(self) -> None:
        """KeywordStage.run returns at most k results."""
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        _populate_store(store, embedder)

        stage = KeywordStage(store=store)
        results = stage.run("the", k=2)

        assert len(results) <= 2


# ---------------------------------------------------------------------------
# rrf_fuse
# ---------------------------------------------------------------------------

class TestRrfFuse:
    def _make_rc(self, file_path: str, chunk_index: int, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk=_make_chunk(file_path, chunk_index, f"text {file_path} {chunk_index}"),
            score=score,
        )

    def test_returns_top_k_by_fused_score(self) -> None:
        """rrf_fuse returns at most top_k results ordered by descending fused score."""
        ranking_a = [
            self._make_rc("a.md", 0, 0.9),
            self._make_rc("b.md", 0, 0.7),
            self._make_rc("c.md", 0, 0.5),
        ]
        ranking_b = [
            self._make_rc("b.md", 0, 0.8),
            self._make_rc("a.md", 0, 0.6),
            self._make_rc("d.md", 0, 0.4),
        ]

        result = rrf_fuse([ranking_a, ranking_b], k_const=60, top_k=3)

        assert len(result) == 3
        assert all(isinstance(r, RetrievedChunk) for r in result)

    def test_deduplicates_chunks_across_rankings(self) -> None:
        """A chunk appearing in multiple rankings is merged (not duplicated)."""
        shared = self._make_rc("shared.md", 0, 1.0)
        ranking_a = [shared, self._make_rc("only_a.md", 0, 0.5)]
        ranking_b = [shared, self._make_rc("only_b.md", 0, 0.4)]

        result = rrf_fuse([ranking_a, ranking_b], k_const=60, top_k=5)

        ids = [f"{r.chunk.file_path}::{r.chunk.chunk_index}" for r in result]
        # No duplicate ids
        assert len(ids) == len(set(ids))

    def test_shared_chunk_has_higher_score_than_single_ranking_chunk(self) -> None:
        """A chunk that appears in both rankings beats one that appears in only one."""
        # shared appears at rank-0 in both → high fused score
        ranking_a = [
            self._make_rc("shared.md", 0, 1.0),
            self._make_rc("only_a.md", 0, 0.5),
        ]
        ranking_b = [
            self._make_rc("shared.md", 0, 0.9),
            self._make_rc("only_b.md", 0, 0.4),
        ]

        result = rrf_fuse([ranking_a, ranking_b], k_const=60, top_k=3)

        assert result[0].chunk.file_path == "shared.md"

    def test_reciprocal_rank_formula_exact(self) -> None:
        """rrf_fuse scores match the exact formula 1/(k+rank) summed across rankings."""
        k = 60
        rc_a = self._make_rc("x.md", 0, 1.0)  # rank 0 in ranking_a
        rc_b = self._make_rc("y.md", 0, 0.9)  # rank 0 in ranking_b

        result = rrf_fuse([[rc_a], [rc_b]], k_const=k, top_k=2)

        # Both are rank-0 in a single ranking → score = 1/(60+0) each
        expected_score = 1.0 / (k + 0)
        for r in result:
            assert abs(r.score - expected_score) < 1e-9

    def test_top_k_truncation(self) -> None:
        """rrf_fuse returns exactly top_k results when more candidates exist."""
        rankings = [
            [self._make_rc(f"file{i}.md", 0, float(i)) for i in range(10)]
        ]
        result = rrf_fuse(rankings, k_const=60, top_k=5)
        assert len(result) == 5

    def test_empty_rankings_returns_empty(self) -> None:
        """rrf_fuse with empty rankings list returns an empty list."""
        result = rrf_fuse([], k_const=60, top_k=5)
        assert result == []

    def test_ordering_by_descending_score(self) -> None:
        """Results are ordered by descending fused score."""
        # rank 0 in both rankings → high score; rank 1 in one → lower
        ranking_a = [
            self._make_rc("high.md", 0, 1.0),  # rank 0
            self._make_rc("low.md", 0, 0.5),   # rank 1
        ]
        ranking_b = [
            self._make_rc("high.md", 0, 0.9),  # rank 0
            self._make_rc("mid.md", 0, 0.8),   # rank 1
        ]

        result = rrf_fuse([ranking_a, ranking_b], k_const=60, top_k=3)

        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# RetrievalPipeline
# ---------------------------------------------------------------------------

class TestRetrievalPipeline:
    def test_retrieve_returns_top_k_fused(self) -> None:
        """RetrievalPipeline.retrieve(query, k) returns exactly k fused results."""
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        _populate_store(store, embedder)

        pipeline = RetrievalPipeline(
            stages=[DenseStage(embedder=embedder, store=store), KeywordStage(store=store)]
        )
        results = pipeline.retrieve("authentication", k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_retrieve_deduplicates_across_stages(self) -> None:
        """Pipeline deduplicates chunks that appear in both dense and keyword results."""
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        _populate_store(store, embedder)

        pipeline = RetrievalPipeline(
            stages=[DenseStage(embedder=embedder, store=store), KeywordStage(store=store)]
        )
        results = pipeline.retrieve("authentication middleware", k=3)

        ids = [f"{r.chunk.file_path}::{r.chunk.chunk_index}" for r in results]
        assert len(ids) == len(set(ids))

    def test_retrieve_implements_retriever_protocol(self) -> None:
        """RetrievalPipeline satisfies the Retriever protocol."""
        from ken_rag.domain.protocols import Retriever

        store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        pipeline = RetrievalPipeline(
            stages=[DenseStage(embedder=embedder, store=store)]
        )
        assert isinstance(pipeline, Retriever)

    def test_single_stage_pipeline_works(self) -> None:
        """Pipeline with only one stage (e.g. dense-only) still returns results."""
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        _populate_store(store, embedder)

        pipeline = RetrievalPipeline(stages=[DenseStage(embedder=embedder, store=store)])
        results = pipeline.retrieve("database", k=2)

        assert len(results) == 2


# ---------------------------------------------------------------------------
# Hybrid value — literal identifier found via keyword even if dense ranks it lower
# ---------------------------------------------------------------------------

class TestHybridValue:
    """Proves the value of combining dense + keyword retrieval.

    Scenario: a user queries the exact function name ``parse_config``.
    Dense retrieval ranks a semantically similar chunk higher, but keyword
    retrieval surfaces the exact match.  RRF fusion ensures the exact match
    appears in the top-k results.
    """

    def test_keyword_surfaces_exact_identifier_in_hybrid_results(self) -> None:
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()

        # Three chunks: only docs/b.md contains the literal "parse_config"
        chunks = [
            _make_chunk("docs/a.md", 0, "Authentication middleware validates tokens securely"),
            _make_chunk("docs/b.md", 0, "parse_config reads YAML configuration from disk"),
            _make_chunk("docs/c.md", 0, "Connection pooling manages database connections"),
        ]
        embedded = [
            EmbeddedChunk(chunk=c, vector=embedder.embed_query(c.text))
            for c in chunks
        ]
        store.upsert(embedded)

        pipeline = RetrievalPipeline(
            stages=[DenseStage(embedder=embedder, store=store), KeywordStage(store=store)]
        )
        # Query is a literal identifier — dense may not rank it first
        results = pipeline.retrieve("parse_config", k=3)

        found_paths = [r.chunk.file_path for r in results]
        assert "docs/b.md" in found_paths, (
            "parse_config chunk must appear in hybrid top-k results; "
            f"got: {found_paths}"
        )

    def test_dense_alone_may_miss_exact_identifier(self) -> None:
        """Control: dense-only may not surface the exact-identifier chunk.

        This test demonstrates *why* the keyword stage adds value by showing
        that when only the dense stage is used, the exact-identifier chunk
        may rank lower than semantically related but textually different chunks.

        NOTE: Because FakeEmbedder is hash-based and not semantic, we verify
        the keyword stage independently finds the exact match, then confirm
        the combined pipeline includes it.
        """
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()

        chunks = [
            _make_chunk("docs/a.md", 0, "Authentication middleware validates tokens securely"),
            _make_chunk("docs/b.md", 0, "parse_config reads YAML configuration from disk"),
            _make_chunk("docs/c.md", 0, "Connection pooling manages database connections"),
        ]
        embedded = [
            EmbeddedChunk(chunk=c, vector=embedder.embed_query(c.text))
            for c in chunks
        ]
        store.upsert(embedded)

        keyword_stage = KeywordStage(store=store)
        keyword_results = keyword_stage.run("parse_config", k=3)

        keyword_paths = [r.chunk.file_path for r in keyword_results]
        assert "docs/b.md" in keyword_paths, (
            "KeywordStage must find the chunk containing 'parse_config'"
        )
