"""Tests for OllamaEmbedder — Task 1.5.

Covers:
- search_document: prefix on embed_texts (CRITICAL invariant)
- search_query: prefix on embed_query (CRITICAL invariant)
- batching in groups of batch_size
- DimensionMismatchError when returned vector has wrong dimension
- model_name and dim properties
- embed_query returns a tuple of floats
- embed_texts returns list of tuples
"""
from __future__ import annotations

import pytest

from ken_rag.domain.errors import DimensionMismatchError
from ken_rag.embedding.ollama_embedder import OllamaEmbedder

# ---------------------------------------------------------------------------
# Spy client — records calls without hitting a real server
# ---------------------------------------------------------------------------


class _SpyClient:
    """Fake Ollama client that records every embed() call."""

    def __init__(self, dim: int = 768) -> None:
        self.calls: list[list[str]] = []
        self._dim = dim

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        self.calls.append(list(inputs))
        return [[0.0] * self._dim for _ in inputs]


# ---------------------------------------------------------------------------
# Prefix invariants (CRITICAL)
# ---------------------------------------------------------------------------


def test_documents_get_search_document_prefix():
    """embed_texts() prepends 'search_document: ' to every text."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text")
    emb.embed_texts(["hello", "world"])
    assert spy.calls[0] == ["search_document: hello", "search_document: world"]


def test_query_gets_search_query_prefix():
    """embed_query() prepends 'search_query: ' to the query text."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text")
    emb.embed_query("how to log in")
    assert spy.calls[0] == ["search_query: how to log in"]


def test_embed_texts_prefix_is_exact():
    """Prefix must be exactly 'search_document: ' (no extra spaces/cases)."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text")
    emb.embed_texts(["test"])
    assert spy.calls[0][0].startswith("search_document: ")


def test_embed_query_prefix_is_exact():
    """Prefix must be exactly 'search_query: '."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text")
    emb.embed_query("test")
    assert spy.calls[0][0].startswith("search_query: ")


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_batches_in_chunks_of_64():
    """130 texts → calls with lengths [64, 64, 2]."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text", batch_size=64)
    emb.embed_texts([f"t{i}" for i in range(130)])
    assert [len(c) for c in spy.calls] == [64, 64, 2]


def test_batches_custom_batch_size():
    """Custom batch_size=10 with 25 texts → calls with lengths [10, 10, 5]."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text", batch_size=10)
    emb.embed_texts([f"doc{i}" for i in range(25)])
    assert [len(c) for c in spy.calls] == [10, 10, 5]


def test_batching_preserves_all_texts():
    """The full set of prefixed texts is passed across all batches."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text", batch_size=3)
    emb.embed_texts(["a", "b", "c", "d"])
    all_sent = [item for batch in spy.calls for item in batch]
    assert all_sent == [
        "search_document: a",
        "search_document: b",
        "search_document: c",
        "search_document: d",
    ]


def test_single_batch_when_texts_fit():
    """Fewer texts than batch_size → exactly one call."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text", batch_size=64)
    emb.embed_texts(["x", "y"])
    assert len(spy.calls) == 1


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


def test_embed_texts_returns_list_of_tuples():
    """embed_texts returns list[tuple[float, ...]]."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text")
    result = emb.embed_texts(["hi"])
    assert isinstance(result, list)
    assert isinstance(result[0], tuple)


def test_embed_query_returns_tuple():
    """embed_query returns a single tuple[float, ...]."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text")
    result = emb.embed_query("test")
    assert isinstance(result, tuple)


def test_embed_texts_vector_length():
    """Each returned tuple has length == dim."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text")
    result = emb.embed_texts(["hi"])
    assert len(result[0]) == 768


def test_embed_query_vector_length():
    """Returned query tuple has length == dim."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text")
    result = emb.embed_query("test")
    assert len(result) == 768


# ---------------------------------------------------------------------------
# Dimension guard
# ---------------------------------------------------------------------------


def test_embed_texts_wrong_dim_raises_dimension_mismatch():
    """DimensionMismatchError when client returns wrong-dimension vectors."""
    spy = _SpyClient(dim=512)  # wrong dim — should be 768
    emb = OllamaEmbedder(spy, "nomic-embed-text", dim=768)
    with pytest.raises(DimensionMismatchError) as exc_info:
        emb.embed_texts(["hello"])
    assert exc_info.value.hint != ""  # hint must be non-empty


def test_embed_query_wrong_dim_raises_dimension_mismatch():
    """DimensionMismatchError on embed_query when dim is wrong."""
    spy = _SpyClient(dim=512)
    emb = OllamaEmbedder(spy, "nomic-embed-text", dim=768)
    with pytest.raises(DimensionMismatchError):
        emb.embed_query("test")


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_model_name_property():
    """model_name returns the model name passed at construction."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text")
    assert emb.model_name == "nomic-embed-text"


def test_dim_property():
    """dim returns the expected vector dimension."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text", dim=768)
    assert emb.dim == 768


def test_default_dim_is_768():
    """Default dim is 768 (matching nomic-embed-text)."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text")
    assert emb.dim == 768


def test_default_batch_size_is_64():
    """Default batch_size is 64 as specified."""
    spy = _SpyClient()
    emb = OllamaEmbedder(spy, "nomic-embed-text")
    # Embed exactly 64 texts — should be one call
    emb.embed_texts([f"t{i}" for i in range(64)])
    assert len(spy.calls) == 1
    # Embed 65 texts — should be two calls
    spy2 = _SpyClient()
    emb2 = OllamaEmbedder(spy2, "nomic-embed-text")
    emb2.embed_texts([f"t{i}" for i in range(65)])
    assert len(spy2.calls) == 2
