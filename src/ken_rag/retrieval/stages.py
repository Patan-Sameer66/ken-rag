"""Retrieval stage implementations for the hybrid pipeline.

Each stage is a thin adapter between one retrieval mechanism and the common
``RetrievedChunk`` output format.  Stages are composable: the
``RetrievalPipeline`` collects their outputs and fuses them via RRF.

Extensibility note:
    To add a v2 re-rank stage, create a class with a ``run(query, k)`` method
    returning ``list[RetrievedChunk]`` and append it to the stages list passed
    to ``RetrievalPipeline``.  No existing code needs to change.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ken_rag.domain.models import RetrievedChunk
from ken_rag.domain.protocols import Embedder, VectorStore
from ken_rag.retrieval.retriever import VectorRetriever


@runtime_checkable
class Stage(Protocol):
    """Minimal protocol for a single retrieval stage.

    Any callable object with this signature can be composed into a
    ``RetrievalPipeline`` without subclassing.
    """

    def run(self, query: str, k: int) -> list[RetrievedChunk]:
        """Execute this stage for *query* and return at most *k* results."""
        ...


class DenseStage:
    """Vector-similarity retrieval stage.

    Embeds *query* via the ``Embedder`` and delegates to
    ``VectorStore.search``.  Internally uses a ``VectorRetriever`` to keep
    the embedding–search pairing in one place.
    """

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._retriever = VectorRetriever(embedder=embedder, store=store)

    def run(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return top-k chunks by dense cosine similarity."""
        return self._retriever.search(query, k)


class KeywordStage:
    """Full-text-search retrieval stage.

    Delegates directly to ``VectorStore.search_text``.  No embedding is
    performed, so this stage is fast and excels at exact-identifier queries.
    """

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def run(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return top-k chunks by full-text (keyword) search."""
        return self._store.search_text(query, k)
