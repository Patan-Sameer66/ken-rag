"""VectorRetriever — the dense retrieval primitive.

Encapsulates the embed-then-search pattern so that ``DenseStage`` and any
future consumer (e.g. a re-rank stage that needs to fetch more candidates)
can reuse it without duplicating the embedding call.
"""
from __future__ import annotations

from ken_rag.domain.models import RetrievedChunk
from ken_rag.domain.protocols import Embedder, VectorStore


class VectorRetriever:
    """Embed a query and search the vector store.

    Parameters
    ----------
    embedder:
        Used to convert the query string into a dense vector.
    store:
        The vector store to search against.
    """

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Embed *query* and return the top-k nearest chunks."""
        vector = self._embedder.embed_query(query)
        return self._store.search(vector, k)
