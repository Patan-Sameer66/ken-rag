"""OllamaEmbedder — implements the Embedder protocol via OllamaClient.

Key invariants (locked by eng-review):
- Documents are embedded as ``search_document: <text>`` (nomic task prefix).
- Queries are embedded as ``search_query: <text>`` (nomic task prefix).
- Texts are batched in groups of *batch_size* (default 64) per HTTP call.
- Each returned vector must be exactly *dim* elements long; any deviation
  raises ``DimensionMismatchError``.
"""
from __future__ import annotations

from ken_rag.config.defaults import BATCH_SIZE, EMBED_DIM
from ken_rag.embedding.dimension import validate_vectors

# Type alias for the low-level client interface.
# We accept any object with an ``embed(model, inputs)`` method so that
# tests can inject a spy without importing OllamaClient.
_ClientLike = object  # structural duck type — see type: ignore comments


class OllamaEmbedder:
    """Batched document/query embedder using the Ollama /api/embed endpoint.

    Parameters
    ----------
    client:
        An object with ``embed(model: str, inputs: list[str]) -> list[list[float]]``.
        Typically an ``OllamaClient`` instance.
    model_name:
        Ollama model identifier, e.g. ``"nomic-embed-text"``.
    dim:
        Expected embedding dimension. Defaults to 768 (nomic-embed-text).
    batch_size:
        Maximum number of texts per HTTP call. Defaults to 64.
    """

    def __init__(
        self,
        client: _ClientLike,
        model_name: str,
        dim: int = EMBED_DIM,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self._client = client
        self._model = model_name
        self._dim = dim
        self._batch = batch_size

    # ------------------------------------------------------------------
    # Embedder protocol properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """The name of the embedding model."""
        return self._model

    @property
    def dim(self) -> int:
        """Expected vector dimension."""
        return self._dim

    # ------------------------------------------------------------------
    # Embedder protocol methods
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        """Embed a list of document texts with the ``search_document:`` prefix.

        Texts are sent in batches of at most *batch_size* per HTTP call.

        Parameters
        ----------
        texts:
            Raw document strings (without any prefix).

        Returns
        -------
        list[tuple[float, ...]]
            One 768-element tuple per input text, in the same order.

        Raises
        ------
        DimensionMismatchError
            If any returned vector has the wrong dimension.
        """
        out: list[tuple[float, ...]] = []
        for i in range(0, len(texts), self._batch):
            batch_texts = texts[i : i + self._batch]
            prefixed = [f"search_document: {t}" for t in batch_texts]
            raw_vectors = self._client.embed(self._model, prefixed)  # type: ignore[attr-defined]
            validate_vectors(raw_vectors, self._dim)
            out.extend(tuple(v) for v in raw_vectors)
        return out

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed a single query string with the ``search_query:`` prefix.

        Parameters
        ----------
        text:
            Raw query string (without any prefix).

        Returns
        -------
        tuple[float, ...]
            A single 768-element tuple.

        Raises
        ------
        DimensionMismatchError
            If the returned vector has the wrong dimension.
        """
        prefixed = [f"search_query: {text}"]
        raw_vectors = self._client.embed(self._model, prefixed)  # type: ignore[attr-defined]
        validate_vectors(raw_vectors, self._dim)
        return tuple(raw_vectors[0])
