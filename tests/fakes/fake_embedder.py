"""FakeEmbedder — deterministic 768-dim embedder for tests.

Implements the ``Embedder`` protocol without any network calls.
Vectors are derived from a stable hash of the (prefixed) text, so:
- The same text always produces the same vector.
- Different texts (almost certainly) produce different vectors.
- No Ollama daemon is required.

Usage::

    from tests.fakes.fake_embedder import FakeEmbedder

    emb = FakeEmbedder()
    vec = emb.embed_query("how does auth work?")
    assert len(vec) == 768
"""
from __future__ import annotations

import hashlib
import struct


_DIM = 768
_DOCUMENT_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


def _text_to_vector(text: str) -> tuple[float, ...]:
    """Deterministically map *text* to a 768-dim vector in [-1, 1].

    Strategy: take sha256(text + str(i)) for successive i values, interpret
    each digest byte as a signed int8 in [-128, 127], then normalize to [-1, 1].
    This avoids NaN/Inf values that can arise from interpreting arbitrary bytes
    as IEEE 754 floats.  Each 32-byte digest contributes 32 scalar values.
    """
    values: list[float] = []
    base = text.encode("utf-8")
    i = 0
    while len(values) < _DIM:
        digest = hashlib.sha256(base + str(i).encode()).digest()
        # Interpret each byte as signed int8 → normalize to [-1, 1]
        for b in struct.unpack("32b", digest):
            values.append(b / 128.0)
        i += 1
    return tuple(values[:_DIM])


class FakeEmbedder:
    """Deterministic 768-dim embedder compatible with the ``Embedder`` protocol.

    Applies the same nomic task prefixes as ``OllamaEmbedder`` so that
    prefix-invariant tests remain meaningful even with this fake.

    Parameters
    ----------
    model_name:
        Reported model name. Defaults to ``"fake-embed"``.
    dim:
        Vector dimension. Defaults to 768.
    """

    def __init__(
        self,
        model_name: str = "fake-embed",
        dim: int = _DIM,
    ) -> None:
        self._model_name = model_name
        self._dim = dim

    # ------------------------------------------------------------------
    # Embedder protocol
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """The fake model name."""
        return self._model_name

    @property
    def dim(self) -> int:
        """Expected vector dimension (768)."""
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        """Embed documents with the ``search_document:`` prefix (deterministic)."""
        return [_text_to_vector(f"{_DOCUMENT_PREFIX}{t}") for t in texts]

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed a query with the ``search_query:`` prefix (deterministic)."""
        return _text_to_vector(f"{_QUERY_PREFIX}{text}")
