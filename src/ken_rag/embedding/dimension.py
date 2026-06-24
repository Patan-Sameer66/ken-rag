"""Dimension validation helper for embedding vectors.

Extracted so it can be reused by any embedder implementation without
duplicating the check logic.
"""
from __future__ import annotations

from ken_rag.domain.errors import DimensionMismatchError


def validate_vectors(vectors: list[list[float]], expected_dim: int) -> list[list[float]]:
    """Check that every vector in *vectors* has exactly *expected_dim* elements.

    Parameters
    ----------
    vectors:
        Raw float vectors returned by the embedding backend.
    expected_dim:
        The dimension that every vector must match (e.g. 768 for nomic-embed-text).

    Returns
    -------
    list[list[float]]
        The same *vectors* list if all dimensions are correct.

    Raises
    ------
    DimensionMismatchError
        On the first vector whose length differs from *expected_dim*.
    """
    for v in vectors:
        if len(v) != expected_dim:
            raise DimensionMismatchError(
                f"Embedder returned {len(v)}-dim vector, expected {expected_dim}.",
                hint=(
                    f"Wrong embedding model? "
                    f"Expected nomic-embed-text ({expected_dim}-dim). "
                    f"Re-check the model name and try again."
                ),
            )
    return vectors
