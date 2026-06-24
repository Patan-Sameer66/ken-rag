"""Citation construction for ken-rag generation.

:func:`build` converts a ranked list of :class:`~ken_rag.domain.models.RetrievedChunk`
objects into a deduplicated, order-preserved tuple of
:class:`~ken_rag.domain.models.Citation` objects.

Deduplication key
-----------------
Two chunks are considered the same citation when they share the same
``(file_path, line_start, line_end)`` triple.  The *first* occurrence in the
input list wins; subsequent duplicates are silently dropped.
"""
from __future__ import annotations

from ken_rag.domain.models import Citation, RetrievedChunk


def build(retrieved: list[RetrievedChunk]) -> tuple[Citation, ...]:
    """Build a deduplicated, order-preserved tuple of citations.

    Parameters
    ----------
    retrieved:
        Ranked list of retrieved chunks, typically from the retrieval pipeline.

    Returns
    -------
    tuple[Citation, ...]
        One :class:`Citation` per unique ``(file_path, line_start, line_end)``
        triple, in the order of first occurrence.
    """
    seen: set[tuple[str, int, int]] = set()
    citations: list[Citation] = []

    for rc in retrieved:
        chunk = rc.chunk
        key = (chunk.file_path, chunk.line_start, chunk.line_end)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                file_path=chunk.file_path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                symbol_name=chunk.symbol_name,
            )
        )

    return tuple(citations)
