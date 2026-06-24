"""Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Merges ranked lists produced by independent retrieval stages (dense, keyword,
etc.) into a single unified ranking without requiring score calibration.

Reference formula:
    fused_score(d) = Σ  1 / (k_const + rank(d, r))
                     r ∈ rankings

where rank is 0-based.
"""
from __future__ import annotations

from ken_rag.domain.models import Chunk, RetrievedChunk


def rrf_fuse(
    rankings: list[list[RetrievedChunk]],
    k_const: int = 60,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Merge ranked lists via Reciprocal Rank Fusion.

    Parameters
    ----------
    rankings:
        Each element is an ordered list of ``RetrievedChunk`` from one stage
        (rank 0 = most relevant).
    k_const:
        The RRF constant (default 60, standard literature value).  A higher
        value diminishes the importance of top-ranked positions.
    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[RetrievedChunk]
        Deduplicated, top-k chunks ordered by descending fused score.
        The ``score`` field on each result holds the RRF fused score.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}

    for ranking in rankings:
        for rank, rc in enumerate(ranking):
            cid = f"{rc.chunk.file_path}::{rc.chunk.chunk_index}"
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_const + rank)
            by_id[cid] = rc.chunk

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [RetrievedChunk(chunk=by_id[cid], score=s) for cid, s in ordered]
