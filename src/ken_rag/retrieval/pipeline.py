"""RetrievalPipeline — hybrid retrieval assembler.

Runs each configured stage, collects their ranked outputs, and fuses them
via Reciprocal Rank Fusion (RRF).

Design intent (extensibility):
    v1 ships with ``[DenseStage, KeywordStage]``.  A v2 ``RerankStage`` can
    be appended to the list without changing any existing code:

        pipeline = RetrievalPipeline(
            stages=[DenseStage(emb, store), KeywordStage(store), RerankStage(model)]
        )

    Each stage only needs to implement ``run(query, k) -> list[RetrievedChunk]``.
"""
from __future__ import annotations

from ken_rag.domain.models import RetrievedChunk
from ken_rag.retrieval.fusion import rrf_fuse
from ken_rag.retrieval.stages import Stage


class RetrievalPipeline:
    """Hybrid retrieval pipeline that fuses multiple stages via RRF.

    Implements the ``Retriever`` protocol (``ken_rag.domain.protocols``).

    Parameters
    ----------
    stages:
        Ordered list of retrieval stages.  Each must implement
        ``run(query: str, k: int) -> list[RetrievedChunk]``.
    """

    def __init__(self, stages: list[Stage]) -> None:
        self._stages = stages

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Run all stages and return the top-k RRF-fused results.

        Each stage is asked for ``k`` candidates.  The rankings are then
        passed to ``rrf_fuse`` which deduplicates and orders by fused score.
        """
        rankings: list[list[RetrievedChunk]] = [
            stage.run(query, k) for stage in self._stages
        ]
        return rrf_fuse(rankings, top_k=k)
