"""ken_rag.retrieval — hybrid retrieval pipeline.

Public surface:
    - ``RetrievalPipeline``: assembles stages and fuses via RRF.
    - ``DenseStage``: vector-similarity stage.
    - ``KeywordStage``: full-text-search stage.
    - ``VectorRetriever``: dense retrieval primitive (embed + search).
    - ``rrf_fuse``: standalone RRF fusion function.
"""
from ken_rag.retrieval.fusion import rrf_fuse
from ken_rag.retrieval.pipeline import RetrievalPipeline
from ken_rag.retrieval.retriever import VectorRetriever
from ken_rag.retrieval.stages import DenseStage, KeywordStage, Stage

__all__ = [
    "rrf_fuse",
    "RetrievalPipeline",
    "VectorRetriever",
    "DenseStage",
    "KeywordStage",
    "Stage",
]
