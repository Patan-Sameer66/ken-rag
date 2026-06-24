"""ken_rag.pipeline — hand-written ingest and query orchestration.

The anti-LangChain mandate: combined ingest.py + query.py stays < 200 lines.
"""
from ken_rag.pipeline.ingest import IngestPipeline
from ken_rag.pipeline.query import QueryPipeline

__all__ = ["IngestPipeline", "QueryPipeline"]
