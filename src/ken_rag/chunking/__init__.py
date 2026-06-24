"""Chunking package for ken-rag.

Exports the registry and concrete chunkers needed by the ingest pipeline.
"""
from ken_rag.chunking.registry import ChunkerRegistry

__all__ = ["ChunkerRegistry"]
