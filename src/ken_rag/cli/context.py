"""DI root for ken-rag — assembles all concrete implementations.

``build_context`` is the single composition point.  Tests inject fakes by
passing an *overrides* dict whose values replace specific wired-up objects.

Supported override keys
-----------------------
``"embedder"``      — replaces OllamaEmbedder
``"generator"``     — replaces OllamaGenerator
``"vector_store"``  — replaces LanceVectorStore
``"metadata_store"``— replaces LanceMetadataStore

All other objects (parsers, chunkers, retrieval pipeline) are always built
from scratch using whatever embedder / store are in effect after overrides.
"""
from __future__ import annotations

from typing import Any

from ken_rag.config.settings import Settings
from ken_rag.generation.ollama_generator import OllamaGenerator
from ken_rag.pipeline.ingest import IngestPipeline
from ken_rag.pipeline.query import QueryPipeline
from ken_rag.retrieval.pipeline import RetrievalPipeline
from ken_rag.retrieval.stages import DenseStage, KeywordStage


class AppContext:
    """Container for all wired-up collaborators.

    Attributes are intentionally public so commands can access them directly.
    """

    def __init__(
        self,
        settings: Settings,
        ingest: IngestPipeline,
        query: QueryPipeline,
    ) -> None:
        self.settings = settings
        self.ingest = ingest
        self.query = query


def build_context(
    settings: Settings,
    *,
    overrides: dict[str, Any] | None = None,
) -> AppContext:
    """Assemble the full dependency graph and return an :class:`AppContext`.

    Parameters
    ----------
    settings:
        Immutable settings object (produced by ``load_settings``).
    overrides:
        Optional map of collaborator overrides for test injection.
        Recognised keys: ``"embedder"``, ``"generator"``,
        ``"vector_store"``, ``"metadata_store"``.

    Returns
    -------
    AppContext
        Ready-to-use context with ``ingest`` and ``query`` pipelines.
    """
    ov = overrides or {}

    # ------------------------------------------------------------------
    # Embedder
    # ------------------------------------------------------------------
    if "embedder" in ov:
        embedder = ov["embedder"]
    else:
        from ken_rag.embedding.ollama_embedder import OllamaEmbedder
        from ken_rag.llm.ollama_client import OllamaClient

        llm_client = OllamaClient(
            base_url=settings.ollama_url,
            timeout=float(settings.timeout_s),
        )
        embedder = OllamaEmbedder(
            client=llm_client,
            model_name=settings.embedder_name,
            dim=settings.embed_dim,
            batch_size=settings.batch_size,
        )

    # ------------------------------------------------------------------
    # Generator
    # ------------------------------------------------------------------
    if "generator" in ov:
        generator = ov["generator"]
    else:
        from ken_rag.llm.ollama_client import OllamaClient

        gen_client = OllamaClient(
            base_url=settings.ollama_url,
            timeout=float(settings.timeout_s),
        )
        generator = OllamaGenerator(client=gen_client, model_name=settings.llm_name)

    # ------------------------------------------------------------------
    # Stores
    # ------------------------------------------------------------------
    if "vector_store" in ov:
        vector_store = ov["vector_store"]
    else:
        from ken_rag.store.lancedb_store import LanceVectorStore

        vector_store = LanceVectorStore.open_or_create(
            str(settings.db_path), settings.embed_dim
        )

    if "metadata_store" in ov:
        metadata_store = ov["metadata_store"]
    else:
        from ken_rag.store.metadata_store import LanceMetadataStore

        metadata_store = LanceMetadataStore.open_or_create(str(settings.db_path))

    # ------------------------------------------------------------------
    # Parsers + chunkers
    # ------------------------------------------------------------------
    from ken_rag.parsing.registry import ParserRegistry
    from ken_rag.chunking.registry import ChunkerRegistry

    parser_registry = ParserRegistry.default()
    chunker_registry = ChunkerRegistry.default()

    # ------------------------------------------------------------------
    # Retrieval pipeline (hybrid: dense + keyword, RRF-fused)
    # ------------------------------------------------------------------
    retrieval_pipeline = RetrievalPipeline(
        stages=[
            DenseStage(embedder=embedder, store=vector_store),
            KeywordStage(store=vector_store),
        ]
    )

    # ------------------------------------------------------------------
    # Assemble pipelines
    # ------------------------------------------------------------------
    ingest = IngestPipeline(
        parser_registry=parser_registry,
        chunker_registry=chunker_registry,
        embedder=embedder,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )

    query = QueryPipeline(
        embedder=embedder,
        retrieval_pipeline=retrieval_pipeline,
        generator=generator,
        metadata_store=metadata_store,
        settings=settings,
    )

    return AppContext(settings=settings, ingest=ingest, query=query)
