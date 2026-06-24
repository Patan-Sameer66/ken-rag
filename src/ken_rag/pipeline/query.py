"""Query pipeline — embed → retrieve → prompt → stream → cite.

Guards the embedder name against the stored metadata before any retrieval.
``ask_stream`` is the primary interface; ``ask`` joins the stream into an Answer.
"""
from __future__ import annotations

from typing import Iterator

from ken_rag.domain.errors import EmbedderMismatchError
from ken_rag.domain.models import Answer
from ken_rag.domain.protocols import Embedder, Generator, MetadataStore
from ken_rag.generation import citation as citation_module
from ken_rag.generation.prompt import PromptBuilder
from ken_rag.retrieval.pipeline import RetrievalPipeline

_META_EMBEDDER_KEY = "embedder_name"


class QueryPipeline:
    """Orchestrates retrieval-augmented generation for a single question.

    Parameters
    ----------
    embedder:
        Used to embed the query string.
    retrieval_pipeline:
        Hybrid retrieval (DenseStage + KeywordStage, RRF-fused).
    generator:
        Streams tokens for the assembled prompt.
    metadata_store:
        Checked to validate the embedder matches the index.
    settings:
        Provides ``k`` (top-k) and ``num_ctx`` for generation.
    """

    def __init__(
        self,
        embedder: Embedder,
        retrieval_pipeline: RetrievalPipeline,
        generator: Generator,
        metadata_store: MetadataStore,
        settings: object,  # Settings (duck-typed to avoid circular import)
    ) -> None:
        self._embedder = embedder
        self._retrieval = retrieval_pipeline
        self._generator = generator
        self._meta = metadata_store
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask_stream(self, question: str) -> Iterator[str]:
        """Yield tokens for *question*, validating the embedder first.

        After the last token is yielded the caller should call
        ``get_citations(retrieved)`` — or use the higher-level ``ask()`` which
        handles this automatically.

        Raises
        ------
        EmbedderMismatchError
            If the stored embedder name differs from the current one.
        """
        self._guard_embedder()
        retrieved = self._retrieval.retrieve(question, self._settings.k)
        # Stash retrieved so get_last_citations() can build citations post-stream.
        self._last_retrieved = retrieved
        prompt = PromptBuilder.build(question, retrieved)
        yield from self._generator.stream(prompt, num_ctx=self._settings.num_ctx)

    def ask(self, question: str) -> Answer:
        """Return a complete Answer (joins streamed tokens, builds citations).

        Parameters
        ----------
        question:
            The user's question.

        Returns
        -------
        Answer
            Contains the full answer text and deduplicated citations.

        Raises
        ------
        EmbedderMismatchError
            If the stored embedder name differs from the current one.
        """
        self._guard_embedder()
        retrieved = self._retrieval.retrieve(question, self._settings.k)
        self._last_retrieved = retrieved
        prompt = PromptBuilder.build(question, retrieved)
        tokens = list(self._generator.stream(prompt, num_ctx=self._settings.num_ctx))
        text = "".join(tokens)
        citations = citation_module.build(retrieved)
        return Answer(text=text, citations=citations)

    def get_last_citations(self):
        """Build citations from the last retrieval (use after ``ask_stream``)."""
        retrieved = getattr(self, "_last_retrieved", [])
        return citation_module.build(retrieved)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _guard_embedder(self) -> None:
        """Raise EmbedderMismatchError if the stored embedder name differs."""
        stored = self._meta.get(_META_EMBEDDER_KEY)
        current = self._embedder.model_name
        if stored is not None and stored != current:
            raise EmbedderMismatchError(
                f"Index was built with embedder '{stored}', "
                f"but the current embedder is '{current}'.",
                hint=(
                    f"Either switch back to '{stored}' or delete the index "
                    f"(remove the .ken/ directory) and re-run `ken add`."
                ),
            )
