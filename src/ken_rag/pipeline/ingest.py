"""Ingest pipeline — walk → detect → hash → parse → chunk → embed → upsert.

Phase-1 semantics:
- Every file whose hash differs from the stored record is (re-)indexed.
- First ingest writes embedder name into the metadata store.
- Subsequent ingests re-validate the stored embedder name; a mismatch raises
  ``EmbedderMismatchError`` with a fix-it hint before any embedding occurs.
- Phase 3 will add ignore-filter + full FileTracker diff (ADDED/MODIFIED/
  DELETED/UNCHANGED); for Phase 1 we simply (re-)index changed-by-hash files.
"""
from __future__ import annotations

from pathlib import Path

from ken_rag.domain.errors import EmbedderMismatchError
from ken_rag.domain.models import EmbeddedChunk
from ken_rag.domain.protocols import (
    Chunker,
    Embedder,
    MetadataStore,
    Parser,
    VectorStore,
)
from ken_rag.parsing.registry import detect_file_type, ParserRegistry
from ken_rag.tracking.hasher import sha256_file
from ken_rag.tracking.walker import walk_files

_META_EMBEDDER_KEY = "embedder_name"


class IngestPipeline:
    """Orchestrates file ingestion into the vector store.

    Parameters
    ----------
    parser_registry:
        Maps FileType → Parser.  Files whose type has no parser are skipped.
    chunker_registry:
        Maps FileType → Chunker.  Files whose type has no chunker are skipped.
    embedder:
        Embedder used to convert chunk texts into vectors.
    vector_store:
        Destination for embedded chunks.
    metadata_store:
        Used to persist and validate the embedder name.
    """

    def __init__(
        self,
        parser_registry: ParserRegistry,
        chunker_registry: object,  # ChunkerRegistry (duck-typed to avoid circular)
        embedder: Embedder,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
    ) -> None:
        self._parsers = parser_registry
        self._chunkers = chunker_registry
        self._embedder = embedder
        self._store = vector_store
        self._meta = metadata_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, path: Path) -> int:
        """Ingest all files under *path* and return the number of files indexed.

        Parameters
        ----------
        path:
            A directory to walk, or a single file to ingest directly.

        Returns
        -------
        int
            Count of files that were (re-)indexed.

        Raises
        ------
        EmbedderMismatchError
            If the stored embedder name differs from the current one.
        """
        self._guard_embedder()

        indexed = 0
        for file_path in walk_files(path):
            file_type = detect_file_type(file_path)
            parser: Parser | None = self._parsers.get(file_type)
            chunker: Chunker | None = self._chunkers.get(file_type)
            if parser is None or chunker is None:
                continue  # unsupported type — skip silently

            content_hash = sha256_file(file_path)

            # Phase-1 change detection: skip unchanged files.
            existing = self._store.get_file(str(file_path))
            if existing is not None and existing.content_hash == content_hash:
                continue  # unchanged — no work needed

            doc = parser.parse(file_path)
            chunks = chunker.chunk(doc, str(file_path), content_hash)
            if not chunks:
                continue

            texts = [c.text for c in chunks]
            vectors = self._embedder.embed_texts(texts)
            embedded = [
                EmbeddedChunk(chunk=c, vector=v)
                for c, v in zip(chunks, vectors, strict=True)
            ]

            # Lifecycle: delete existing chunks for this file, then upsert fresh.
            self._store.delete_by_file(str(file_path))
            self._store.upsert(embedded)
            indexed += 1

        return indexed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _guard_embedder(self) -> None:
        """Validate or initialise the stored embedder name."""
        stored = self._meta.get(_META_EMBEDDER_KEY)
        current = self._embedder.model_name
        if stored is None:
            # First ingest — record the embedder name.
            self._meta.set(_META_EMBEDDER_KEY, current)
        elif stored != current:
            raise EmbedderMismatchError(
                f"Index was built with embedder '{stored}', "
                f"but the current embedder is '{current}'.",
                hint=(
                    f"Either switch back to '{stored}' or delete the index "
                    f"(remove the .ken/ directory) and re-run `ken add`."
                ),
            )
