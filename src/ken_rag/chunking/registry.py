"""ChunkerRegistry — maps FileType values to Chunker implementations.

Phase 1 wires:
  TXT, MD  → ProseChunker
  UNKNOWN  → FallbackChunker
  CODE     → **not wired** (Phase 2 seam — use ``register()`` to wire later)
  PDF      → not wired (Phase 1 PDF chunking not in scope)

The ``register(file_type, chunker)`` method provides the seam for Phase 2 to
attach the code-aware chunker WITHOUT any import of chunking/code/ here.
"""
from __future__ import annotations

from ken_rag.domain.enums import FileType
from ken_rag.domain.protocols import Chunker


class ChunkerRegistry:
    """Registry mapping :class:`~ken_rag.domain.enums.FileType` to a Chunker.

    Construction is done through :meth:`default` to keep the mapping in one
    place and make the seam for Phase 2 obvious.
    """

    def __init__(self) -> None:
        self._map: dict[FileType, Chunker] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "ChunkerRegistry":
        """Build the registry pre-wired for Phase 1 (prose + fallback only)."""
        # Import here to avoid circular imports at module level
        from ken_rag.chunking.fallback_chunker import FallbackChunker
        from ken_rag.chunking.prose_chunker import ProseChunker

        registry = cls()
        prose = ProseChunker()
        fallback = FallbackChunker()

        registry._map[FileType.TXT] = prose
        registry._map[FileType.MD] = prose
        registry._map[FileType.UNKNOWN] = fallback
        # CODE and PDF are intentionally NOT wired here.
        # Phase 2 calls registry.register(FileType.CODE, code_chunker) externally.

        return registry

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, file_type: FileType) -> Chunker | None:
        """Return the chunker for *file_type*, or ``None`` if unregistered."""
        return self._map.get(file_type)

    def register(self, file_type: FileType, chunker: Chunker) -> None:
        """Register or replace the chunker for *file_type*.

        This is the Phase 2 seam — call this from the DI root (``cli/context.py``)
        to wire the code-aware chunker after it is built in Phase 2, without
        importing ``chunking/code/`` from this module.
        """
        self._map[file_type] = chunker
