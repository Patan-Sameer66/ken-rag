"""Fallback chunker: line-window chunker for UNKNOWN file types.

Used when no language-specific or prose chunker can handle the file.
Slides a window over the raw lines of the document, emitting a new chunk
each time the token budget is reached.  No sentence-boundary awareness or
overlap is applied — this is a best-effort chunker for arbitrary binary-ish
or unstructured text.

Resulting chunks carry ``chunk_kind = ChunkKind.FALLBACK``.
"""
from __future__ import annotations

from ken_rag.chunking.base import build_chunk, estimate_tokens
from ken_rag.config.defaults import PROSE_MAX_TOK
from ken_rag.domain.enums import ChunkKind
from ken_rag.domain.models import Chunk, ParsedDocument


class FallbackChunker:
    """Line-window chunker for files with UNKNOWN or unsupported types.

    Satisfies the :class:`~ken_rag.domain.protocols.Chunker` protocol.
    """

    def __init__(self, max_tokens: int = PROSE_MAX_TOK) -> None:
        self._max_tokens = max_tokens

    def chunk(
        self,
        doc: ParsedDocument,
        file_path: str,
        content_hash: str,
    ) -> list[Chunk]:
        """Split *doc* into line-windowed FALLBACK chunks."""
        text = doc.text
        if not text or not text.strip():
            # Empty document — return empty list (no chunk to emit)
            return []

        lines = text.splitlines()
        chunks: list[Chunk] = []
        chunk_index = 0
        current_lines: list[str] = []
        line_start = 1  # 1-indexed

        for line_no, line in enumerate(lines, start=1):
            # If adding this line would exceed the budget and we have content,
            # flush the current window.  Use the joined estimate to avoid
            # integer-truncation accumulation errors.
            candidate = current_lines + [line]
            candidate_tokens = estimate_tokens("\n".join(candidate))

            if candidate_tokens > self._max_tokens and current_lines:
                chunk_text = "\n".join(current_lines)
                chunks.append(
                    build_chunk(
                        text=chunk_text,
                        file_path=file_path,
                        file_type=doc.file_type,
                        content_hash=content_hash,
                        chunk_index=chunk_index,
                        line_start=line_start,
                        line_end=line_no - 1,
                        symbol_name=None,
                        chunk_kind=ChunkKind.FALLBACK,
                    )
                )
                chunk_index += 1
                current_lines = []
                line_start = line_no

            current_lines.append(line)

        # Flush remaining lines
        if current_lines:
            chunk_text = "\n".join(current_lines)
            chunks.append(
                build_chunk(
                    text=chunk_text,
                    file_path=file_path,
                    file_type=doc.file_type,
                    content_hash=content_hash,
                    chunk_index=chunk_index,
                    line_start=line_start,
                    line_end=len(lines),
                    symbol_name=None,
                    chunk_kind=ChunkKind.FALLBACK,
                )
            )

        return chunks
