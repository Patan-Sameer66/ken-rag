"""Prose chunker: header/paragraph-aware, sentence-safe, with overlap.

Algorithm
---------
1. Split the document on markdown ``#`` headers.  Each header begins a new
   logical section; the header text is captured as ``symbol_name``.
2. Within each section, split the body on blank-line paragraph boundaries.
3. Within each paragraph, split into individual sentences using a simple
   regex (end of ``.``, ``!``, or ``?`` not followed by a digit — avoids
   splitting "v1.0" or "99.9%").
4. Pack sentences greedily into chunks up to ``PROSE_MAX_TOK`` tokens.
   When a chunk is full, emit it, then seed the next chunk with the
   overlap window (trailing sentences from the previous chunk).
5. Never split mid-sentence: a sentence is always placed whole into one chunk.

The resulting chunks carry:
- ``chunk_kind = ChunkKind.PROSE``
- ``symbol_name`` = the heading text of the section they belong to, or
  ``None`` for unheaded preamble.
- ``chunk_index`` sequential across the whole document.
- ``content_hash`` and ``file_path`` propagated from the caller.
"""
from __future__ import annotations

import re
from ken_rag.chunking.base import build_chunk, estimate_tokens, overlap_window
from ken_rag.config.defaults import PROSE_MAX_TOK, PROSE_OVERLAP
from ken_rag.domain.enums import ChunkKind, FileType
from ken_rag.domain.models import Chunk, ParsedDocument

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches a markdown header line: one or more leading `#` chars + space + title.
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Sentence boundary: split after . ! ? when not followed by a digit or
# lowercase letter (avoids "v1.0", "e.g. foo").  We keep the delimiter on the
# left side of the split via a zero-width look-behind.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\(])")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences at `.`, `!`, `?` boundaries.

    The split preserves the punctuation at the end of each sentence.
    Empty strings are discarded.
    """
    raw = _SENTENCE_END_RE.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Split *text* on ``#`` headers.

    Returns a list of ``(symbol_name, body)`` tuples where *symbol_name* is
    the header text (stripped of ``#`` marks) or ``None`` for preamble content
    before the first header.
    """
    sections: list[tuple[str | None, str]] = []

    # Find all header positions
    header_matches = list(_HEADER_RE.finditer(text))

    if not header_matches:
        # No headers at all — treat the whole text as one unnamed section
        return [(None, text.strip())]

    # Preamble before the first header
    preamble = text[: header_matches[0].start()].strip()
    if preamble:
        sections.append((None, preamble))

    for i, match in enumerate(header_matches):
        header_text = match.group(2).strip()
        body_start = match.end()
        body_end = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(text)
        # Include the header line itself at the top of the body for context
        body = match.group(0) + "\n" + text[body_start:body_end].strip()
        sections.append((header_text, body.strip()))

    return sections


def _pack_sentences_into_chunks(
    sentences: list[str],
    symbol_name: str | None,
    file_path: str,
    file_type: FileType,
    content_hash: str,
    start_index: int,
    max_tokens: int,
    overlap_ratio: float,
) -> list[Chunk]:
    """Pack *sentences* greedily into Chunk objects respecting *max_tokens*.

    Overlap between consecutive chunks is seeded from the trailing sentences
    of the previous chunk.
    """
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    chunk_index = start_index

    # Track approximate line numbers (we use word-count-based lines as proxy)
    line_cursor: int = 1

    def _emit(sents: list[str]) -> None:
        nonlocal chunk_index, line_cursor
        if not sents:
            return
        text = " ".join(sents)
        line_count = max(1, text.count("\n") + 1)
        chunks.append(
            build_chunk(
                text=text,
                file_path=file_path,
                file_type=file_type,
                content_hash=content_hash,
                chunk_index=chunk_index,
                line_start=line_cursor,
                line_end=line_cursor + line_count - 1,
                symbol_name=symbol_name,
                chunk_kind=ChunkKind.PROSE,
            )
        )
        line_cursor += line_count
        chunk_index += 1

    for sentence in sentences:
        # If a single sentence exceeds the budget all by itself, emit it alone.
        # We cannot split mid-sentence, so this is the best we can do.
        if not current_sentences and estimate_tokens(sentence) > max_tokens:
            _emit([sentence])
            continue

        candidate = current_sentences + [sentence]
        candidate_tokens = estimate_tokens(" ".join(candidate))

        if candidate_tokens > max_tokens and current_sentences:
            # Flush current chunk
            _emit(current_sentences)
            # Seed next chunk with overlap window
            current_sentences = overlap_window(current_sentences, overlap_ratio)

            # Re-check: if overlap alone already exceeds budget, trim it
            while current_sentences and estimate_tokens(" ".join(current_sentences + [sentence])) > max_tokens:
                current_sentences = current_sentences[1:]

        current_sentences.append(sentence)

    # Flush remaining sentences
    if current_sentences:
        _emit(current_sentences)

    return chunks


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class ProseChunker:
    """Markdown-aware prose chunker with header boundaries and overlap.

    Satisfies the :class:`~ken_rag.domain.protocols.Chunker` protocol.
    """

    def __init__(
        self,
        max_tokens: int = PROSE_MAX_TOK,
        overlap_ratio: float = PROSE_OVERLAP,
    ) -> None:
        self._max_tokens = max_tokens
        self._overlap_ratio = overlap_ratio

    def chunk(
        self,
        doc: ParsedDocument,
        file_path: str,
        content_hash: str,
    ) -> list[Chunk]:
        """Split *doc* into sentence-safe, overlap-aware prose chunks."""
        text = doc.text.strip()
        if not text:
            return [
                build_chunk(
                    text="",
                    file_path=file_path,
                    file_type=doc.file_type,
                    content_hash=content_hash,
                    chunk_index=0,
                    line_start=1,
                    line_end=1,
                    symbol_name=None,
                    chunk_kind=ChunkKind.PROSE,
                )
            ]

        sections = _split_into_sections(text)
        all_chunks: list[Chunk] = []
        chunk_index = 0

        for symbol_name, body in sections:
            if not body.strip():
                continue

            # Collect all sentences from paragraphs within this section
            paragraphs = re.split(r"\n{2,}", body)
            all_sentences: list[str] = []
            for para in paragraphs:
                para = para.strip()
                if para:
                    all_sentences.extend(_split_sentences(para))

            if not all_sentences:
                continue

            new_chunks = _pack_sentences_into_chunks(
                sentences=all_sentences,
                symbol_name=symbol_name,
                file_path=file_path,
                file_type=doc.file_type,
                content_hash=content_hash,
                start_index=chunk_index,
                max_tokens=self._max_tokens,
                overlap_ratio=self._overlap_ratio,
            )
            all_chunks.extend(new_chunks)
            chunk_index += len(new_chunks)

        if not all_chunks:
            # Fallback: emit the whole text as a single chunk
            all_chunks = [
                build_chunk(
                    text=text,
                    file_path=file_path,
                    file_type=doc.file_type,
                    content_hash=content_hash,
                    chunk_index=0,
                    line_start=1,
                    line_end=doc.line_count,
                    symbol_name=None,
                    chunk_kind=ChunkKind.PROSE,
                )
            ]

        return all_chunks
