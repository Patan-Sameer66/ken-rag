"""Low-level chunking helpers shared across all chunker implementations.

Provides:
- estimate_tokens(text): word-count–based approximation at ~0.75 tok/word.
- overlap_window(sentences, overlap_ratio): return trailing sentences that
  form the overlap prefix for the next chunk.
- build_chunk(...): stamp a Chunk with all required metadata fields.
"""
from __future__ import annotations

from ken_rag.domain.enums import ChunkKind, FileType
from ken_rag.domain.models import Chunk

# Empirical ratio: ~0.75 tokens per whitespace-split word for English prose.
_TOKENS_PER_WORD: float = 0.75


def estimate_tokens(text: str) -> int:
    """Estimate the token count of *text* using a word-count heuristic.

    Uses the empirical ratio of ~0.75 tokens per whitespace-delimited word,
    which is accurate to ±10% for typical English/markdown prose.
    """
    if not text:
        return 0
    words = text.split()
    return int(len(words) * _TOKENS_PER_WORD)


def overlap_window(sentences: list[str], overlap_ratio: float) -> list[str]:
    """Return the trailing sentences that form the overlap prefix.

    The overlap targets *overlap_ratio* × total token count of *sentences*.
    Sentences are taken from the end of the list until the token budget is met
    or exceeded; at most all-but-one sentence is returned to prevent infinite
    loops when a single sentence is larger than the budget.
    """
    if not sentences or overlap_ratio <= 0.0:
        return []

    total_tokens = estimate_tokens(" ".join(sentences))
    budget = int(total_tokens * overlap_ratio)
    if budget <= 0:
        return []

    accumulated = 0
    window: list[str] = []
    for sentence in reversed(sentences):
        tok = estimate_tokens(sentence)
        if accumulated + tok > budget and window:
            # Adding this sentence would exceed budget; stop if we already have some
            break
        window.insert(0, sentence)
        accumulated += tok
        if accumulated >= budget:
            break

    # Safety: never return the entire chunk as overlap
    if len(window) >= len(sentences):
        window = window[1:] if len(window) > 1 else []

    return window


def build_chunk(
    *,
    text: str,
    file_path: str,
    file_type: FileType,
    content_hash: str,
    chunk_index: int,
    line_start: int,
    line_end: int,
    symbol_name: str | None,
    chunk_kind: ChunkKind,
) -> Chunk:
    """Construct a :class:`~ken_rag.domain.models.Chunk` with all fields stamped.

    This is the single place that assembles the frozen Chunk dataclass so that
    callers never have to remember the full field list.
    """
    return Chunk(
        text=text,
        file_path=file_path,
        file_type=str(file_type.value) if isinstance(file_type, FileType) else str(file_type),
        chunk_index=chunk_index,
        content_hash=content_hash,
        line_start=line_start,
        line_end=line_end,
        symbol_name=symbol_name,
        chunk_kind=str(chunk_kind.value) if isinstance(chunk_kind, ChunkKind) else str(chunk_kind),
    )
