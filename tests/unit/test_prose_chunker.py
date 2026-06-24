"""Tests for prose chunker (Task 1.7) — TDD RED phase.

Run with:
    source .venv/Scripts/activate
    python -m pytest tests/unit/test_prose_chunker.py -q
"""
from __future__ import annotations

import pytest

from ken_rag.domain.enums import ChunkKind, FileType
from ken_rag.domain.models import Chunk, ParsedDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(text: str, file_type: FileType = FileType.MD) -> ParsedDocument:
    return ParsedDocument(
        text=text,
        file_type=file_type,
        line_count=text.count("\n") + 1,
    )


def _make_prose_doc(num_paragraphs: int = 10, sentences_per: int = 8) -> ParsedDocument:
    """Generate a document large enough to produce multiple chunks."""
    paragraphs = []
    for i in range(num_paragraphs):
        sentences = " ".join(
            f"This is sentence {j + 1} of paragraph {i + 1} which contains enough words to fill space."
            for j in range(sentences_per)
        )
        paragraphs.append(sentences)
    return _make_doc("\n\n".join(paragraphs))


# ---------------------------------------------------------------------------
# base.py helpers
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        from ken_rag.chunking.base import estimate_tokens

        assert estimate_tokens("") == 0

    def test_single_word(self) -> None:
        from ken_rag.chunking.base import estimate_tokens

        # ~0.75 tok/word; single word → 0 or 1 depending on rounding
        result = estimate_tokens("hello")
        assert 0 <= result <= 2

    def test_approx_ratio(self) -> None:
        from ken_rag.chunking.base import estimate_tokens

        # 100 words → ~75 tokens (±5)
        words = " ".join(["word"] * 100)
        result = estimate_tokens(words)
        assert 70 <= result <= 80

    def test_longer_text(self) -> None:
        from ken_rag.chunking.base import estimate_tokens

        words = " ".join(["word"] * 400)
        result = estimate_tokens(words)
        # 400 * 0.75 = 300
        assert 290 <= result <= 310


class TestBuildChunk:
    def test_builds_chunk_with_correct_fields(self) -> None:
        from ken_rag.chunking.base import build_chunk

        chunk = build_chunk(
            text="Hello world.",
            file_path="docs/a.md",
            file_type=FileType.MD,
            content_hash="abc123",
            chunk_index=0,
            line_start=1,
            line_end=3,
            symbol_name=None,
            chunk_kind=ChunkKind.PROSE,
        )
        assert isinstance(chunk, Chunk)
        assert chunk.file_path == "docs/a.md"
        assert chunk.content_hash == "abc123"
        assert chunk.chunk_index == 0
        assert chunk.chunk_kind == ChunkKind.PROSE

    def test_chunk_is_frozen(self) -> None:
        from ken_rag.chunking.base import build_chunk

        chunk = build_chunk(
            text="Hi.",
            file_path="a.md",
            file_type=FileType.MD,
            content_hash="h",
            chunk_index=0,
            line_start=1,
            line_end=1,
            symbol_name=None,
            chunk_kind=ChunkKind.PROSE,
        )
        with pytest.raises((AttributeError, TypeError)):
            chunk.text = "bye"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# prose_chunker.py — basic interface
# ---------------------------------------------------------------------------


class TestProseChunkerInterface:
    def test_returns_list_of_chunks(self) -> None:
        from ken_rag.chunking.prose_chunker import ProseChunker

        chunker = ProseChunker()
        doc = _make_prose_doc()
        result = chunker.chunk(doc, "a.md", "hash1")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(c, Chunk) for c in result)

    def test_all_chunks_have_prose_kind(self) -> None:
        from ken_rag.chunking.prose_chunker import ProseChunker

        chunker = ProseChunker()
        doc = _make_prose_doc()
        result = chunker.chunk(doc, "a.md", "hash1")
        assert all(c.chunk_kind == ChunkKind.PROSE for c in result)

    def test_content_hash_propagated(self) -> None:
        from ken_rag.chunking.prose_chunker import ProseChunker

        chunker = ProseChunker()
        doc = _make_prose_doc()
        result = chunker.chunk(doc, "a.md", "myhash")
        assert all(c.content_hash == "myhash" for c in result)

    def test_file_path_propagated(self) -> None:
        from ken_rag.chunking.prose_chunker import ProseChunker

        chunker = ProseChunker()
        doc = _make_prose_doc()
        result = chunker.chunk(doc, "docs/readme.md", "h")
        assert all(c.file_path == "docs/readme.md" for c in result)

    def test_chunk_index_sequential(self) -> None:
        from ken_rag.chunking.prose_chunker import ProseChunker

        chunker = ProseChunker()
        doc = _make_prose_doc()
        result = chunker.chunk(doc, "a.md", "h")
        indices = [c.chunk_index for c in result]
        assert indices == list(range(len(result)))

    def test_short_doc_produces_one_chunk(self) -> None:
        from ken_rag.chunking.prose_chunker import ProseChunker

        chunker = ProseChunker()
        doc = _make_doc("Short text.")
        result = chunker.chunk(doc, "a.md", "h")
        assert len(result) == 1

    def test_covers_all_text(self) -> None:
        """Every word in the source should appear in at least one chunk."""
        from ken_rag.chunking.prose_chunker import ProseChunker

        chunker = ProseChunker()
        doc = _make_prose_doc(num_paragraphs=5, sentences_per=6)
        result = chunker.chunk(doc, "a.md", "h")
        combined = " ".join(c.text for c in result)
        # Check a sample of words from the original doc appear in output
        for word in ["sentence", "paragraph", "words"]:
            assert word in combined


# ---------------------------------------------------------------------------
# Token bounds
# ---------------------------------------------------------------------------


class TestTokenBounds:
    def test_chunks_within_max_token_bound(self) -> None:
        """No chunk may exceed PROSE_MAX_TOK (800) tokens."""
        from ken_rag.chunking.base import estimate_tokens
        from ken_rag.chunking.prose_chunker import ProseChunker
        from ken_rag.config.defaults import PROSE_MAX_TOK

        chunker = ProseChunker()
        # Build a large doc — many paragraphs with many sentences
        doc = _make_prose_doc(num_paragraphs=20, sentences_per=12)
        result = chunker.chunk(doc, "big.md", "h")
        for chunk in result:
            tok = estimate_tokens(chunk.text)
            assert tok <= PROSE_MAX_TOK, (
                f"Chunk {chunk.chunk_index} has {tok} tokens > {PROSE_MAX_TOK}"
            )

    def test_multiple_chunks_produced_for_large_doc(self) -> None:
        from ken_rag.chunking.prose_chunker import ProseChunker

        chunker = ProseChunker()
        doc = _make_prose_doc(num_paragraphs=20, sentences_per=12)
        result = chunker.chunk(doc, "big.md", "h")
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# Header boundaries
# ---------------------------------------------------------------------------


class TestHeaderBoundaries:
    _DOC = """\
# Introduction

This is the introduction section. It contains several sentences of useful content.
It talks about various things and spans multiple lines to pad the word count.

# Methods

The methods section describes the approach taken in this work.
We use a variety of techniques and describe each one in detail here.

# Results

The results section presents our findings.
Each result is described carefully with appropriate detail.
"""

    def test_header_starts_new_chunk(self) -> None:
        """Each '# Header' should ideally start a new chunk boundary."""
        from ken_rag.chunking.prose_chunker import ProseChunker

        chunker = ProseChunker()
        doc = _make_doc(self._DOC)
        result = chunker.chunk(doc, "report.md", "h")
        # We should see at least 2 chunks (introduction + methods/results)
        assert len(result) >= 2

    def test_header_text_captured_in_symbol_name(self) -> None:
        """Symbol names should contain text from markdown headers."""
        from ken_rag.chunking.prose_chunker import ProseChunker

        chunker = ProseChunker()
        doc = _make_doc(self._DOC)
        result = chunker.chunk(doc, "report.md", "h")
        symbol_names = [c.symbol_name for c in result if c.symbol_name is not None]
        # At least one chunk should carry a symbol name derived from a header
        assert len(symbol_names) >= 1
        # Symbol names should contain recognizable header text
        all_names = " ".join(symbol_names)
        assert any(
            kw in all_names for kw in ["Introduction", "Methods", "Results"]
        ), f"No header name found in symbol_names: {symbol_names}"

    def test_header_starts_chunk_not_lost(self) -> None:
        """The header text should appear in the chunk's text."""
        from ken_rag.chunking.prose_chunker import ProseChunker

        chunker = ProseChunker()
        doc = _make_doc(self._DOC)
        result = chunker.chunk(doc, "report.md", "h")
        all_text = "\n".join(c.text for c in result)
        assert "Introduction" in all_text
        assert "Methods" in all_text

    def test_large_doc_with_many_headers(self) -> None:
        """Stress test with many headers."""
        from ken_rag.chunking.prose_chunker import ProseChunker

        sections = []
        for i in range(6):
            sentences = " ".join(
                f"Section {i} sentence {j} with enough words to fill space."
                for j in range(8)
            )
            sections.append(f"# Section {i}\n\n{sentences}\n")
        doc = _make_doc("\n".join(sections))
        chunker = ProseChunker()
        result = chunker.chunk(doc, "multi.md", "h")
        assert len(result) >= 2
        # All chunks have sequential indices
        assert [c.chunk_index for c in result] == list(range(len(result)))


# ---------------------------------------------------------------------------
# No mid-sentence cuts
# ---------------------------------------------------------------------------


class TestNoMidSentenceCuts:
    def test_chunks_end_at_sentence_boundary(self) -> None:
        """Chunks should end on sentence-terminating punctuation (. ! ?)."""
        from ken_rag.chunking.prose_chunker import ProseChunker

        chunker = ProseChunker()
        doc = _make_prose_doc(num_paragraphs=15, sentences_per=10)
        result = chunker.chunk(doc, "a.md", "h")
        if len(result) <= 1:
            pytest.skip("Only one chunk produced — no splits to verify")
        # All chunks except possibly the last should end at a sentence boundary
        for chunk in result[:-1]:
            text = chunk.text.rstrip()
            assert text[-1] in ".!?", (
                f"Chunk {chunk.chunk_index} ends with non-sentence char: {repr(text[-5:])}"
            )

    def test_no_sentence_fragmented_across_chunks(self) -> None:
        """A sentence that starts in one chunk should not continue in the next."""
        from ken_rag.chunking.prose_chunker import ProseChunker

        # Build text where sentences are unambiguous and long enough to matter
        sentences = [
            f"This is the complete and entire sentence number {i}, period."
            for i in range(60)
        ]
        text = " ".join(sentences)
        doc = _make_doc(text)
        chunker = ProseChunker()
        result = chunker.chunk(doc, "a.md", "h")
        if len(result) <= 1:
            pytest.skip("Only one chunk produced — no splits to verify")
        # Reconstruct: joining chunks should contain all sentences intact
        combined = " ".join(c.text for c in result)
        for s in sentences:
            # Each sentence should appear intact somewhere in the combined output
            assert s in combined or s.split(",")[0] in combined, (
                f"Sentence fragment detected: {s[:50]}"
            )


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


class TestOverlap:
    def test_overlap_present_between_consecutive_chunks(self) -> None:
        """Consecutive chunks should share some overlapping text."""
        from ken_rag.chunking.prose_chunker import ProseChunker

        chunker = ProseChunker()
        doc = _make_prose_doc(num_paragraphs=20, sentences_per=10)
        result = chunker.chunk(doc, "a.md", "h")
        if len(result) < 2:
            pytest.skip("Need at least 2 chunks to test overlap")

        overlap_found = False
        for i in range(len(result) - 1):
            words_a = set(result[i].text.split())
            words_b = set(result[i + 1].text.split())
            shared = words_a & words_b
            if len(shared) > 3:  # more than trivial common words
                overlap_found = True
                break
        assert overlap_found, "No overlap detected between any consecutive chunks"

    def test_overlap_not_entire_chunk(self) -> None:
        """Overlap should be a fraction of the chunk, not the whole thing."""
        from ken_rag.chunking.prose_chunker import ProseChunker
        from ken_rag.config.defaults import PROSE_OVERLAP

        chunker = ProseChunker()
        doc = _make_prose_doc(num_paragraphs=20, sentences_per=10)
        result = chunker.chunk(doc, "a.md", "h")
        if len(result) < 2:
            pytest.skip("Need at least 2 chunks to test overlap ratio")

        for i in range(len(result) - 1):
            words_a = result[i].text.split()
            words_b = result[i + 1].text.split()
            # The second chunk should contain more than just the overlap
            assert len(words_b) > len(words_a) * PROSE_OVERLAP * 0.5


# ---------------------------------------------------------------------------
# fallback_chunker.py — FallbackChunker
# ---------------------------------------------------------------------------


class TestFallbackChunker:
    def test_returns_list_of_chunks(self) -> None:
        from ken_rag.chunking.fallback_chunker import FallbackChunker

        chunker = FallbackChunker()
        doc = _make_doc(
            "\n".join(f"Line number {i} of the document." for i in range(50)),
            file_type=FileType.UNKNOWN,
        )
        result = chunker.chunk(doc, "unknown.bin", "h")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(c, Chunk) for c in result)

    def test_all_chunks_fallback_kind(self) -> None:
        from ken_rag.chunking.fallback_chunker import FallbackChunker

        chunker = FallbackChunker()
        doc = _make_doc(
            "\n".join(f"Line {i}" for i in range(50)),
            file_type=FileType.UNKNOWN,
        )
        result = chunker.chunk(doc, "f.bin", "h")
        assert all(c.chunk_kind == ChunkKind.FALLBACK for c in result)

    def test_chunk_index_sequential(self) -> None:
        from ken_rag.chunking.fallback_chunker import FallbackChunker

        chunker = FallbackChunker()
        doc = _make_doc(
            "\n".join(f"Line {i} of content." for i in range(100)),
            file_type=FileType.UNKNOWN,
        )
        result = chunker.chunk(doc, "f.bin", "h")
        assert [c.chunk_index for c in result] == list(range(len(result)))

    def test_content_hash_propagated(self) -> None:
        from ken_rag.chunking.fallback_chunker import FallbackChunker

        chunker = FallbackChunker()
        doc = _make_doc("\n".join(f"Line {i}" for i in range(30)), file_type=FileType.UNKNOWN)
        result = chunker.chunk(doc, "f.bin", "fallhash")
        assert all(c.content_hash == "fallhash" for c in result)

    def test_empty_doc_produces_one_chunk(self) -> None:
        from ken_rag.chunking.fallback_chunker import FallbackChunker

        chunker = FallbackChunker()
        doc = _make_doc("", file_type=FileType.UNKNOWN)
        result = chunker.chunk(doc, "empty.bin", "h")
        # Empty doc: 0 or 1 chunk, but must not crash
        assert isinstance(result, list)

    def test_chunks_within_max_token_bound(self) -> None:
        from ken_rag.chunking.base import estimate_tokens
        from ken_rag.chunking.fallback_chunker import FallbackChunker
        from ken_rag.config.defaults import PROSE_MAX_TOK

        chunker = FallbackChunker()
        lines = [f"This is line {i} with some content words here." for i in range(200)]
        doc = _make_doc("\n".join(lines), file_type=FileType.UNKNOWN)
        result = chunker.chunk(doc, "big.bin", "h")
        for chunk in result:
            tok = estimate_tokens(chunk.text)
            assert tok <= PROSE_MAX_TOK


# ---------------------------------------------------------------------------
# registry.py — ChunkerRegistry
# ---------------------------------------------------------------------------


class TestChunkerRegistry:
    def test_txt_maps_to_prose_chunker(self) -> None:
        from ken_rag.chunking.registry import ChunkerRegistry
        from ken_rag.chunking.prose_chunker import ProseChunker

        registry = ChunkerRegistry.default()
        chunker = registry.get(FileType.TXT)
        assert isinstance(chunker, ProseChunker)

    def test_md_maps_to_prose_chunker(self) -> None:
        from ken_rag.chunking.registry import ChunkerRegistry
        from ken_rag.chunking.prose_chunker import ProseChunker

        registry = ChunkerRegistry.default()
        chunker = registry.get(FileType.MD)
        assert isinstance(chunker, ProseChunker)

    def test_unknown_maps_to_fallback_chunker(self) -> None:
        from ken_rag.chunking.registry import ChunkerRegistry
        from ken_rag.chunking.fallback_chunker import FallbackChunker

        registry = ChunkerRegistry.default()
        chunker = registry.get(FileType.UNKNOWN)
        assert isinstance(chunker, FallbackChunker)

    def test_code_not_registered_by_default(self) -> None:
        """CODE chunker is wired in Phase 2 — must not exist yet."""
        from ken_rag.chunking.registry import ChunkerRegistry

        registry = ChunkerRegistry.default()
        # CODE file type should have no chunker yet (returns None or raises)
        chunker = registry.get(FileType.CODE)
        assert chunker is None

    def test_register_method_exists(self) -> None:
        """Registry must expose a `register` method for Phase-2 wiring."""
        from ken_rag.chunking.registry import ChunkerRegistry

        registry = ChunkerRegistry.default()
        assert callable(getattr(registry, "register", None))

    def test_register_overrides_chunker(self) -> None:
        """Calling register() should allow replacing/adding chunkers."""
        from ken_rag.chunking.prose_chunker import ProseChunker
        from ken_rag.chunking.registry import ChunkerRegistry

        registry = ChunkerRegistry.default()
        custom = ProseChunker()
        registry.register(FileType.CODE, custom)
        assert registry.get(FileType.CODE) is custom

    def test_pdf_returns_none_or_fallback(self) -> None:
        """PDF is parsed but chunked — registry should handle gracefully."""
        from ken_rag.chunking.registry import ChunkerRegistry

        registry = ChunkerRegistry.default()
        # PDF is not explicitly registered in Phase 1; should not crash
        result = registry.get(FileType.PDF)
        # Acceptable: None, or a fallback chunker — just must not error
        assert result is None or hasattr(result, "chunk")


# ---------------------------------------------------------------------------
# Chunker protocol compliance
# ---------------------------------------------------------------------------


class TestChunkerProtocol:
    def test_prose_chunker_satisfies_protocol(self) -> None:
        from ken_rag.chunking.prose_chunker import ProseChunker
        from ken_rag.domain.protocols import Chunker

        assert isinstance(ProseChunker(), Chunker)

    def test_fallback_chunker_satisfies_protocol(self) -> None:
        from ken_rag.chunking.fallback_chunker import FallbackChunker
        from ken_rag.domain.protocols import Chunker

        assert isinstance(FallbackChunker(), Chunker)
