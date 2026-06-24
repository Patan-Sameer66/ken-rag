"""Tests for ken_rag.generation.prompt — TDD (write first, implement after)."""
from __future__ import annotations

from ken_rag.domain.models import Chunk, RetrievedChunk
from ken_rag.generation.prompt import PromptBuilder


def _make_chunk(
    file_path: str = "docs/guide.md",
    line_start: int = 1,
    line_end: int = 10,
    text: str = "Some helpful content.",
    chunk_index: int = 0,
    symbol_name: str | None = None,
) -> Chunk:
    return Chunk(
        text=text,
        file_path=file_path,
        file_type="md",
        chunk_index=chunk_index,
        content_hash="abc123",
        line_start=line_start,
        line_end=line_end,
        symbol_name=symbol_name,
        chunk_kind="prose",
    )


def _make_retrieved(chunk: Chunk, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=score)


class TestPromptBuilderBuild:
    """PromptBuilder.build(question, retrieved) → prompt string."""

    def test_returns_string(self) -> None:
        chunk = _make_chunk()
        prompt = PromptBuilder.build("What is this?", [_make_retrieved(chunk)])
        assert isinstance(prompt, str)

    def test_contains_system_instruction(self) -> None:
        chunk = _make_chunk()
        prompt = PromptBuilder.build("What is this?", [_make_retrieved(chunk)])
        lower = prompt.lower()
        # Must instruct to answer ONLY from sources
        assert "only" in lower or "solely" in lower or "sources" in lower

    def test_system_instruction_mentions_cite(self) -> None:
        chunk = _make_chunk()
        prompt = PromptBuilder.build("What is this?", [_make_retrieved(chunk)])
        lower = prompt.lower()
        assert "cit" in lower  # cite / citation / citations

    def test_numbered_source_format(self) -> None:
        chunk = _make_chunk(file_path="src/auth.py", line_start=14, line_end=37)
        prompt = PromptBuilder.build("How does auth work?", [_make_retrieved(chunk)])
        # Expect "[1] src/auth.py:14-37" somewhere in the prompt
        assert "[1]" in prompt
        assert "src/auth.py:14-37" in prompt

    def test_source_includes_chunk_text(self) -> None:
        chunk = _make_chunk(text="The decorator validates the bearer token.")
        prompt = PromptBuilder.build("What does auth do?", [_make_retrieved(chunk)])
        assert "The decorator validates the bearer token." in prompt

    def test_multiple_sources_numbered(self) -> None:
        c1 = _make_chunk(file_path="a.md", line_start=1, line_end=5, chunk_index=0)
        c2 = _make_chunk(file_path="b.md", line_start=10, line_end=20, chunk_index=1)
        prompt = PromptBuilder.build("q?", [_make_retrieved(c1), _make_retrieved(c2)])
        assert "[1]" in prompt
        assert "a.md:1-5" in prompt
        assert "[2]" in prompt
        assert "b.md:10-20" in prompt

    def test_question_appears_last(self) -> None:
        question = "Where is the login function defined?"
        chunk = _make_chunk()
        prompt = PromptBuilder.build(question, [_make_retrieved(chunk)])
        assert prompt.rstrip().endswith(question) or prompt.rstrip().endswith(question + "?") or question in prompt
        # The question must appear AFTER the sources section
        source_pos = prompt.index("[1]")
        question_pos = prompt.index(question)
        assert question_pos > source_pos

    def test_empty_retrieved_still_returns_prompt(self) -> None:
        prompt = PromptBuilder.build("What is 2+2?", [])
        assert isinstance(prompt, str)
        assert "What is 2+2?" in prompt

    def test_source_line_range_format_colon_dash(self) -> None:
        """Source format must use 'file_path:line_start-line_end' (colon then dash)."""
        chunk = _make_chunk(file_path="lib/utils.py", line_start=5, line_end=22)
        prompt = PromptBuilder.build("q", [_make_retrieved(chunk)])
        assert "lib/utils.py:5-22" in prompt

    def test_system_block_precedes_sources(self) -> None:
        """System instruction comes before any numbered source."""
        chunk = _make_chunk()
        prompt = PromptBuilder.build("q", [_make_retrieved(chunk)])
        source_pos = prompt.index("[1]")
        # Some text about 'sources' or 'only' must appear before the first [1]
        preamble = prompt[:source_pos].lower()
        assert "source" in preamble or "only" in preamble or "cit" in preamble
