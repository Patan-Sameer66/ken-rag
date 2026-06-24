"""Tests for ken_rag.generation.citation — TDD (write first, implement after)."""
from __future__ import annotations

from ken_rag.domain.models import Chunk, Citation, RetrievedChunk
from ken_rag.generation.citation import build


def _make_chunk(
    file_path: str = "src/app.py",
    line_start: int = 1,
    line_end: int = 10,
    chunk_index: int = 0,
    symbol_name: str | None = None,
) -> Chunk:
    return Chunk(
        text="content",
        file_path=file_path,
        file_type="md",
        chunk_index=chunk_index,
        content_hash="abc",
        line_start=line_start,
        line_end=line_end,
        symbol_name=symbol_name,
        chunk_kind="prose",
    )


def _rc(chunk: Chunk, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=score)


class TestBuildCitations:
    """citation.build(retrieved) -> tuple[Citation, ...]"""

    def test_returns_tuple(self) -> None:
        result = build([_rc(_make_chunk())])
        assert isinstance(result, tuple)

    def test_single_chunk_single_citation(self) -> None:
        chunk = _make_chunk(file_path="a.py", line_start=1, line_end=10)
        result = build([_rc(chunk)])
        assert len(result) == 1
        cit = result[0]
        assert isinstance(cit, Citation)
        assert cit.file_path == "a.py"
        assert cit.line_start == 1
        assert cit.line_end == 10

    def test_symbol_name_propagated(self) -> None:
        chunk = _make_chunk(symbol_name="my_function")
        result = build([_rc(chunk)])
        assert result[0].symbol_name == "my_function"

    def test_no_symbol_name_is_none(self) -> None:
        chunk = _make_chunk(symbol_name=None)
        result = build([_rc(chunk)])
        assert result[0].symbol_name is None

    def test_dedup_same_file_and_line_range(self) -> None:
        """Two chunks with identical file_path + line_start + line_end → one citation."""
        c1 = _make_chunk(file_path="auth.py", line_start=14, line_end=37, chunk_index=0)
        c2 = _make_chunk(file_path="auth.py", line_start=14, line_end=37, chunk_index=1)
        result = build([_rc(c1), _rc(c2)])
        assert len(result) == 1
        assert result[0].file_path == "auth.py"
        assert result[0].line_start == 14
        assert result[0].line_end == 37

    def test_different_files_not_deduped(self) -> None:
        c1 = _make_chunk(file_path="a.py", line_start=1, line_end=10, chunk_index=0)
        c2 = _make_chunk(file_path="b.py", line_start=1, line_end=10, chunk_index=1)
        result = build([_rc(c1), _rc(c2)])
        assert len(result) == 2

    def test_same_file_different_line_range_not_deduped(self) -> None:
        c1 = _make_chunk(file_path="a.py", line_start=1, line_end=10, chunk_index=0)
        c2 = _make_chunk(file_path="a.py", line_start=20, line_end=40, chunk_index=1)
        result = build([_rc(c1), _rc(c2)])
        assert len(result) == 2

    def test_order_preserved_by_first_occurrence(self) -> None:
        """Citations appear in the order they were first encountered."""
        c1 = _make_chunk(file_path="z.py", line_start=1, line_end=5, chunk_index=0)
        c2 = _make_chunk(file_path="a.py", line_start=1, line_end=5, chunk_index=1)
        c3 = _make_chunk(file_path="m.py", line_start=1, line_end=5, chunk_index=2)
        result = build([_rc(c1), _rc(c2), _rc(c3)])
        assert [r.file_path for r in result] == ["z.py", "a.py", "m.py"]

    def test_dedup_preserves_first_occurrence_order(self) -> None:
        """When a duplicate appears later, the first occurrence position is kept."""
        c1 = _make_chunk(file_path="a.py", line_start=1, line_end=5, chunk_index=0)
        c2 = _make_chunk(file_path="b.py", line_start=1, line_end=5, chunk_index=1)
        c3 = _make_chunk(file_path="a.py", line_start=1, line_end=5, chunk_index=2)  # dup of c1
        result = build([_rc(c1), _rc(c2), _rc(c3)])
        assert len(result) == 2
        assert result[0].file_path == "a.py"
        assert result[1].file_path == "b.py"

    def test_empty_returns_empty_tuple(self) -> None:
        result = build([])
        assert result == ()

    def test_triple_duplicate_collapsed(self) -> None:
        chunks = [
            _make_chunk(file_path="lib.py", line_start=5, line_end=15, chunk_index=i)
            for i in range(3)
        ]
        result = build([_rc(c) for c in chunks])
        assert len(result) == 1
