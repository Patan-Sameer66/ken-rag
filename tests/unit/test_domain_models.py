"""Tests for domain models (Task 1.1)."""
import pytest
from ken_rag.domain.models import (
    Chunk,
    EmbeddedChunk,
    RetrievedChunk,
    Citation,
    Answer,
    FileRecord,
    FileChange,
    CandidateFile,
    ParsedDocument,
)
from ken_rag.domain.enums import FileType, ChunkKind, FileChangeState, ModelTier


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

def _make_chunk(**overrides) -> Chunk:
    defaults = dict(
        text="hello world",
        file_path="docs/a.md",
        file_type="md",
        chunk_index=0,
        content_hash="deadbeef",
        line_start=1,
        line_end=5,
        symbol_name=None,
        chunk_kind="prose",
    )
    defaults.update(overrides)
    return Chunk(**defaults)


def test_chunk_is_frozen():
    c = _make_chunk()
    with pytest.raises((AttributeError, TypeError)):
        c.text = "y"  # type: ignore[misc]


def test_chunk_fields():
    c = _make_chunk()
    assert c.text == "hello world"
    assert c.file_path == "docs/a.md"
    assert c.chunk_index == 0
    assert c.symbol_name is None


# ---------------------------------------------------------------------------
# EmbeddedChunk
# ---------------------------------------------------------------------------

def test_embedded_chunk_vector_len():
    c = _make_chunk()
    ec = EmbeddedChunk(chunk=c, vector=tuple([0.0] * 768))
    assert len(ec.vector) == 768


def test_embedded_chunk_is_frozen():
    c = _make_chunk()
    ec = EmbeddedChunk(chunk=c, vector=tuple([0.0] * 768))
    with pytest.raises((AttributeError, TypeError)):
        ec.vector = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RetrievedChunk
# ---------------------------------------------------------------------------

def test_retrieved_chunk():
    c = _make_chunk()
    rc = RetrievedChunk(chunk=c, score=0.92)
    assert rc.score == 0.92
    assert rc.chunk is c


def test_retrieved_chunk_is_frozen():
    c = _make_chunk()
    rc = RetrievedChunk(chunk=c, score=0.5)
    with pytest.raises((AttributeError, TypeError)):
        rc.score = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------

def test_citation_fields():
    cit = Citation(file_path="src/foo.py", line_start=10, line_end=20, symbol_name="parse_config")
    assert cit.file_path == "src/foo.py"
    assert cit.symbol_name == "parse_config"


def test_citation_is_frozen():
    cit = Citation(file_path="x.py", line_start=1, line_end=2, symbol_name=None)
    with pytest.raises((AttributeError, TypeError)):
        cit.file_path = "y.py"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

def test_answer_with_citations():
    cit = Citation(file_path="a.py", line_start=1, line_end=5, symbol_name=None)
    ans = Answer(text="The answer.", citations=(cit,))
    assert ans.text == "The answer."
    assert len(ans.citations) == 1


def test_answer_is_frozen():
    ans = Answer(text="hi", citations=())
    with pytest.raises((AttributeError, TypeError)):
        ans.text = "bye"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FileRecord
# ---------------------------------------------------------------------------

def test_file_record_fields():
    import datetime
    fr = FileRecord(
        file_path="src/main.py",
        content_hash="abc123",
        file_type="code",
        chunk_count=7,
        indexed_at=datetime.datetime(2026, 6, 24),
        git_commit=None,
    )
    assert fr.chunk_count == 7
    assert fr.git_commit is None


# ---------------------------------------------------------------------------
# FileChange
# ---------------------------------------------------------------------------

def test_file_change():
    fc = FileChange(file_path="docs/x.md", state=FileChangeState.ADDED)
    assert fc.state == FileChangeState.ADDED


# ---------------------------------------------------------------------------
# CandidateFile
# ---------------------------------------------------------------------------

def test_candidate_file():
    cf = CandidateFile(path="docs/y.md", content_hash="h1", file_type=FileType.MD)
    assert cf.file_type == FileType.MD


# ---------------------------------------------------------------------------
# ParsedDocument
# ---------------------------------------------------------------------------

def test_parsed_document():
    pd = ParsedDocument(text="some text\nline2", file_type=FileType.MD, line_count=2)
    assert pd.line_count == 2


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

def test_enums():
    assert FileType.MD == "md"
    assert ChunkKind.PROSE == "prose"
    assert FileChangeState.MODIFIED == "modified"
    assert ModelTier.HIGH == "qwen2.5:3b"
    assert ModelTier.MID == "qwen2.5:1.5b"
    assert ModelTier.LOW == "llama3.2:1b"
