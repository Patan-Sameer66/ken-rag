"""Tests for KenError hierarchy (Task 1.2)."""
import pytest
from ken_rag.domain.errors import (
    KenError,
    OllamaUnavailableError,
    ModelNotPulledError,
    EmbedderMismatchError,
    DimensionMismatchError,
    UnsupportedFileTypeError,
)


def test_ken_error_is_exception():
    err = KenError("base error", hint="try this")
    assert isinstance(err, Exception)
    assert str(err) == "base error"
    assert err.hint == "try this"


def test_ken_error_default_hint():
    err = KenError("oops")
    assert err.hint == ""


def test_ollama_unavailable_error_subclasses_ken_error():
    err = OllamaUnavailableError("Ollama is down", hint="run ollama serve")
    assert isinstance(err, KenError)
    assert err.hint == "run ollama serve"


def test_model_not_pulled_error_subclasses_ken_error():
    err = ModelNotPulledError("model missing", hint="ollama pull qwen2.5:3b")
    assert isinstance(err, KenError)
    assert "ollama pull" in err.hint


def test_embedder_mismatch_error_subclasses_ken_error():
    err = EmbedderMismatchError("mismatch", hint="re-add files")
    assert isinstance(err, KenError)
    assert err.hint == "re-add files"


def test_dimension_mismatch_error_subclasses_ken_error():
    err = DimensionMismatchError("dim wrong", hint="expected 768")
    assert isinstance(err, KenError)
    assert "768" in err.hint


def test_unsupported_file_type_error_subclasses_ken_error():
    err = UnsupportedFileTypeError("bad type", hint="use .md or .txt")
    assert isinstance(err, KenError)
    assert err.hint != ""


def test_all_subclasses_carry_hint():
    """Every subclass must accept a hint kwarg."""
    subclasses = [
        OllamaUnavailableError,
        ModelNotPulledError,
        EmbedderMismatchError,
        DimensionMismatchError,
        UnsupportedFileTypeError,
    ]
    for cls in subclasses:
        err = cls("msg", hint="some hint")
        assert hasattr(err, "hint"), f"{cls.__name__} is missing .hint"
        assert isinstance(err, KenError)


def test_errors_can_be_raised_and_caught():
    with pytest.raises(KenError):
        raise OllamaUnavailableError("down", hint="fix it")
