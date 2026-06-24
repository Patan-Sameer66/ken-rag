"""Tests for config defaults, paths, settings, and loader (Task 1.3)."""
import json

import pytest

from ken_rag.config.defaults import (
    DEFAULT_K,
    NUM_CTX,
    EMBED_DIM,
    EMBEDDER_NAME,
    DEFAULT_LLM,
    BATCH_SIZE,
    PROSE_MIN_TOK,
    PROSE_MAX_TOK,
    PROSE_OVERLAP,
    OLLAMA_URL,
    TIMEOUT_S,
)
from ken_rag.config.paths import data_dir, db_path, config_path
from ken_rag.config.settings import Settings
from ken_rag.config.loader import load_settings
from ken_rag.domain.errors import KenError


# ---------------------------------------------------------------------------
# Defaults constants
# ---------------------------------------------------------------------------

def test_default_k():
    assert DEFAULT_K == 5


def test_num_ctx():
    assert NUM_CTX == 8192


def test_embed_dim():
    assert EMBED_DIM == 768


def test_embedder_name():
    assert EMBEDDER_NAME == "nomic-embed-text"


def test_default_llm():
    assert DEFAULT_LLM == "qwen2.5:3b"


def test_batch_size():
    assert BATCH_SIZE == 64


def test_prose_min_tok():
    assert PROSE_MIN_TOK == 300


def test_prose_max_tok():
    assert PROSE_MAX_TOK == 800


def test_prose_overlap():
    assert abs(PROSE_OVERLAP - 0.12) < 1e-9


def test_ollama_url():
    assert OLLAMA_URL == "http://localhost:11434"


def test_timeout_s():
    assert TIMEOUT_S == 120


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def test_data_dir(tmp_path):
    assert data_dir(tmp_path) == tmp_path / ".ken"


def test_db_path(tmp_path):
    assert db_path(tmp_path) == tmp_path / ".ken" / "lancedb"


def test_config_path(tmp_path):
    assert config_path(tmp_path) == tmp_path / ".ken" / "config.json"


# ---------------------------------------------------------------------------
# Settings.default
# ---------------------------------------------------------------------------

def test_settings_default(tmp_path):
    s = Settings.default(tmp_path)
    assert s.k == DEFAULT_K
    assert s.num_ctx == NUM_CTX
    assert s.embed_dim == EMBED_DIM
    assert s.embedder_name == EMBEDDER_NAME
    assert s.llm_name == DEFAULT_LLM
    assert s.batch_size == BATCH_SIZE
    assert s.db_path == db_path(tmp_path)
    assert s.ollama_url == OLLAMA_URL


def test_settings_is_frozen(tmp_path):
    s = Settings.default(tmp_path)
    with pytest.raises((AttributeError, TypeError)):
        s.k = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Loader — defaults path
# ---------------------------------------------------------------------------

def test_load_settings_defaults(tmp_path):
    s = load_settings(tmp_path)
    assert s.k == DEFAULT_K
    assert s.embedder_name == EMBEDDER_NAME


def test_load_settings_from_config_file(tmp_path):
    cfg = tmp_path / ".ken" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"k": 10, "llm_name": "qwen2.5:1.5b"}))
    s = load_settings(tmp_path)
    assert s.k == 10
    assert s.llm_name == "qwen2.5:1.5b"
    # unset fields still get defaults
    assert s.embedder_name == EMBEDDER_NAME


def test_load_settings_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("KEN_K", "7")
    monkeypatch.setenv("KEN_LLM_NAME", "llama3.2:1b")
    s = load_settings(tmp_path)
    assert s.k == 7
    assert s.llm_name == "llama3.2:1b"


def test_load_settings_explicit_override(tmp_path):
    s = load_settings(tmp_path, overrides={"k": 3})
    assert s.k == 3


def test_load_settings_bad_type_raises_ken_error(tmp_path):
    cfg = tmp_path / ".ken" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"k": "not-an-int"}))
    with pytest.raises(KenError):
        load_settings(tmp_path)
