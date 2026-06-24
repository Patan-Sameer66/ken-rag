"""Tests for OllamaClient — Task 1.4.

Uses respx to mock httpx calls. Tests cover:
- embed() happy path returns vectors
- embed() ConnectError → OllamaUnavailableError with 'ollama serve' in hint
- embed() HTTP 404 → ModelNotPulledError with 'ollama pull' in hint
- generate_stream() yields tokens from JSONL stream
- generate_stream() ConnectError → OllamaUnavailableError
- generate_stream() HTTP 404 → ModelNotPulledError
"""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from ken_rag.domain.errors import ModelNotPulledError, OllamaUnavailableError
from ken_rag.llm.ollama_client import OllamaClient

BASE = "http://localhost:11434"


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------


@respx.mock
def test_embed_returns_vectors():
    """Happy path: POST /api/embed returns embeddings list."""
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    respx.post(f"{BASE}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": vectors})
    )
    client = OllamaClient(BASE)
    result = client.embed("nomic-embed-text", ["hello", "world"])
    assert result == vectors


@respx.mock
def test_embed_passes_model_and_inputs():
    """embed() sends the correct JSON body."""
    respx.post(f"{BASE}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.0] * 768]})
    )
    client = OllamaClient(BASE)
    client.embed("nomic-embed-text", ["test input"])

    sent_request = respx.calls.last.request
    body = json.loads(sent_request.content)
    assert body["model"] == "nomic-embed-text"
    assert body["input"] == ["test input"]


def test_embed_connect_error_raises_ollama_unavailable():
    """ConnectError on embed() → OllamaUnavailableError with 'ollama serve' hint."""
    with respx.mock:
        respx.post(f"{BASE}/api/embed").mock(side_effect=httpx.ConnectError("refused"))
        client = OllamaClient(BASE)
        with pytest.raises(OllamaUnavailableError) as exc_info:
            client.embed("nomic-embed-text", ["hello"])
    assert "ollama serve" in exc_info.value.hint


@respx.mock
def test_embed_404_raises_model_not_pulled():
    """HTTP 404 on embed() → ModelNotPulledError with 'ollama pull' in hint."""
    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(404))
    client = OllamaClient(BASE)
    with pytest.raises(ModelNotPulledError) as exc_info:
        client.embed("nomic-embed-text", ["hello"])
    assert "ollama pull" in exc_info.value.hint


@respx.mock
def test_embed_non_404_http_error_raises():
    """Other HTTP errors (e.g. 500) bubble as httpx.HTTPStatusError."""
    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(500))
    client = OllamaClient(BASE)
    with pytest.raises(httpx.HTTPStatusError):
        client.embed("nomic-embed-text", ["hello"])


# ---------------------------------------------------------------------------
# generate_stream()
# ---------------------------------------------------------------------------


def _make_jsonl(*token_response_pairs: dict) -> bytes:
    """Build a JSONL byte stream from a sequence of response dicts."""
    lines = [json.dumps(d).encode() for d in token_response_pairs]
    return b"\n".join(lines)


@respx.mock
def test_generate_stream_yields_tokens():
    """generate_stream() yields the 'response' field from each JSONL line."""
    jsonl = _make_jsonl(
        {"response": "Hello", "done": False},
        {"response": " world", "done": False},
        {"response": "", "done": True},
    )
    respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, content=jsonl)
    )
    client = OllamaClient(BASE)
    tokens = list(client.generate_stream("qwen2.5:3b", "hi", num_ctx=8192))
    assert tokens == ["Hello", " world"]


@respx.mock
def test_generate_stream_sends_num_ctx_option():
    """generate_stream() includes options={'num_ctx': num_ctx} in the request body."""
    jsonl = _make_jsonl({"response": "ok", "done": True})
    respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, content=jsonl)
    )
    client = OllamaClient(BASE)
    list(client.generate_stream("qwen2.5:3b", "test prompt", num_ctx=8192))

    sent_request = respx.calls.last.request
    body = json.loads(sent_request.content)
    assert body["options"]["num_ctx"] == 8192
    assert body["stream"] is True


def test_generate_stream_connect_error_raises_ollama_unavailable():
    """ConnectError on generate_stream() → OllamaUnavailableError."""
    with respx.mock:
        respx.post(f"{BASE}/api/generate").mock(
            side_effect=httpx.ConnectError("refused")
        )
        client = OllamaClient(BASE)
        with pytest.raises(OllamaUnavailableError) as exc_info:
            list(client.generate_stream("qwen2.5:3b", "hi", num_ctx=8192))
    assert "ollama serve" in exc_info.value.hint


@respx.mock
def test_generate_stream_404_raises_model_not_pulled():
    """HTTP 404 on generate_stream() → ModelNotPulledError."""
    respx.post(f"{BASE}/api/generate").mock(return_value=httpx.Response(404))
    client = OllamaClient(BASE)
    with pytest.raises(ModelNotPulledError) as exc_info:
        list(client.generate_stream("qwen2.5:3b", "hi", num_ctx=8192))
    assert "ollama pull" in exc_info.value.hint
