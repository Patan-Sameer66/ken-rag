"""End-to-end CLI tests for ``ken add`` and ``ken ask``.

Uses Typer's CliRunner. The real ``build_context`` is reused with *overrides*
so the CLI runs against fakes (no Ollama daemon, no LanceDB on disk). We
monkeypatch ``ken_rag.cli.app.build_context`` to return that fake-wired context.
"""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ken_rag.cli import app as app_module
from ken_rag.cli.context import build_context
from ken_rag.config.settings import Settings
from ken_rag.domain.errors import OllamaUnavailableError
from tests.fakes.fake_embedder import FakeEmbedder
from tests.fakes.fake_generator import FakeGenerator
from tests.fakes.fake_meta import FakeMetadataStore
from tests.fakes.in_memory_store import InMemoryVectorStore

runner = CliRunner()

_DOC = """# Authentication

The require_auth decorator wraps each route and checks the bearer token
before the handler runs. Expired tokens are rejected with a 401 response.
"""


def _fake_context(tokens=None):
    """Build a real AppContext wired with fakes via build_context overrides."""
    settings = Settings.default(Path.cwd())
    return build_context(
        settings,
        overrides={
            "embedder": FakeEmbedder(model_name="nomic-embed-text"),
            "generator": FakeGenerator(
                tokens=tokens or ["Auth", " uses", " tokens."],
                expected_num_ctx=8192,
            ),
            "vector_store": InMemoryVectorStore(),
            "metadata_store": FakeMetadataStore(),
        },
    )


def test_add_then_ask_end_to_end(tmp_path, monkeypatch):
    (tmp_path / "auth.md").write_text(_DOC, encoding="utf-8")
    ctx = _fake_context()
    monkeypatch.setattr(app_module, "build_context", lambda settings, **kw: ctx)

    add_result = runner.invoke(app_module.app, ["add", str(tmp_path)])
    assert add_result.exit_code == 0, add_result.output
    assert "Indexed" in add_result.output

    ask_result = runner.invoke(app_module.app, ["ask", "how does auth work?"])
    assert ask_result.exit_code == 0, ask_result.output
    assert "Auth uses tokens." in ask_result.output
    # citation block references the indexed file
    assert "auth.md" in ask_result.output


def test_add_single_file(tmp_path, monkeypatch):
    f = tmp_path / "single.md"
    f.write_text(_DOC, encoding="utf-8")
    ctx = _fake_context()
    monkeypatch.setattr(app_module, "build_context", lambda settings, **kw: ctx)

    result = runner.invoke(app_module.app, ["add", str(f)])
    assert result.exit_code == 0, result.output
    assert "Indexed 1 file" in result.output


def test_add_nonexistent_path_errors(monkeypatch):
    ctx = _fake_context()
    monkeypatch.setattr(app_module, "build_context", lambda settings, **kw: ctx)
    result = runner.invoke(app_module.app, ["add", "does/not/exist"])
    assert result.exit_code == 1
    assert "does not exist" in result.output.lower()


def test_ask_ollama_unavailable_shows_hint_not_traceback(tmp_path, monkeypatch):
    (tmp_path / "auth.md").write_text(_DOC, encoding="utf-8")

    class _RaisingGenerator(FakeGenerator):
        def stream(self, prompt, *, num_ctx):
            raise OllamaUnavailableError(
                "Cannot reach Ollama", hint="Start it with `ollama serve`"
            )
            yield  # pragma: no cover - makes this a generator

    settings = Settings.default(Path.cwd())
    ctx = build_context(
        settings,
        overrides={
            "embedder": FakeEmbedder(model_name="nomic-embed-text"),
            "generator": _RaisingGenerator(tokens=[]),
            "vector_store": InMemoryVectorStore(),
            "metadata_store": FakeMetadataStore(),
        },
    )
    monkeypatch.setattr(app_module, "build_context", lambda settings, **kw: ctx)

    runner.invoke(app_module.app, ["add", str(tmp_path)])
    result = runner.invoke(app_module.app, ["ask", "q"])

    assert result.exit_code == 1
    assert "ollama serve" in result.output.lower()
    assert "Traceback" not in result.output
