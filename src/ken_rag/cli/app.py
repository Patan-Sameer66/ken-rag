"""Typer root application for ken-rag.

The root callback builds an :class:`AppContext` (via :func:`build_context`) and
stores it on ``ctx.obj`` so every command can reach the wired-up pipelines.
Commands are registered additively — later phases append more commands here.

Testability: e2e tests monkeypatch ``ken_rag.cli.app.build_context`` to return a
context wired with fakes, so the CLI can be exercised without a live Ollama or a
real LanceDB on disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from ken_rag.cli.context import build_context
from ken_rag.cli.commands.add import add_command
from ken_rag.cli.commands.ask import ask_command
from ken_rag.cli.commands.list_cmd import list_command
from ken_rag.config.loader import load_settings

app = typer.Typer(
    name="ken",
    help="RAG that actually understands your code — local, private, in your terminal.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show extra detail.")
    ] = False,
    db: Annotated[
        Optional[Path],
        typer.Option("--db", help="Override the index location (defaults to ./.ken)."),
    ] = None,
) -> None:
    """Build the application context shared by all subcommands."""
    # A subcommand is always required (no_args_is_help=True handles the bare call).
    root = Path.cwd()
    overrides = {"db_path": db} if db is not None else None
    settings = load_settings(root, overrides=overrides)
    ctx.obj = build_context(settings)
    ctx.obj.verbose = verbose


# Register commands (additive — later phases add list/status/search/chat/setup/model).
app.command("add", help="Index a file or folder.")(add_command)
app.command("ask", help="Ask a question and get a grounded, cited answer.")(ask_command)
app.command("list", help="List indexed files.")(list_command)


def main() -> None:
    """Console-script entry point (``ken``)."""
    app()


if __name__ == "__main__":
    main()
