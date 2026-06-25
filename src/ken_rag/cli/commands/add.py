"""``ken add <path>`` — ingest files into the index.

Thin command: resolves the path, delegates to ``IngestPipeline.run``,
and prints a summary.  All heavy lifting is in the pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ken_rag.cli.errors import handle_ken_error
from ken_rag.cli.render import status_spinner
from ken_rag.domain.errors import KenError


def add_command(
    path: Annotated[Path, typer.Argument(help="Directory or file to index.")],
    ctx: typer.Context,
) -> None:
    """Index files under PATH and store their embeddings."""
    app_ctx = ctx.obj
    if app_ctx is None:
        typer.echo("Internal error: no app context.", err=True)
        raise typer.Exit(code=1)

    resolved = path.resolve()
    if not resolved.exists():
        typer.echo(f"Path does not exist: {resolved}", err=True)
        raise typer.Exit(code=1)

    target = "file" if resolved.is_file() else "folder"
    try:
        with status_spinner(f"Indexing {target}: {resolved}"):
            count = app_ctx.ingest.run(resolved)
    except KenError as exc:
        handle_ken_error(exc)
        return  # unreachable — handle_ken_error raises; here for type-checker

    if count == 0:
        typer.echo("No new or changed files to index.")
    else:
        typer.echo(f"Indexed {count} file{'s' if count != 1 else ''}.")
