"""``ken list`` — show the files currently in the index.

Thin command: reads ``store.list_files()`` and renders a table.
"""
from __future__ import annotations

import typer

from ken_rag.cli.errors import handle_ken_error
from ken_rag.cli.render import print_file_table
from ken_rag.domain.errors import KenError


def list_command(ctx: typer.Context) -> None:
    """List indexed files: path, type, chunk count, last indexed time."""
    app_ctx = ctx.obj
    if app_ctx is None:
        typer.echo("Internal error: no app context.", err=True)
        raise typer.Exit(code=1)

    try:
        records = app_ctx.store.list_files()
    except KenError as exc:
        handle_ken_error(exc)
        return  # unreachable

    print_file_table(records)
