"""``ken ask "<question>"`` — ask a question against the indexed knowledge base.

Thin command: delegates to ``QueryPipeline.ask``, then renders the answer
and citations using the shared ``render`` helpers.
"""
from __future__ import annotations

from typing import Annotated

import typer

from ken_rag.cli.errors import handle_ken_error
from ken_rag.cli.render import print_answer
from ken_rag.domain.errors import KenError


def ask_command(
    question: Annotated[str, typer.Argument(help="Question to ask.")],
    ctx: typer.Context,
) -> None:
    """Answer QUESTION using the indexed knowledge base and show citations."""
    app_ctx = ctx.obj
    if app_ctx is None:
        typer.echo("Internal error: no app context.", err=True)
        raise typer.Exit(code=1)

    try:
        answer = app_ctx.query.ask(question)
    except KenError as exc:
        handle_ken_error(exc)
        return  # unreachable

    print_answer(answer)
