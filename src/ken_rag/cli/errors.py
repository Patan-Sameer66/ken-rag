"""KenError → friendly Rich message mapper for the CLI.

Converts ``KenError`` subclasses into user-readable output and exits with
code 1.  Never prints a traceback.

Usage
-----
Wrap any CLI callback body in a try/except block and delegate to
``handle_ken_error``, or use the ``@with_ken_errors`` decorator.
"""
from __future__ import annotations

import typer

from ken_rag.cli.render import print_error
from ken_rag.domain.errors import KenError


def handle_ken_error(exc: KenError) -> None:
    """Print a friendly error message for *exc* and raise ``typer.Exit(1)``.

    Parameters
    ----------
    exc:
        The ``KenError`` (or subclass) to render.

    Raises
    ------
    typer.Exit
        Always raised with code 1 after printing.
    """
    print_error(str(exc), hint=exc.hint)
    raise typer.Exit(code=1)
