"""Shared Rich rendering primitives for ken-rag CLI output.

Design spec (locked 2026-06-24):
- Signature accent: cyan/teal — prompts, citation numbers, active selections.
- ``dim`` for metadata (paths, line ranges, counts, key-hints).
- ``red`` ONLY for errors.
- Default fg for body text and answers.
- One frame max (box-drawing). No emoji-as-bullets, no multi-color.
- Respect ``NO_COLOR`` env and non-TTY → plain text, no spinners/frames.

Citation format (``① path:lines  symbol``):
    ① auth/middleware.py:14-37  require_auth
    ② auth/tokens.py:5-22       verify_token

Path and line range are dim; number and symbol_name are accent.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.theme import Theme

from ken_rag.domain.models import Answer, Citation

# ---------------------------------------------------------------------------
# Theme — defined once, reused everywhere
# ---------------------------------------------------------------------------

KEN_THEME = Theme(
    {
        "accent": "cyan",
        "meta": "dim",
        "error": "bold red",
        "user": "cyan",
        "ken": "default",
    }
)

# Circled digit characters for citation labels (① through ⑳).
_CIRCLED_DIGITS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def _make_console(*, stderr: bool = False) -> Console:
    """Build a Console that respects NO_COLOR and non-TTY environments."""
    no_color = bool(os.environ.get("NO_COLOR"))
    force_terminal = None  # let Rich auto-detect
    return Console(
        theme=KEN_THEME,
        no_color=no_color,
        force_terminal=force_terminal,
        stderr=stderr,
    )


# Default consoles — created lazily by helpers below.
def _stdout() -> Console:
    return _make_console(stderr=False)


def _stderr() -> Console:
    return _make_console(stderr=True)


# ---------------------------------------------------------------------------
# Citation rendering
# ---------------------------------------------------------------------------

def citation_label(index: int) -> str:
    """Return the circled digit for *index* (1-based).  Falls back to ``[N]``."""
    if 1 <= index <= len(_CIRCLED_DIGITS):
        return _CIRCLED_DIGITS[index - 1]
    return f"[{index}]"


def format_citation(citation: Citation, index: int) -> str:
    """Return a plain-text citation line for *citation*.

    Format: ``① path:start-end  symbol``
    """
    label = citation_label(index)
    loc = f"{citation.file_path}:{citation.line_start}-{citation.line_end}"
    sym = f"  {citation.symbol_name}" if citation.symbol_name else ""
    return f"{label} {loc}{sym}"


def print_answer(answer: Answer, *, console: Console | None = None) -> None:
    """Print the answer text followed by a dim citation block.

    Layout (matches TUI / CLI Design Spec ``ken ask`` one-shot output)::

        <answer text>

          ① auth/middleware.py:14-37  require_auth
          ② auth/tokens.py:5-22       verify_token

    Parameters
    ----------
    answer:
        The Answer to render.
    console:
        Rich Console to write to.  Creates a stdout console if not provided.
    """
    con = console or _stdout()
    # Answer block — default foreground, markdown-style.
    con.print(answer.text)

    if answer.citations:
        con.print()  # blank separator
        for i, cit in enumerate(answer.citations, start=1):
            label = citation_label(i)
            loc = f"{cit.file_path}:{cit.line_start}-{cit.line_end}"
            sym = f"  {cit.symbol_name}" if cit.symbol_name else ""
            # label+symbol in accent, path+range in dim
            con.print(
                f"  [accent]{label}[/accent] [meta]{loc}[/meta][accent]{sym}[/accent]"
            )


def print_stream(tokens: Iterator[str], *, console: Console | None = None) -> str:
    """Print tokens as they arrive and return the joined text.

    In non-TTY / NO_COLOR mode, tokens are flushed with ``print()`` to avoid
    Rich's buffering.  In TTY mode, Rich writes each token directly.

    Parameters
    ----------
    tokens:
        Iterator of token strings from the generator.
    console:
        Rich Console to write to.  Creates a stdout console if not provided.

    Returns
    -------
    str
        The complete answer text (all tokens joined).
    """
    con = console or _stdout()
    parts: list[str] = []
    for token in tokens:
        parts.append(token)
        con.print(token, end="", highlight=False)
    con.print()  # trailing newline after stream ends
    return "".join(parts)


@contextmanager
def status_spinner(message: str, *, console: Console | None = None):
    """Show a simple dot spinner with *message* while the block runs.

    Uses Rich's ``dots`` spinner in the accent color. On a non-TTY or with
    ``NO_COLOR`` set, Rich renders this without animation (no garbage in pipes).

    Usage::

        with status_spinner("Indexing…"):
            do_work()
    """
    con = console or _stdout()
    with con.status(f"[accent]{message}[/accent]", spinner="dots"):
        yield


def print_error(message: str, hint: str = "", *, console: Console | None = None) -> None:
    """Print a user-facing error message in red, optionally with a hint.

    Parameters
    ----------
    message:
        The main error description.
    hint:
        Optional actionable next step (shown in dim below the error).
    console:
        Rich Console to write to stderr.  Creates one if not provided.
    """
    con = console or _stderr()
    con.print(f"[error]Error:[/error] {message}")
    if hint:
        con.print(f"[meta]Hint:[/meta] {hint}")
