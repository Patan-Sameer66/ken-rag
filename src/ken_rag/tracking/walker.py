"""Filesystem walker for ken-rag — Phase 1 minimal implementation.

Yields all regular files under *root*, skipping ``.ken/`` and ``.git/``
directories.  Phase 3 adds pathspec-based ignore_filter and git-awareness.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

# Directories to always skip (Phase 1 hard-coded list).
_SKIP_DIRS: frozenset[str] = frozenset({".ken", ".git"})


def walk_files(root: Path) -> Iterator[Path]:
    """Yield every regular file under *root*, recursively.

    Skips any directory whose name is in ``_SKIP_DIRS`` (``.ken``, ``.git``).
    Symbolic links to directories are not followed (default ``rglob`` behaviour).

    Parameters
    ----------
    root:
        Directory to walk.  If *root* is itself a regular file, yields it
        directly (allows callers to pass a single-file path).

    Yields
    ------
    Path
        Absolute (or rooted) paths of regular files.
    """
    if root.is_file():
        yield root
        return

    # Use a manual traversal so we can prune skip dirs efficiently.
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except PermissionError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in _SKIP_DIRS:
                    stack.append(entry)
            elif entry.is_file(follow_symlinks=False):
                yield entry
