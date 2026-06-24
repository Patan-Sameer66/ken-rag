"""Streaming SHA-256 file hasher for ken-rag.

Reads files in chunks so arbitrarily large files do not hit memory limits.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 65_536  # 64 KiB read buffer


def sha256_file(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of the file at *path*.

    Reads the file in streaming 64 KiB blocks — safe for large files.

    Parameters
    ----------
    path:
        Path to the file to hash.

    Returns
    -------
    str
        Lowercase hex digest, e.g. ``"a3f1..."``.

    Raises
    ------
    OSError
        If the file cannot be opened or read.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(_CHUNK_SIZE)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
