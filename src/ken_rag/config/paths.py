"""Path helpers for the ken-rag data directory.

All paths are derived from a single *root* directory so the store can be
placed anywhere (default: current working directory).
"""
from pathlib import Path


def data_dir(root: Path) -> Path:
    """Return the hidden data directory: <root>/.ken"""
    return root / ".ken"


def db_path(root: Path) -> Path:
    """Return the LanceDB directory: <root>/.ken/lancedb"""
    return data_dir(root) / "lancedb"


def config_path(root: Path) -> Path:
    """Return the JSON config file path: <root>/.ken/config.json"""
    return data_dir(root) / "config.json"
