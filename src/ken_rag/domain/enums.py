"""Domain enumerations for ken-rag.

All enums are str-based so they serialise cleanly to/from JSON and LanceDB.
"""
from enum import Enum


class FileType(str, Enum):
    TXT = "txt"
    MD = "md"
    PDF = "pdf"
    CODE = "code"
    UNKNOWN = "unknown"


class ChunkKind(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    PROSE = "prose"
    MODULE = "module"
    FALLBACK = "fallback"


class FileChangeState(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


class ModelTier(str, Enum):
    HIGH = "qwen2.5:3b"
    MID = "qwen2.5:1.5b"
    LOW = "llama3.2:1b"
