"""File-type detection and parser registry for ken-rag.

detect_file_type(path)  — map any Path to a FileType based on its suffix.
ParserRegistry          — maps FileType → Parser; built via .default().

Design notes
------------
* Extension lookup is case-insensitive (.MD == .md).
* CODE extensions follow the architecture doc §Task 1.6:
  .py .js .ts .go .rs .java .c .cpp .h .rb .php
* PDF and CODE parsers are NOT wired in this task; the mapping leaves None
  for those slots so future tasks can register their parsers without changing
  detect_file_type or the registry contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ken_rag.domain.enums import FileType

if TYPE_CHECKING:
    from ken_rag.domain.protocols import Parser

# ---------------------------------------------------------------------------
# Extension → FileType mapping (case-insensitive on lookup)
# ---------------------------------------------------------------------------

_EXT_TO_FILE_TYPE: dict[str, FileType] = {
    ".txt": FileType.TXT,
    ".md": FileType.MD,
    ".pdf": FileType.PDF,
    # Code extensions per architecture doc §Task 1.6
    ".py": FileType.CODE,
    ".js": FileType.CODE,
    ".ts": FileType.CODE,
    ".go": FileType.CODE,
    ".rs": FileType.CODE,
    ".java": FileType.CODE,
    ".c": FileType.CODE,
    ".cpp": FileType.CODE,
    ".h": FileType.CODE,
    ".rb": FileType.CODE,
    ".php": FileType.CODE,
}


def detect_file_type(path: Path) -> FileType:
    """Return the FileType for *path* based on its file extension.

    The lookup is case-insensitive (.MD and .md both → FileType.MD).
    Extensions not in the map return FileType.UNKNOWN.
    """
    suffix = path.suffix.lower()
    return _EXT_TO_FILE_TYPE.get(suffix, FileType.UNKNOWN)


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------


class ParserRegistry:
    """Mapping of FileType → Parser, assembled once via .default().

    Seams for future parsers (PDF via PyMuPDF, CODE via tree-sitter) are left
    as None entries and filled in by the respective Phase-2/3 tasks.
    """

    def __init__(self, mapping: dict[FileType, "Parser | None"]) -> None:
        self._mapping = mapping

    def get(self, file_type: FileType) -> "Parser | None":
        """Return the Parser for *file_type*, or None if not yet registered."""
        return self._mapping.get(file_type)

    @classmethod
    def default(cls) -> "ParserRegistry":
        """Build the default registry with text/markdown parsers registered.

        PDF and CODE parsers are deliberately omitted here — they will be
        registered in Phase 2 (code-aware chunking) and Phase 3 (PDF).
        """
        # Import here to avoid circular imports at module load time.
        from ken_rag.parsing.text_parser import TextParser

        text_parser = TextParser()
        mapping: dict[FileType, Parser | None] = {
            FileType.TXT: text_parser,
            FileType.MD: text_parser,
            # Seams for future parsers — None signals "not yet implemented"
            FileType.PDF: None,
            FileType.CODE: None,
            FileType.UNKNOWN: None,
        }
        return cls(mapping)
