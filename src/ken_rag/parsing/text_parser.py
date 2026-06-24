"""TextParser — UTF-8 reader for plain text and Markdown files.

Implements the Parser protocol from ken_rag.domain.protocols.
Reads the full file content with errors="replace" (never raises on bad bytes).
line_count is computed as text.count("\\n") + 1, consistent with the spec.
"""
from __future__ import annotations

from pathlib import Path

from ken_rag.domain.models import ParsedDocument
from ken_rag.parsing.registry import detect_file_type


class TextParser:
    """Parse .txt and .md files into a ParsedDocument.

    File bytes are decoded as UTF-8 with errors="replace" so malformed
    sequences never raise — they are replaced with the Unicode replacement
    character (U+FFFD) instead.
    """

    def parse(self, path: Path) -> ParsedDocument:
        """Read *path* and return a ParsedDocument with its text and metadata."""
        text = path.read_text(encoding="utf-8", errors="replace")
        file_type = detect_file_type(path)
        line_count = text.count("\n") + 1
        return ParsedDocument(text=text, file_type=file_type, line_count=line_count)
