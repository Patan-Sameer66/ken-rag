"""Tests for Task 1.6 — parsing module (registry + text/markdown parser).

TDD order: all tests in this file were written BEFORE the implementation.
"""
from __future__ import annotations

from pathlib import Path

from ken_rag.domain.enums import FileType


# ---------------------------------------------------------------------------
# detect_file_type
# ---------------------------------------------------------------------------


class TestDetectFileType:
    """detect_file_type maps extensions to FileType values."""

    def test_md_extension(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("README.md")) == FileType.MD

    def test_MD_uppercase_extension(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("README.MD")) == FileType.MD

    def test_txt_extension(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("notes.txt")) == FileType.TXT

    def test_pdf_extension(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("paper.pdf")) == FileType.PDF

    def test_py_extension_returns_code(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("main.py")) == FileType.CODE

    def test_js_extension_returns_code(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("app.js")) == FileType.CODE

    def test_ts_extension_returns_code(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("index.ts")) == FileType.CODE

    def test_go_extension_returns_code(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("main.go")) == FileType.CODE

    def test_rs_extension_returns_code(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("lib.rs")) == FileType.CODE

    def test_java_extension_returns_code(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("Main.java")) == FileType.CODE

    def test_c_extension_returns_code(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("main.c")) == FileType.CODE

    def test_cpp_extension_returns_code(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("main.cpp")) == FileType.CODE

    def test_h_extension_returns_code(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("header.h")) == FileType.CODE

    def test_rb_extension_returns_code(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("app.rb")) == FileType.CODE

    def test_php_extension_returns_code(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("index.php")) == FileType.CODE

    def test_unknown_extension_returns_unknown(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("archive.xyz")) == FileType.UNKNOWN

    def test_no_extension_returns_unknown(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("Makefile")) == FileType.UNKNOWN

    def test_nested_path_uses_final_extension(self) -> None:
        from ken_rag.parsing.registry import detect_file_type

        assert detect_file_type(Path("src/utils/helpers.ts")) == FileType.CODE


# ---------------------------------------------------------------------------
# ParserRegistry
# ---------------------------------------------------------------------------


class TestParserRegistry:
    """ParserRegistry.default() returns a mapping with the right parsers."""

    def test_default_returns_registry(self) -> None:
        from ken_rag.parsing.registry import ParserRegistry

        registry = ParserRegistry.default()
        assert registry is not None

    def test_txt_parser_registered(self) -> None:
        from ken_rag.parsing.registry import ParserRegistry
        from ken_rag.domain.protocols import Parser

        registry = ParserRegistry.default()
        parser = registry.get(FileType.TXT)
        assert parser is not None
        assert isinstance(parser, Parser)

    def test_md_parser_registered(self) -> None:
        from ken_rag.parsing.registry import ParserRegistry
        from ken_rag.domain.protocols import Parser

        registry = ParserRegistry.default()
        parser = registry.get(FileType.MD)
        assert parser is not None
        assert isinstance(parser, Parser)

    def test_pdf_not_registered(self) -> None:
        """PDF parser is out of scope for Task 1.6 — seam left for later."""
        from ken_rag.parsing.registry import ParserRegistry

        registry = ParserRegistry.default()
        assert registry.get(FileType.PDF) is None

    def test_code_not_registered(self) -> None:
        """CODE parser is wired in Phase 2 — seam left for later."""
        from ken_rag.parsing.registry import ParserRegistry

        registry = ParserRegistry.default()
        assert registry.get(FileType.CODE) is None

    def test_unknown_not_registered(self) -> None:
        from ken_rag.parsing.registry import ParserRegistry

        registry = ParserRegistry.default()
        assert registry.get(FileType.UNKNOWN) is None

    def test_txt_and_md_share_same_parser_type(self) -> None:
        """Both text types use TextParser — confirmed by class identity."""
        from ken_rag.parsing.registry import ParserRegistry
        from ken_rag.parsing.text_parser import TextParser

        registry = ParserRegistry.default()
        assert isinstance(registry.get(FileType.TXT), TextParser)
        assert isinstance(registry.get(FileType.MD), TextParser)


# ---------------------------------------------------------------------------
# TextParser
# ---------------------------------------------------------------------------


class TestTextParser:
    """TextParser.parse reads a file and returns a ParsedDocument."""

    def test_parse_returns_parsed_document(self, tmp_path: Path) -> None:
        from ken_rag.parsing.text_parser import TextParser
        from ken_rag.domain.models import ParsedDocument

        f = tmp_path / "hello.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        doc = TextParser().parse(f)
        assert isinstance(doc, ParsedDocument)

    def test_text_content_matches_file(self, tmp_path: Path) -> None:
        from ken_rag.parsing.text_parser import TextParser

        content = "Line one\nLine two\nLine three"
        f = tmp_path / "sample.txt"
        f.write_text(content, encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.text == content

    def test_line_count_single_line(self, tmp_path: Path) -> None:
        from ken_rag.parsing.text_parser import TextParser

        f = tmp_path / "single.txt"
        f.write_text("no newline", encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.line_count == 1

    def test_line_count_multiple_lines(self, tmp_path: Path) -> None:
        from ken_rag.parsing.text_parser import TextParser

        f = tmp_path / "multi.txt"
        f.write_text("a\nb\nc", encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.line_count == 3

    def test_line_count_trailing_newline(self, tmp_path: Path) -> None:
        """text.count('\\n')+1 — trailing newline counts as an extra line."""
        from ken_rag.parsing.text_parser import TextParser

        f = tmp_path / "trailing.txt"
        f.write_text("a\nb\n", encoding="utf-8")
        doc = TextParser().parse(f)
        # "a\nb\n".count("\n") == 2, so line_count == 3
        assert doc.line_count == 3

    def test_file_type_txt_for_txt_file(self, tmp_path: Path) -> None:
        from ken_rag.parsing.text_parser import TextParser

        f = tmp_path / "notes.txt"
        f.write_text("content", encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.file_type == FileType.TXT

    def test_file_type_md_for_md_file(self, tmp_path: Path) -> None:
        from ken_rag.parsing.text_parser import TextParser

        f = tmp_path / "README.md"
        f.write_text("# Title\n\nParagraph.", encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.file_type == FileType.MD

    def test_utf8_errors_replaced_not_raised(self, tmp_path: Path) -> None:
        """Files with bad byte sequences must not raise — bytes are replaced."""
        from ken_rag.parsing.text_parser import TextParser

        f = tmp_path / "bad.txt"
        f.write_bytes(b"Good text\xff\xfeMore text")
        doc = TextParser().parse(f)
        # Should not raise; text should be non-empty string
        assert isinstance(doc.text, str)
        assert "Good text" in doc.text
        assert "More text" in doc.text

    def test_empty_file(self, tmp_path: Path) -> None:
        from ken_rag.parsing.text_parser import TextParser

        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.text == ""
        # "".count("\n") == 0, so line_count == 1
        assert doc.line_count == 1

    def test_markdown_content_preserved(self, tmp_path: Path) -> None:
        """TextParser does NOT strip markdown — raw text is preserved."""
        from ken_rag.parsing.text_parser import TextParser

        md_content = "# Heading\n\n- item 1\n- item 2\n\n```python\nprint('hi')\n```\n"
        f = tmp_path / "doc.md"
        f.write_text(md_content, encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.text == md_content

    def test_implements_parser_protocol(self) -> None:
        from ken_rag.parsing.text_parser import TextParser
        from ken_rag.domain.protocols import Parser

        assert isinstance(TextParser(), Parser)
