"""Parsing package for ken-rag.

Public surface:
    detect_file_type  — map a file path to its FileType
    ParserRegistry    — mapping of FileType → Parser, built via .default()
    TextParser        — Parser for .txt and .md files
"""
from ken_rag.parsing.registry import ParserRegistry, detect_file_type
from ken_rag.parsing.text_parser import TextParser

__all__ = ["ParserRegistry", "TextParser", "detect_file_type"]
