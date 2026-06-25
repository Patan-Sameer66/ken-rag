"""Tests for the Phase-1 filesystem walker."""
from __future__ import annotations

from ken_rag.tracking.walker import walk_files


def test_single_file_yields_itself(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("hello", encoding="utf-8")
    assert [p.name for p in walk_files(f)] == ["note.md"]


def test_walks_nested_files(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.py").write_text("x", encoding="utf-8")
    names = sorted(p.name for p in walk_files(tmp_path))
    assert names == ["a.md", "b.py"]


def test_skips_dependency_and_vcs_dirs(tmp_path):
    (tmp_path / "keep.md").write_text("x", encoding="utf-8")
    for junk_dir in (".venv", ".git", ".ken", "node_modules", "__pycache__"):
        d = tmp_path / junk_dir
        d.mkdir()
        (d / "junk.py").write_text("x", encoding="utf-8")
    names = sorted(p.name for p in walk_files(tmp_path))
    assert names == ["keep.md"]


def test_skips_nested_skip_dirs(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text("x", encoding="utf-8")
    cache = pkg / "__pycache__"
    cache.mkdir()
    (cache / "mod.cpython-313.pyc").write_text("x", encoding="utf-8")
    names = sorted(p.name for p in walk_files(tmp_path))
    assert names == ["mod.py"]
