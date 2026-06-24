"""ken_rag.tracking — minimal Phase-1 file hashing and walking.

Phase 3 adds ignore_filter, git_client, and the full FileTracker diff.
"""
from ken_rag.tracking.hasher import sha256_file
from ken_rag.tracking.walker import walk_files

__all__ = ["sha256_file", "walk_files"]
