"""Settings loader for ken-rag.

Merges three sources in ascending priority order:
  1. Compile-time defaults (lowest)
  2. JSON config file at <root>/.ken/config.json
  3. Environment variables prefixed KEN_  (e.g. KEN_K, KEN_LLM_NAME)
  4. Explicit overrides dict passed by the caller (highest)

All values are type-validated at the boundary; invalid values raise KenError
with an actionable hint rather than a raw TypeError.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ken_rag.config.defaults import (
    BATCH_SIZE,
    DEFAULT_K,
    DEFAULT_LLM,
    EMBED_DIM,
    EMBEDDER_NAME,
    NUM_CTX,
    OLLAMA_URL,
    TIMEOUT_S,
)
from ken_rag.config.paths import config_path, db_path
from ken_rag.config.settings import Settings
from ken_rag.domain.errors import KenError

# Mapping: settings field → (env var suffix, cast function)
_FIELD_SPEC: dict[str, tuple[str, type]] = {
    "k": ("K", int),
    "num_ctx": ("NUM_CTX", int),
    "embed_dim": ("EMBED_DIM", int),
    "embedder_name": ("EMBEDDER_NAME", str),
    "llm_name": ("LLM_NAME", str),
    "batch_size": ("BATCH_SIZE", int),
    "ollama_url": ("OLLAMA_URL", str),
    "timeout_s": ("TIMEOUT_S", int),
}


def _cast(field: str, value: Any, cast: type) -> Any:
    """Cast *value* to *cast*, raising KenError on failure."""
    try:
        return cast(value)
    except (ValueError, TypeError) as exc:
        raise KenError(
            f"Invalid config value for '{field}': {value!r} is not a valid {cast.__name__}",
            hint=f"Check your config file or the KEN_{field.upper()} environment variable.",
        ) from exc


def load_settings(
    root: Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load Settings by merging defaults → file → env → overrides."""
    # Start from compile-time defaults (as plain dict so we can mutate)
    values: dict[str, Any] = {
        "k": DEFAULT_K,
        "num_ctx": NUM_CTX,
        "embed_dim": EMBED_DIM,
        "embedder_name": EMBEDDER_NAME,
        "llm_name": DEFAULT_LLM,
        "batch_size": BATCH_SIZE,
        "db_path": db_path(root),
        "ollama_url": OLLAMA_URL,
        "timeout_s": TIMEOUT_S,
    }

    # Layer 2: JSON config file
    cfg_path = config_path(root)
    if cfg_path.exists():
        try:
            file_data: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise KenError(
                f"Config file is not valid JSON: {cfg_path}",
                hint="Fix the JSON syntax or delete the file to use defaults.",
            ) from exc

        for field, raw_value in file_data.items():
            if field not in _FIELD_SPEC:
                continue  # unknown keys are ignored silently
            _, cast = _FIELD_SPEC[field]
            values[field] = _cast(field, raw_value, cast)

    # Layer 3: environment variables  KEN_<SUFFIX>
    for field, (suffix, cast) in _FIELD_SPEC.items():
        env_key = f"KEN_{suffix}"
        raw = os.environ.get(env_key)
        if raw is not None:
            values[field] = _cast(field, raw, cast)

    # Layer 4: explicit overrides (highest priority, types assumed correct)
    if overrides:
        for field, val in overrides.items():
            if field in values:
                values[field] = val

    return Settings(
        k=values["k"],
        num_ctx=values["num_ctx"],
        embed_dim=values["embed_dim"],
        embedder_name=values["embedder_name"],
        llm_name=values["llm_name"],
        batch_size=values["batch_size"],
        db_path=values["db_path"],
        ollama_url=values["ollama_url"],
        timeout_s=values["timeout_s"],
    )
