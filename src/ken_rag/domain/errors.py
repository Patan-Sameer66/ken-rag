"""Error hierarchy for ken-rag.

All errors inherit from KenError and carry a user-facing `.hint` string
that the CLI renders as an actionable next step (never a raw traceback).
"""


class KenError(Exception):
    """Base exception for ken-rag. Carries a user-facing hint."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class OllamaUnavailableError(KenError):
    """Raised when the Ollama daemon cannot be reached."""


class ModelNotPulledError(KenError):
    """Raised when the requested model is not available in Ollama."""


class EmbedderMismatchError(KenError):
    """Raised when the current embedder name differs from the stored one."""


class DimensionMismatchError(KenError):
    """Raised when the embedder returns vectors of an unexpected dimension."""


class UnsupportedFileTypeError(KenError):
    """Raised when a file type has no registered parser."""
