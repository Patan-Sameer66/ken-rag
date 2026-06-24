"""OllamaGenerator — streaming token generator backed by an Ollama-compatible client.

The *client* argument is duck-typed: any object that exposes::

    def generate_stream(self, model: str, prompt: str, num_ctx: int) -> Iterator[str]:
        ...

This decouples the generator from the concrete ``ken_rag.llm.OllamaClient``
implementation (built by a separate agent) so this module can be tested in
isolation with a fake client and will not fail to import at collection time.
"""
from __future__ import annotations

from typing import Any, Iterator


class OllamaGenerator:
    """Streams tokens from an Ollama-compatible language-model client.

    Parameters
    ----------
    client:
        Duck-typed client with a ``generate_stream(model, prompt, num_ctx)``
        method that yields token strings.  The concrete type is intentionally
        not imported here to avoid circular imports and allow isolated testing.
    model_name:
        The Ollama model identifier, e.g. ``"qwen2.5:3b"``.
    """

    def __init__(self, client: Any, model_name: str) -> None:
        self._client = client
        self._model = model_name

    # ------------------------------------------------------------------
    # Generator protocol
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """The name of the active generation model."""
        return self._model

    def stream(self, prompt: str, *, num_ctx: int) -> Iterator[str]:
        """Yield token strings for *prompt*.

        Forwards the call to ``client.generate_stream`` with the configured
        model and the caller-supplied ``num_ctx``.  The ``num_ctx`` value is
        threaded through on every call — no code path bypasses it.

        Parameters
        ----------
        prompt:
            The fully-formed prompt string from :class:`~ken_rag.generation.prompt.PromptBuilder`.
        num_ctx:
            Context window size passed to the model (must be ≥ 8192 per spec).
        """
        yield from self._client.generate_stream(self._model, prompt, num_ctx)
