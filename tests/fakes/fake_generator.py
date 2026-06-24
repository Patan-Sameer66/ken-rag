"""Fake Generator for use in tests.

Implements the Generator protocol from ken_rag.domain.protocols.
Yields a scripted list of tokens passed at construction time.
Verifies that num_ctx is forwarded correctly.
"""
from __future__ import annotations

from typing import Iterator


class FakeGenerator:
    """A scripted Generator that yields a pre-set token list.

    Usage::

        gen = FakeGenerator(tokens=["Hello", " world", "!"], expected_num_ctx=8192)
        tokens = list(gen.stream("any prompt", num_ctx=8192))
        # tokens == ["Hello", " world", "!"]

    If *expected_num_ctx* is given and the caller passes a different value,
    :py:exc:`AssertionError` is raised to help catch missing num_ctx propagation.
    """

    def __init__(
        self,
        tokens: list[str],
        model: str = "fake-model",
        expected_num_ctx: int | None = None,
    ) -> None:
        self._tokens = tokens
        self._model = model
        self._expected_num_ctx = expected_num_ctx
        # Record calls for test assertions
        self.calls: list[dict] = []

    # ------------------------------------------------------------------
    # Generator protocol
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    def stream(self, prompt: str, *, num_ctx: int) -> Iterator[str]:
        if self._expected_num_ctx is not None:
            assert num_ctx == self._expected_num_ctx, (
                f"FakeGenerator: expected num_ctx={self._expected_num_ctx}, got {num_ctx}"
            )
        self.calls.append({"prompt": prompt, "num_ctx": num_ctx})
        yield from self._tokens
