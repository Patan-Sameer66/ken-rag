"""Prompt construction for ken-rag generation.

:class:`PromptBuilder` assembles a retrieval-augmented prompt from a question
and a ranked list of :class:`~ken_rag.domain.models.RetrievedChunk` objects.

Prompt layout
-------------
1. System instruction — directs the model to answer **only** from the provided
   sources and to cite them by number.
2. Numbered sources — each rendered as::

       [n] file_path:line_start-line_end
       <chunk text>

3. Question — appears last so it is closest to the generation start.
"""
from __future__ import annotations

from ken_rag.domain.models import RetrievedChunk

_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "sources provided below. Do not use any outside knowledge. For every claim "
    "you make, cite the relevant source number (e.g. [1], [2]) inline. If the "
    "sources do not contain enough information to answer, say so."
)


class PromptBuilder:
    """Builds a RAG prompt from a question and retrieved chunks.

    This class is intentionally stateless — all methods are class-level so
    callers do not need to instantiate it.
    """

    @classmethod
    def build(cls, question: str, retrieved: list[RetrievedChunk]) -> str:
        """Return a fully-formed prompt string.

        Parameters
        ----------
        question:
            The user's question.
        retrieved:
            Ordered list of retrieved chunks to use as sources.
            May be empty; the prompt is still valid in that case.

        Returns
        -------
        str
            A prompt ready for submission to a language model.
        """
        parts: list[str] = [_SYSTEM_INSTRUCTION, ""]

        if retrieved:
            parts.append("Sources:")
            for idx, rc in enumerate(retrieved, start=1):
                chunk = rc.chunk
                header = f"[{idx}] {chunk.file_path}:{chunk.line_start}-{chunk.line_end}"
                parts.append(header)
                parts.append(chunk.text)
                parts.append("")  # blank line between sources
        else:
            parts.append("(No sources available.)")
            parts.append("")

        parts.append("Question:")
        parts.append(question)

        return "\n".join(parts)
