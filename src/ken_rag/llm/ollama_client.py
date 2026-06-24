"""Low-level HTTP client for the Ollama REST API.

Wraps /api/embed (synchronous) and /api/generate (streaming JSONL).
Maps httpx transport errors to typed ken-rag errors so callers never
see raw httpx exceptions.
"""
from __future__ import annotations

import json
from typing import Iterator

import httpx

from ken_rag.domain.errors import ModelNotPulledError, OllamaUnavailableError


class OllamaClient:
    """Thin httpx wrapper over the Ollama REST API.

    Parameters
    ----------
    base_url:
        Root URL of the Ollama daemon, e.g. ``"http://localhost:11434"``.
    timeout:
        Seconds before an HTTP request times out. Defaults to 120 s so that
        large model loads don't abort prematurely.
    """

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        """Call ``POST /api/embed`` and return a list of embedding vectors.

        Parameters
        ----------
        model:
            The Ollama model name, e.g. ``"nomic-embed-text"``.
        inputs:
            Texts to embed (already prefixed by the caller if required).

        Returns
        -------
        list[list[float]]
            One vector per input string.

        Raises
        ------
        OllamaUnavailableError
            When the Ollama daemon cannot be reached.
        ModelNotPulledError
            When the model is not found (HTTP 404).
        httpx.HTTPStatusError
            For other non-2xx responses.
        """
        try:
            response = httpx.post(
                f"{self._base}/api/embed",
                json={"model": model, "input": inputs},
                timeout=self._timeout,
            )
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(
                "Cannot reach Ollama daemon.",
                hint=(
                    "Start it with `ollama serve`, "
                    "or install from https://ollama.com"
                ),
            ) from exc

        if response.status_code == 404:
            raise ModelNotPulledError(
                f"Model '{model}' not found in Ollama.",
                hint=f"Run `ollama pull {model}` to download it.",
            )

        response.raise_for_status()
        return response.json()["embeddings"]

    def generate_stream(
        self,
        model: str,
        prompt: str,
        num_ctx: int,
    ) -> Iterator[str]:
        """Stream tokens from ``POST /api/generate`` (JSONL response).

        Each line of the response is a JSON object with a ``"response"`` field
        containing the next token string. Lines with ``"done": true`` signal
        end-of-stream; their (empty) response field is not yielded.

        Parameters
        ----------
        model:
            The Ollama generation model name, e.g. ``"qwen2.5:3b"``.
        prompt:
            Full prompt text.
        num_ctx:
            Context-window size. Passed as ``options.num_ctx`` to Ollama.

        Yields
        ------
        str
            Individual token strings in generation order.

        Raises
        ------
        OllamaUnavailableError
            When the Ollama daemon cannot be reached.
        ModelNotPulledError
            When the model is not found (HTTP 404).
        httpx.HTTPStatusError
            For other non-2xx responses.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_ctx": num_ctx},
        }

        try:
            response = httpx.post(
                f"{self._base}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(
                "Cannot reach Ollama daemon.",
                hint=(
                    "Start it with `ollama serve`, "
                    "or install from https://ollama.com"
                ),
            ) from exc

        if response.status_code == 404:
            raise ModelNotPulledError(
                f"Model '{model}' not found in Ollama.",
                hint=f"Run `ollama pull {model}` to download it.",
            )

        response.raise_for_status()

        # Parse JSONL — each line is a separate JSON object.
        for raw_line in response.content.split(b"\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            obj = json.loads(raw_line)
            token: str = obj.get("response", "")
            done: bool = obj.get("done", False)
            if token:
                yield token
            if done:
                break
