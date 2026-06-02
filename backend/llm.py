"""Local LLM client (Ollama)."""
from __future__ import annotations

from functools import lru_cache

import ollama

from .config import settings


@lru_cache(maxsize=1)
def _client() -> ollama.Client:
    return ollama.Client(host=settings.ollama_host)


def chat(messages: list[dict[str, str]], temperature: float = 0.2) -> str:
    """Synchronous chat completion against the local Ollama server."""
    resp = _client().chat(
        model=settings.llm_model,
        messages=messages,
        options={"temperature": temperature},
    )
    return resp["message"]["content"].strip()


def complete(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
    msgs: list[dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return chat(msgs, temperature=temperature)
