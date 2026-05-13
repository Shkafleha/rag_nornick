"""Обёртка над Ollama /api/embed для батч-эмбеддинга."""
from __future__ import annotations

import os
import requests

OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://ollama_cpu:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:latest")


def embed_batch(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Эмбеддит список текстов, возвращает список векторов той же длины."""
    total = len(texts)
    out: list[list[float]] = []
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        r = requests.post(
            f"{OLLAMA_EMBED_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": batch},
            timeout=300,
        )
        r.raise_for_status()
        embs = r.json().get("embeddings") or []
        if len(embs) != len(batch):
            raise RuntimeError(f"Ollama returned {len(embs)} embeddings for {len(batch)} inputs")
        out.extend(embs)
        done = min(i + batch_size, total)
        print(f"\r  embed: {done}/{total} ({100 * done // total}%)", end="", flush=True)
    if total > 1:
        print()
    return out
