"""Index pre-chunked data from data/03_chunks into Qdrant.

Loads chunks.json files, embeds them, and uploads to Qdrant.

Env:
    CHUNKS_DIR       default /data/03_chunks
    COLLECTION       REQUIRED
    QDRANT_URL       default http://qdrant:6333
    OLLAMA_EMBED_URL default http://ollama_cpu:11434
    EMBED_MODEL      default bge-m3:latest
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import requests

CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "/data/03_chunks"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://ollama_cpu:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:latest")
COLLECTION = os.getenv("COLLECTION")

if not COLLECTION:
    raise ValueError("COLLECTION environment variable is required")


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed texts using Ollama."""
    resp = requests.post(
        f"{OLLAMA_EMBED_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def ensure_collection(vector_size: int) -> None:
    """Create collection if it doesn't exist."""
    r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=10)
    if r.status_code == 200:
        return

    r = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        timeout=30,
    )
    r.raise_for_status()


def upload_chunks(chunks: list[dict], vectors: list[list[float]], batch_size: int = 128) -> None:
    """Upload chunks and vectors to Qdrant."""
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks={len(chunks)} != vectors={len(vectors)}")

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_vecs = vectors[i : i + batch_size]
        points = [
            {
                "id": str(uuid.uuid4()),
                "vector": vec,
                "payload": chunk,
            }
            for chunk, vec in zip(batch_chunks, batch_vecs)
        ]
        r = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
            json={"points": points},
            timeout=300,
        )
        r.raise_for_status()


def main() -> None:
    chunks_dirs = sorted(CHUNKS_DIR.glob("*/chunks.json"))
    if not chunks_dirs:
        print(f"No chunks.json found in {CHUNKS_DIR}")
        return

    print(f"[chunks_indexer] Found {len(chunks_dirs)} chunk files")

    all_chunks = []
    for chunks_file in chunks_dirs:
        print(f"[chunks_indexer] Loading {chunks_file.parent.name}/chunks.json...")
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"[chunks_indexer]   {len(chunks)} chunks loaded")
        all_chunks.extend(chunks)

    print(f"[chunks_indexer] Total chunks: {len(all_chunks)}")

    if not all_chunks:
        return

    # Create collection
    print(f"[chunks_indexer] Creating collection...")
    probe = embed_batch(["probe"])
    vector_size = len(probe[0])
    ensure_collection(vector_size=vector_size)

    # Embed and upload
    print(f"[chunks_indexer] Embedding {len(all_chunks)} chunks...")
    texts = [ch.get("text", "") for ch in all_chunks]
    vectors = embed_batch(texts)

    print(f"[chunks_indexer] Uploading to Qdrant...")
    upload_chunks(all_chunks, vectors)

    print(f"[chunks_indexer] Done. Total chunks uploaded: {len(all_chunks)}")


if __name__ == "__main__":
    main()
