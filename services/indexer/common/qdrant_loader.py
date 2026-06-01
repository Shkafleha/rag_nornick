"""Создание коллекции и загрузка чанков в Qdrant."""
from __future__ import annotations

import os
import uuid
import requests

from .schema import Chunk

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.getenv("COLLECTION")
if not COLLECTION:
    raise ValueError("COLLECTION environment variable is required")


def ensure_collection(vector_size: int, recreate: bool = False) -> None:
    """Создаёт коллекцию если её нет. При recreate=True — пересоздаёт."""
    if recreate:
        requests.delete(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=30)

    r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=10)
    if r.status_code == 200:
        return

    r = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        timeout=30,
    )
    r.raise_for_status()


def upload_chunks(chunks: list[Chunk], vectors: list[list[float]], batch_size: int = 128) -> None:
    """Загружает чанки батчами. chunks и vectors должны быть одинаковой длины."""
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks={len(chunks)} != vectors={len(vectors)}")

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_vecs = vectors[i : i + batch_size]
        points = [
            {
                "id": str(uuid.uuid4()),
                "vector": vec,
                "payload": ch.to_payload(),
            }
            for ch, vec in zip(batch_chunks, batch_vecs)
        ]
        r = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
            json={"points": points},
            timeout=300,
        )
        r.raise_for_status()


def delete_by_source(source: str) -> None:
    """Удаляет все точки с payload.source == source. Для переиндексации одного файла."""
    r = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/delete",
        json={"filter": {"must": [{"key": "source", "match": {"value": source}}]}},
        timeout=60,
    )
    r.raise_for_status()
