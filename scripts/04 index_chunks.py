"""Index pre-chunked data from data/03_chunks into Qdrant with index_text field.

Usage:
    python scripts/04\ index_chunks.py <collection_name> [chunks_file]
    python scripts/04\ index_chunks.py CEN1_TI_3_48200234
    python scripts/04\ index_chunks.py CEN1_TI_3_48200234 custom_chunks.json

Env:
    QDRANT_URL       default http://localhost:6333
    OLLAMA_EMBED_URL default http://localhost:11434
    EMBED_MODEL      default bge-m3:latest

Features:
    - Creates index_text field with section breadcrumb, page range, and overlap text
    - Includes table headers and rows for table chunks
    - Embeds index_text (not raw text) for better semantic search
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import requests

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:latest")

# Find project root (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CHUNKS_DIR = PROJECT_ROOT / "data" / "03_chunks"


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed texts using Ollama."""
    resp = requests.post(
        f"{OLLAMA_EMBED_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=900,  # 15 minutes timeout
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def ensure_collection(collection: str, vector_size: int) -> None:
    """Create collection if it doesn't exist."""
    r = requests.get(f"{QDRANT_URL}/collections/{collection}", timeout=10)
    if r.status_code == 200:
        print(f"[index_chunks] Collection '{collection}' exists")
        return

    print(f"[index_chunks] Creating collection '{collection}'...")
    r = requests.put(
        f"{QDRANT_URL}/collections/{collection}",
        json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        timeout=30,
    )
    r.raise_for_status()


def upload_chunks(collection: str, chunks: list[dict], vectors: list[list[float]], batch_size: int = 128) -> None:
    """Upload chunks and vectors to Qdrant."""
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks={len(chunks)} != vectors={len(vectors)}")

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch_num, i in enumerate(range(0, len(chunks), batch_size), 1):
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
            f"{QDRANT_URL}/collections/{collection}/points?wait=true",
            json={"points": points},
            timeout=300,
        )
        r.raise_for_status()
        print(f"  batch {batch_num}/{total_batches}")


def build_index_text(chunk: dict) -> str:
    """Build index_text field for a chunk with context and metadata."""
    parts = []

    # Add section breadcrumb
    if "section_breadcrumb" in chunk:
        parts.append(f"[Раздел: {chunk['section_breadcrumb']}]")

    # Add page range
    if "page_range" in chunk:
        parts.append(f"[Страницы: {chunk['page_range']}]")

    # Add previous overlap text if exists
    if "prev_overlap_text" in chunk and chunk["prev_overlap_text"]:
        parts.append(chunk["prev_overlap_text"])

    # Add empty line
    parts.append("")

    # Add main content based on chunk type
    chunk_type = chunk.get("type", "text")

    if chunk_type == "table":
        # For tables: add header info and table rows
        if "table_header" in chunk and chunk["table_header"]:
            parts.append("Колонки таблицы:")
            header_line = " | ".join(chunk["table_header"])
            parts.append(header_line)
            parts.append("")

        parts.append("Строки таблицы:")
        parts.append(chunk.get("text", ""))
    else:
        # For text chunks, just add the text
        parts.append(chunk.get("text", ""))

    return "\n".join(parts)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/04\\ index_chunks.py <collection_name> [chunks_file]")
        print("Example: python scripts/04\\ index_chunks.py CEN1_TI_3_48200234")
        print("Example: python scripts/04\\ index_chunks.py CEN1_TI_3_48200234 custom_chunks.json")
        sys.exit(1)

    collection = sys.argv[1]
    chunks_dir = CHUNKS_DIR / collection

    # Try exact match first, then partial match
    if not chunks_dir.exists():
        # Try to find matching directory by partial name
        matching_dirs = [d for d in CHUNKS_DIR.iterdir() if d.is_dir() and collection in str(d.name)]
        if matching_dirs:
            chunks_dir = matching_dirs[0]
            print(f"[index_chunks] Found matching directory: {chunks_dir.name}")
        else:
            print(f"[index_chunks] ERROR: {chunks_dir} not found")
            print(f"[index_chunks] Available directories:")
            for d in CHUNKS_DIR.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
            sys.exit(1)

    # Determine which file to use
    if len(sys.argv) > 2:
        chunks_file = chunks_dir / sys.argv[2]
    else:
        # Try default chunks.json first, then look for any .json file
        chunks_file = chunks_dir / "chunks.json"
        if not chunks_file.exists():
            json_files = list(chunks_dir.glob("*.json"))
            if json_files:
                chunks_file = json_files[0]
                print(f"[index_chunks] chunks.json not found, using {chunks_file.name}")

    if not chunks_file.exists():
        print(f"[index_chunks] ERROR: {chunks_file} not found")
        sys.exit(1)

    print(f"[index_chunks] Loading {chunks_file}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[index_chunks] Loaded {len(chunks)} chunks")

    if not chunks:
        return

    # Build index_text for each chunk
    print(f"[index_chunks] Building index_text for {len(chunks)} chunks...")
    for chunk in chunks:
        chunk["index_text"] = build_index_text(chunk)

    # Create collection
    print(f"[index_chunks] Creating collection...")
    probe = embed_batch(["probe"])
    vector_size = len(probe[0])
    ensure_collection(collection, vector_size=vector_size)

    # Embed and upload
    print(f"[index_chunks] Embedding {len(chunks)} chunks...")
    texts = [ch.get("index_text", "") for ch in chunks]

    # Embed in smaller batches to avoid timeout
    vectors = []
    embed_batch_size = 32
    total_batches = (len(texts) + embed_batch_size - 1) // embed_batch_size
    for batch_num, i in enumerate(range(0, len(texts), embed_batch_size), 1):
        batch_texts = texts[i : i + embed_batch_size]
        batch_vectors = embed_batch(batch_texts)
        vectors.extend(batch_vectors)
        print(f"  embedded batch {batch_num}/{total_batches}")

    print(f"[index_chunks] Uploading to Qdrant (batch_size=128)...")
    upload_chunks(collection, chunks, vectors)

    print(f"[index_chunks] Done. Total chunks uploaded: {len(chunks)}")


if __name__ == "__main__":
    main()
