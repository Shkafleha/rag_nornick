"""Точка входа пайплайна image_pdf (сканированные PDF через OCR).

Запуск:
    docker compose --profile indexing run --rm indexer_image

TODO: реализовать после выбора OCR-движка в ocr.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from common import chunk_pages, embed_batch, ensure_collection, upload_chunks, delete_by_source
from image_pdf.ocr import ocr_pdf
from router import detect_pdf_type


def main() -> None:
    raw_dir = Path(os.getenv("RAW_DIR", "/data/00_raw"))
    recreate = os.getenv("RECREATE", "0") == "1"
    only = os.getenv("ONLY", "").strip()

    pdfs = sorted(raw_dir.glob("*.pdf"))
    if only:
        pdfs = [p for p in pdfs if only in p.name]

    force_ocr = os.getenv("FORCE_OCR", "0") == "1"
    if force_ocr:
        scans = pdfs
    else:
        scans = [p for p in pdfs if detect_pdf_type(p) == "ocr"]
    print(f"[image_pdf] Found {len(pdfs)} PDFs total, {len(scans)} to process (force_ocr={force_ocr})")

    if not scans:
        return

    probe = embed_batch(["probe"])
    ensure_collection(vector_size=len(probe[0]), recreate=recreate)

    global_chunk_id = 0
    for pdf in scans:
        print(f"[image_pdf] OCR {pdf.name}...")
        pages = ocr_pdf(pdf)
        chunks = chunk_pages(pages, start_chunk_id=global_chunk_id)
        if not chunks:
            continue
        delete_by_source(pdf.name)
        vectors = embed_batch([c.text for c in chunks])
        upload_chunks(chunks, vectors)
        global_chunk_id += len(chunks)

    print(f"[image_pdf] Done. Total chunks uploaded: {global_chunk_id}")


if __name__ == "__main__":
    main()
