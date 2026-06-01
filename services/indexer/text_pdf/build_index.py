"""Точка входа пайплайна text_pdf.

Обходит data/raw/*.pdf, фильтрует по типу (только цифровые),
парсит → (опционально LLM-коррекция) → чанкит → эмбеддит → грузит в Qdrant.

Запуск:
    docker compose --profile indexing run --rm indexer_text
    docker compose --profile indexing run --rm -e LLM_FIX=1 indexer_text

Env:
    RAW_DIR          default /data/00_raw
    COLLECTION       REQUIRED
    QDRANT_URL       default http://qdrant:6333
    OLLAMA_EMBED_URL default http://ollama_cpu:11434
    EMBED_MODEL      default bge-m3:latest
    OLLAMA_URL       default http://ollama:11434
    LLM_FIX          "1" для LLM-коррекции текста (default "0")
    LLM_FIX_MODEL    default qwen3:8b
    RECREATE         "1" чтобы пересоздать коллекцию (default "0")
    ONLY             опциональный фильтр по имени файла (подстрока)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

# common/ монтируется как /app/common через volume (см. docker-compose.yml)
sys.path.insert(0, "/app")

from common import (
    chunk_pages,
    embed_batch,
    ensure_collection,
    upload_chunks,
    delete_by_source,
)
from text_pdf.parser import parse_pdf
from router import detect_pdf_type

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
LLM_FIX_MODEL = os.getenv("LLM_FIX_MODEL", "qwen3:8b")
LLM_FIX_ENABLED = os.getenv("LLM_FIX", "0") == "1"
LLM_FIX_TIMEOUT = int(os.getenv("LLM_FIX_TIMEOUT", "300"))
FIXED_OUT_DIR = Path(os.getenv("FIXED_OUT_DIR", "/data/02_normalized_text"))


def _llm_fix(text: str) -> str:
    """Исправляет мусор и ошибки извлечения текста через LLM."""
    if not text.strip():
        return text
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": LLM_FIX_MODEL,
                "messages": [{"role": "user", "content": (
                    "/no_think\n"
                    "Ниже текст, извлечённый из PDF технической инструкции. "
                    "В нём могут быть мусорные строки из чертежей (одиночные буквы, "
                    "бессмысленные последовательности символов). "
                    "Удали мусорные строки и исправь очевидные ошибки, "
                    "не меняя смысл и структуру. "
                    "Верни только исправленный текст, без комментариев.\n\n"
                    f"ТЕКСТ:\n{text}"
                )}],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 4096, "num_ctx": 4096},
            },
            timeout=LLM_FIX_TIMEOUT,
        )
        fixed = resp.json().get("message", {}).get("content", "")
        return fixed.strip() if fixed.strip() else text
    except Exception as e:
        print(f"  [warn] LLM fix failed: {e}")
        return text


def main() -> None:
    raw_dir = Path(os.getenv("RAW_DIR", "/data/00_raw"))
    recreate = os.getenv("RECREATE", "0") == "1"
    only = os.getenv("ONLY", "").strip()

    pdfs = sorted(raw_dir.glob("*.pdf"))
    if only:
        pdfs = [p for p in pdfs if only in p.name]
    if not pdfs:
        print(f"No PDFs found in {raw_dir}")
        return

    # Берём только цифровые PDF
    digital = [p for p in pdfs if detect_pdf_type(p) == "text"]
    print(f"[text_pdf] Found {len(pdfs)} PDFs total, {len(digital)} digital, LLM_FIX={LLM_FIX_ENABLED}")

    if not digital:
        return

    # Создаём коллекцию (размер вектора узнаём по первому эмбеддингу)
    probe = embed_batch(["probe"])
    vector_size = len(probe[0])
    ensure_collection(vector_size=vector_size, recreate=recreate)

    global_chunk_id = 0
    for pdf in digital:
        print(f"[text_pdf] Parsing {pdf.name}...")
        pages = parse_pdf(pdf)

        if LLM_FIX_ENABLED:
            total = len(pages)
            ckpt_dir = FIXED_OUT_DIR / pdf.stem
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            print(f"[text_pdf]   {pdf.name}: LLM fixing {total} pages (checkpoint: {ckpt_dir})...")
            skipped = 0
            for i, page in enumerate(pages, 1):
                ckpt = ckpt_dir / f"page_{page.page:03d}.json"
                if ckpt.exists():
                    try:
                        data = json.loads(ckpt.read_text(encoding="utf-8"))
                        page.text = data.get("text", page.text)
                        page.tables = data.get("tables", page.tables)
                        skipped += 1
                        if skipped <= 3 or skipped % 20 == 0:
                            print(f"  page {page.page}: cached  [{i}/{total}]")
                        continue
                    except Exception as e:
                        print(f"  page {page.page}: bad checkpoint ({e}), re-fixing")
                if page.text:
                    page.text = _llm_fix(page.text)
                page.tables = [_llm_fix(t) for t in page.tables if t.strip()]
                ckpt.write_text(
                    json.dumps({"page": page.page, "text": page.text, "tables": page.tables}, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"  page {page.page}: {len(page.text)} chars (fixed)  [{i}/{total} {100*i//total}%]")
            if skipped:
                print(f"[text_pdf]   {pdf.name}: skipped {skipped} cached pages")

        chunks = chunk_pages(pages, start_chunk_id=global_chunk_id)
        if not chunks:
            print(f"[text_pdf]   {pdf.name}: no chunks")
            continue

        # Идемпотентность: удаляем старые чанки этого файла перед загрузкой
        delete_by_source(pdf.name)

        print(f"[text_pdf]   {pdf.name}: {len(chunks)} chunks, embedding...")
        vectors = embed_batch([c.text for c in chunks])
        upload_chunks(chunks, vectors)
        print(f"[text_pdf]   {pdf.name}: uploaded")
        global_chunk_id += len(chunks)

    print(f"[text_pdf] Done. Total chunks uploaded: {global_chunk_id}")


if __name__ == "__main__":
    main()
