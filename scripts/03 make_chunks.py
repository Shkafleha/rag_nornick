"""Чанкинг нормализованных страниц → 03_chunks.

Читает page_*.json из 02_normalized_text/{doc_name}/,
прогоняет через chunk_pages() из services/indexer/common,
сохраняет chunks.json в 03_chunks/{doc_name}/.

Использование:
    python scripts/make_chunks.py CEN1_ТИ_3-48200234-05.1-12-2020_Очистка_электролита_от_примесей
    python scripts/make_chunks.py doc1 doc2
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Импортируем chunker из services/indexer
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "services" / "indexer"))

from common.chunker import chunk_pages
from common.schema import Page

if Path("/data").exists():
    _base = Path("/data")
else:
    _base = _root / "data"

NORM_DIR = Path(os.getenv("NORM_DIR", _base / "02_normalized_text"))
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", _base / "03_chunks"))


def chunk_document(doc_name: str) -> None:
    norm_doc_dir = NORM_DIR / doc_name
    if not norm_doc_dir.exists():
        print(f"[error] {norm_doc_dir} не найдена")
        return

    page_files = sorted(norm_doc_dir.glob("page_*.json"))
    if not page_files:
        print(f"[error] Нет page_*.json в {norm_doc_dir}")
        return

    pages: list[Page] = []
    page_texts: list[tuple[int, str]] = []  # (page_num, text)
    for pf in page_files:
        data = json.loads(pf.read_text(encoding="utf-8"))
        pages.append(Page(
            source=doc_name + ".pdf",
            page=data["page"],
            text=data.get("text", ""),
            tables=data.get("tables", []),
            images=data.get("images", []),
            source_type="ocr",
        ))
        page_texts.append((data["page"], data.get("text", "")))

    chunks = chunk_pages(pages)

    # Для text-чанков определяем страницу по первым 80 символам текста
    for c in chunks:
        if c.type != "text" or c.page is not None:
            continue
        # Убираем breadcrumb-префикс вида "[...]\n"
        raw = c.text
        if raw.startswith("["):
            newline = raw.find("\n")
            if newline != -1:
                raw = raw[newline + 1:]
        probe = raw.strip()[:80]
        if not probe:
            continue
        for pg_num, pg_text in page_texts:
            if probe in pg_text:
                c.page = pg_num
                break

    out_dir = CHUNKS_DIR / doc_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "chunks.json"
    out_file.write_text(
        json.dumps([c.to_payload() for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    text_chunks = sum(1 for c in chunks if c.type == "text")
    table_chunks = sum(1 for c in chunks if c.type == "table")
    image_chunks = sum(1 for c in chunks if c.type == "image")
    print(f"{doc_name}: {len(pages)} pages -> {len(chunks)} chunks "
          f"(text={text_chunks}, table={table_chunks}, image={image_chunks})")
    print(f"  -> {out_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python scripts/make_chunks.py <doc_name> [doc_name2 ...]")
        sys.exit(1)
    for name in sys.argv[1:]:
        chunk_document(name)
