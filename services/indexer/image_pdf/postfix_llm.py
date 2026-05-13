"""Пост-обработка OCR: читает JSON из ocr_debug, исправляет тексты через LLM, индексирует в Qdrant.

Запуск (без GPU — PaddleX не нужен):
    docker compose --profile indexing run --rm \
        -e ONLY=CEN1 \
        -e LLM_FIX_MODEL=qwen3:8b \
        --entrypoint python \
        indexer_text /app/image_pdf/postfix_llm.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, "/app")

from common import chunk_pages, embed_batch, ensure_collection, upload_chunks, delete_by_source
from common.schema import Page

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
LLM_FIX_MODEL = os.getenv("LLM_FIX_MODEL", "qwen3:8b")
OCR_OUT_DIR = Path(os.getenv("OCR_OUT_DIR", "/data/01_extracted_pages"))
ONLY = os.getenv("ONLY", "").strip()

_SKIP_LABELS = {"header", "footer", "number", "page_number"}


def _extract_text_and_tables(parsed_items: list[dict]) -> tuple[str, list[str]]:
    """Копия из ocr.py — достаёт текст и таблицы из JSON PP-StructureV3."""
    blocks: list[dict] = []
    tables: list[str] = []

    for item in parsed_items:
        res = item.get("res", item) if isinstance(item, dict) else item
        if not isinstance(res, dict):
            continue

        for t in res.get("table_res_list", []) or []:
            html = t.get("pred_html") or t.get("html")
            if html:
                tables.append(html)

        parsing = res.get("parsing_res_list") or []
        for block in parsing:
            content = block.get("block_content", "").strip()
            if not content:
                continue
            label = block.get("block_label", "")
            if label in _SKIP_LABELS:
                continue
            order = block.get("block_order")
            bbox = block.get("block_bbox", [0, 0, 0, 0])
            if "title" in label:
                content = f"## {content}"
            blocks.append({
                "order": order if order is not None else 9999,
                "y": bbox[1] if len(bbox) >= 2 else 0,
                "label": label,
                "text": content,
            })

        if not parsing:
            ocr = res.get("overall_ocr_res") or res.get("ocr_res") or {}
            rec_texts = ocr.get("rec_texts") or []
            if rec_texts:
                blocks.append({"order": 0, "y": 0, "label": "ocr", "text": "\n".join(rec_texts)})

    blocks.sort(key=lambda b: (b["order"], b["y"]))
    return "\n\n".join(b["text"] for b in blocks).strip(), tables


def _html_table_to_markdown(html: str) -> str:
    import re
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
    if not rows:
        return html
    md_rows = []
    for i, row in enumerate(rows):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        md_rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(md_rows)


def _llm_fix(text: str) -> str:
    if not text.strip():
        return text
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": LLM_FIX_MODEL,
                "messages": [{"role": "user", "content": (
                    "/no_think\n"
                    "Ниже текст, извлечённый OCR из скана технической инструкции на русском языке. "
                    "Исправь очевидные ошибки OCR, не меняя смысл и структуру. "
                    "Верни только исправленный текст, без комментариев.\n\n"
                    f"ТЕКСТ:\n{text}"
                )}],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 4096, "num_ctx": 4096},
            },
            timeout=180,
        )
        fixed = resp.json().get("message", {}).get("content", "")
        return fixed.strip() if fixed.strip() else text
    except Exception as e:
        print(f"  [warn] LLM fix failed: {e}")
        return text


def main() -> None:
    pdf_dirs = sorted(d for d in OCR_OUT_DIR.iterdir() if d.is_dir())
    if ONLY:
        pdf_dirs = [d for d in pdf_dirs if ONLY in d.name]

    if not pdf_dirs:
        print(f"[postfix] No OCR dirs found in {OCR_OUT_DIR} (ONLY={ONLY!r})")
        return

    probe = embed_batch(["probe"])
    ensure_collection(vector_size=len(probe[0]))

    global_chunk_id = 0
    for doc_dir in pdf_dirs:
        source = doc_dir.name + ".pdf"
        json_files = sorted(doc_dir.glob("page_*.json"))
        if not json_files:
            print(f"[postfix] {doc_dir.name}: no JSON files, skip")
            continue

        print(f"[postfix] {source}: {len(json_files)} pages, LLM fixing...")
        pages: list[Page] = []

        for jf in json_files:
            page_num = int(jf.stem.split("_")[1])
            with open(jf, encoding="utf-8") as f:
                parsed_items = json.load(f)

            text, tables = _extract_text_and_tables(parsed_items)
            tables = [_html_table_to_markdown(t) for t in tables]

            if text:
                text = _llm_fix(text)
            tables = [_llm_fix(t) for t in tables if t.strip()]

            print(f"  page {page_num}: {len(text)} chars (fixed), {len(tables)} tables")
            pages.append(Page(source=source, page=page_num, text=text, tables=tables, source_type="ocr"))

        chunks = chunk_pages(pages, start_chunk_id=global_chunk_id)
        if not chunks:
            print(f"[postfix] {source}: no chunks")
            continue

        delete_by_source(source)
        print(f"[postfix] {source}: {len(chunks)} chunks, embedding...")
        vectors = embed_batch([c.text for c in chunks])
        upload_chunks(chunks, vectors)
        global_chunk_id += len(chunks)
        print(f"[postfix] {source}: uploaded")

    print(f"[postfix] Done. Total chunks: {global_chunk_id}")


if __name__ == "__main__":
    main()
