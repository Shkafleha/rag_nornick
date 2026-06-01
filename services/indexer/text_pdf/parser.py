"""PDF парсер для цифровых документов. Обёртка над pdfplumber + PyMuPDF.

Извлекает текст, таблицы и встроенные изображения.
Изображения сохраняются на диск; описание через vision LLM — только если VISION_DESCRIBE=1.

Env:
    VISION_DESCRIBE  "1" включает LLM-описание изображений (default "0")
    VISION_MODEL     модель Ollama с vision (default "llava:7b")
    IMG_MIN_PX       минимальный размер стороны изображения в пикселях (default 80)
    FIXED_OUT_DIR    куда сохранять кропы (default /data/02_normalized_text)
    OLLAMA_URL       default http://ollama:11434
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import requests

from common.schema import Page

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "llava:7b")
VISION_ENABLED = os.getenv("VISION_DESCRIBE", "0") == "1"
IMG_MIN_PX = int(os.getenv("IMG_MIN_PX", "80"))
_IMG_OUT_BASE = Path(os.getenv("FIXED_OUT_DIR", "/data/02_normalized_text"))

_VISION_PROMPT = (
    "/no_think\n"
    "Ты описываешь рисунок из технической инструкции на русском языке. "
    "Опиши кратко, что изображено: схема, диаграмма, фотография, чертёж, "
    "и какие ключевые элементы или подписи видны. "
    "Ответ только на русском языке, одним абзацем, без списков."
)


def _describe_image_llm(img_bytes: bytes, source: str, page: int, idx: int) -> str:
    """Vision LLM описание — только если VISION_DESCRIBE=1."""
    if not VISION_ENABLED or not img_bytes:
        return ""
    try:
        b64 = base64.b64encode(img_bytes).decode()
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": VISION_MODEL,
                "messages": [{"role": "user", "content": _VISION_PROMPT, "images": [b64]}],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=120,
        )
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"  [warn] vision LLM {source} p{page} img{idx}: {e}")
        return ""


def parse_pdf(pdf_path: Path) -> list[Page]:
    pages: list[Page] = []
    source = pdf_path.name
    img_out_dir = _IMG_OUT_BASE / pdf_path.stem

    fitz_doc = fitz.open(str(pdf_path))
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, p in enumerate(pdf.pages, start=1):
                text = (p.extract_text() or "").strip()

                tables_md: list[str] = []
                for tbl in p.extract_tables() or []:
                    rows = ["| " + " | ".join((c or "").strip() for c in row) + " |" for row in tbl]
                    if rows:
                        sep = "|" + "|".join(["---"] * len(tbl[0])) + "|"
                        tables_md.append("\n".join([rows[0], sep, *rows[1:]]))

                # Встроенные изображения через PyMuPDF
                images: list[dict] = []
                fitz_page = fitz_doc[i - 1]
                seen_xrefs: set[int] = set()
                for img_ref in fitz_page.get_images(full=True):
                    xref = img_ref[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        img_info = fitz_doc.extract_image(xref)
                    except Exception:
                        continue
                    w, h = img_info.get("width", 0), img_info.get("height", 0)
                    if w < IMG_MIN_PX or h < IMG_MIN_PX:
                        continue
                    img_bytes = img_info.get("image", b"")
                    ext = img_info.get("ext", "png")
                    if not img_bytes:
                        continue

                    idx = len(images)

                    # Сохраняем на диск
                    img_out_dir.mkdir(parents=True, exist_ok=True)
                    img_file = img_out_dir / f"page_{i:03d}_img{idx:02d}.{ext}"
                    img_file.write_bytes(img_bytes)

                    desc = _describe_image_llm(img_bytes, source, i, idx)
                    images.append({"width": w, "height": h, "path": str(img_file), "description": desc})
                    print(f"  p{i} img{idx}: {w}x{h}px → {img_file.name}" + (" described" if desc else ""))

                pages.append(Page(
                    source=source,
                    page=i,
                    text=text,
                    tables=tables_md,
                    images=images,
                    source_type="text",
                ))
    finally:
        fitz_doc.close()

    return pages
