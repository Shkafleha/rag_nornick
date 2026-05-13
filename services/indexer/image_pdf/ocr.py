"""OCR для сканированных PDF через PaddleX PP-StructureV3.

Стратегия: рендерим каждую страницу в PNG (matrix=2.0), прогоняем через
pipeline, складываем сырой результат в JSON для последующего разбора в
ноутбуках, а в Page кладём весь распознанный текст подряд + таблицы.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import fitz  # PyMuPDF
import requests
from PIL import Image

from common.schema import Page

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
LLM_FIX_MODEL = os.getenv("LLM_FIX_MODEL", "qwen3:8b")
LLM_FIX_ENABLED = os.getenv("LLM_FIX", "0") == "1"

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from paddleocr import PPStructureV3
        _pipeline = PPStructureV3(
            lang="ru",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_seal_recognition=False,
        )
    return _pipeline


def _render_page(page: fitz.Page, out_path: Path, zoom: float = 1.0) -> None:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save(out_path)


def _parsed(item):
    if hasattr(item, "json"):
        return item.json
    if hasattr(item, "res"):
        return item.res
    return item


def _extract_text_and_tables(parsed_items: list[dict]) -> tuple[str, list[str]]:
    """Достаём текст и таблицы из сырого ответа PP-StructureV3.

    Используем parsing_res_list — готовые блоки с block_order (порядок чтения),
    block_label и block_content.
    """
    blocks: list[dict] = []
    tables: list[str] = []

    for item in parsed_items:
        res = item.get("res", item) if isinstance(item, dict) else item
        if not isinstance(res, dict):
            continue

        # Таблицы отдельно
        for t in res.get("table_res_list", []) or []:
            html = t.get("pred_html") or t.get("html")
            if html:
                tables.append(html)

        # parsing_res_list — основной источник текста
        _SKIP_LABELS = {"header", "footer", "number", "page_number"}
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
            # Заголовки → markdown-формат для чанкера
            if "title" in label:
                content = f"## {content}"
            blocks.append({
                "order": order if order is not None else 9999,
                "y": bbox[1] if len(bbox) >= 2 else 0,
                "label": label,
                "text": content,
            })

        # Фолбэк: если parsing_res_list пуст — берём overall_ocr_res
        if not parsing:
            ocr = res.get("overall_ocr_res") or res.get("ocr_res") or {}
            rec_texts = ocr.get("rec_texts") or []
            if rec_texts:
                blocks.append({"order": 0, "y": 0, "label": "ocr", "text": "\n".join(rec_texts)})

    # Сортируем по block_order, фолбэк по y-координатe
    blocks.sort(key=lambda b: (b["order"], b["y"]))

    text_parts = [b["text"] for b in blocks]
    return "\n\n".join(text_parts).strip(), tables


def _html_table_to_markdown(html: str) -> str:
    """Конвертирует HTML-таблицу в markdown."""
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
    """Исправляет ошибки OCR через LLM. Возвращает исходный текст при ошибке."""
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
            timeout=120,
        )
        fixed = resp.json().get("message", {}).get("content", "")
        return fixed.strip() if fixed.strip() else text
    except Exception as e:
        print(f"  [warn] LLM fix failed: {e}")
        return text


def ocr_pdf(pdf_path: Path) -> list[Page]:
    """Распознаёт сканированный PDF постранично."""
    out_dir = Path(os.getenv("OCR_OUT_DIR", "/data/01_extracted_pages")) / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _get_pipeline()
    pages: list[Page] = []

    doc = fitz.open(pdf_path)
    try:
        for i in range(len(doc)):
            page_num = i + 1
            img_path = out_dir / f"page_{page_num:03d}.png"
            _render_page(doc[i], img_path)

            result = pipeline.predict(str(img_path))
            parsed_items = [_parsed(it) for it in result]

            # Сохраняем сырой JSON для проверки в ноутбуках
            try:
                (out_dir / f"page_{page_num:03d}.json").write_text(
                    json.dumps(parsed_items, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"  [warn] dump json page {page_num}: {e}")

            text, tables = _extract_text_and_tables(parsed_items)

            # HTML → Markdown для таблиц
            tables = [_html_table_to_markdown(t) for t in tables]

            if LLM_FIX_ENABLED:
                if text:
                    text = _llm_fix(text)
                tables = [_llm_fix(t) for t in tables if t.strip()]
                print(f"  page {page_num}: {len(text)} chars (LLM fixed), {len(tables)} tables")
            else:
                print(f"  page {page_num}: {len(text)} chars, {len(tables)} tables")

            pages.append(
                Page(
                    source=pdf_path.name,
                    page=page_num,
                    text=text,
                    tables=tables,
                    source_type="ocr",
                )
            )
    finally:
        doc.close()

    return pages
