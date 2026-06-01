"""LLM-фикс для OCR текста и таблиц.

Читает raw JSON из 01_extracted_pages/, прогоняет через LLM (qwen3 по умолчанию),
сохраняет очищенные page_NNN.json в 02_normalized_text/.

Env:
    OLLAMA_URL       default http://ollama:11434
    LLM_FIX_MODEL    default qwen3:8b
    EXTRACTED_DIR    где лежат raw JSON (default /data/01_extracted_pages)
    FIXED_OUT_DIR    куда сохранять результат (default /data/02_normalized_text)

Использование:
    python scripts/llm_fix_ocr.py CEN1_ТИ_3-48200234-05.1-12-2020_Очистка_электролита_от_примесей
    python scripts/llm_fix_ocr.py doc1 doc2 doc3
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
LLM_FIX_MODEL = os.getenv("LLM_FIX_MODEL", "qwen3:8b")

# Auto-detect paths: Docker (/data) или Windows (c:/Users/...)
if Path("/data").exists():
    _base = Path("/data")
else:
    # Windows: найти data/ рядом со скриптом
    _script_dir = Path(__file__).parent.parent  # scripts/.. = project root
    _base = _script_dir / "data"

EXTRACTED_BASE = Path(os.getenv("EXTRACTED_DIR", _base / "01_extracted_pages"))
NORM_OUT_BASE = Path(os.getenv("FIXED_OUT_DIR", _base / "02_normalized_text"))


def _build_ocr_tokens(res: dict) -> List[Tuple]:
    """Извлекает OCR-токены: список (text, x1, y1, x2, y2)."""
    ocr = res.get("overall_ocr_res") or res.get("ocr_res") or {}
    texts = ocr.get("rec_texts") or []
    boxes = ocr.get("rec_boxes") or ocr.get("rec_polys") or []
    tokens = []
    for i, text in enumerate(texts):
        if i < len(boxes) and len(boxes[i]) >= 4:
            x1, y1, x2, y2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            tokens.append((text, x1, y1, x2, y2))
    return tokens


def _reconstruct_text(bbox: list, tokens: List[Tuple], y_tol: int = 12) -> str:
    """Реконструирует текст блока из OCR-токенов по bbox, избегая слипания слов."""
    if not bbox or len(bbox) < 4 or not tokens:
        return ""
    bx1, by1, bx2, by2 = bbox
    inside = []
    for text, tx1, ty1, tx2, ty2 in tokens:
        cx = (tx1 + tx2) / 2
        cy = (ty1 + ty2) / 2
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            inside.append((cy, tx1, text))
    if not inside:
        return ""
    inside.sort(key=lambda t: (t[0], t[1]))
    lines: list = [[inside[0]]]
    for token in inside[1:]:
        if abs(token[0] - lines[-1][-1][0]) <= y_tol:
            lines[-1].append(token)
        else:
            lines.append([token])
    return "\n".join(
        " ".join(t[2] for t in sorted(line, key=lambda t: t[1])) for line in lines
    )


def _llm_fix(text: str) -> str:
    """Исправляет OCR-текст через LLM."""
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
            timeout=300,
        )
        fixed = resp.json().get("message", {}).get("content", "")
        return fixed.strip() if fixed.strip() else text
    except Exception as e:
        print(f"  [warn] LLM fix failed: {e}")
        return text


def _html_table_to_markdown(html: str) -> str:
    """Конвертит HTML таблицу в markdown."""
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


def fix_document(doc_name: str, batch_fix: bool = True, chunk_size: int = 0) -> None:
    """Прогоняет все страницы документа через LLM-фикс.

    Args:
        doc_name: имя папки документа
        batch_fix: если True — склеить текст и отправить одним/несколькими запросами,
                   если False — фиксить каждую страницу отдельно
        chunk_size: если > 0 — обрабатывать по N страниц за раз (для batch_fix=True)
                    если 0 — весь документ одним запросом
    """
    extracted_dir = EXTRACTED_BASE / doc_name
    norm_dir = NORM_OUT_BASE / doc_name

    if not extracted_dir.exists():
        print(f"[error] {extracted_dir} не найдена")
        return

    norm_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted(extracted_dir.glob("page_*.json"))
    if not pages:
        print(f"[error] Нет page_*.json в {extracted_dir}")
        return

    print(f"Документ: {doc_name}")
    print(f"Страниц: {len(pages)}")
    print(f"Режим: {'batch (весь текст одним запросом)' if batch_fix else 'per-page'}")

    # Предварительная загрузка всех страниц
    pages_data = []
    for page_file in pages:
        try:
            data = json.loads(page_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [error] {page_file.name}: {e}")
            continue

        # Сырые данные — список результатов PP-StructureV3
        if isinstance(data, list) and data:
            res = data[0].get("res", data[0]) if isinstance(data[0], dict) else data[0]
        else:
            res = data.get("res", data) if isinstance(data, dict) else data

        if not isinstance(res, dict):
            print(f"  [error] {page_file.name}: неожиданный формат res")
            continue

        # Извлекаем текст — реконструируем из OCR-токенов, чтобы избежать
        # слипания слов в block_content (баг PP-StructureV3 на многоколоночных блоках)
        ocr_tokens = _build_ocr_tokens(res)
        parsing = res.get("parsing_res_list") or []
        text_blocks = []
        for block in parsing:
            label = block.get("block_label", "")
            if label in ("figure", "image", "figure_caption", "figure_title"):
                continue
            if label in ("header", "footer", "number", "page_number"):
                continue
            bbox = block.get("block_bbox", [])
            if ocr_tokens and bbox and len(bbox) >= 4:
                content = _reconstruct_text(bbox, ocr_tokens)
            else:
                content = block.get("block_content", "").strip()
            if content:
                text_blocks.append(content)

        text = "\n\n".join(text_blocks).strip()

        # Извлекаем таблицы (как в ocr.py)
        tables = []
        for t in res.get("table_res_list", []) or []:
            html = t.get("pred_html") or t.get("html")
            if html:
                tables.append(_html_table_to_markdown(html))

        page_num = int(page_file.stem.split("_")[1])
        pages_data.append({
            "page_num": page_num,
            "text": text,
            "tables": tables,
        })

    if batch_fix:
        # Режим batch: обрабатываем по чанкам или весь документ целиком
        if chunk_size <= 0:
            chunks = [pages_data]  # Весь документ одним запросом
        else:
            chunks = [pages_data[i:i+chunk_size] for i in range(0, len(pages_data), chunk_size)]

        print(f"\nОбработка {len(chunks)} чанк(ов)...")

        for chunk_idx, chunk_pages in enumerate(chunks, 1):
            # Обёртываем каждую страницу в маркеры (нижний регистр для LLM)
            marked_text = ""
            for p in chunk_pages:
                if p["text"]:
                    marked_text += f"[page_{p['page_num']:03d}_start]\n{p['text']}\n[page_{p['page_num']:03d}_end]\n\n"

            if marked_text:
                print(f"\nЧанк {chunk_idx}: отправка {len(marked_text)} символов в LLM...")
                fixed_text = _llm_fix(marked_text)

                # Парсим обратно по маркерам (case-insensitive, гибкие пробелы и переносы)
                import re
                # Гибкий паттерн: пробелы/переносы вокруг маркеров, разные регистры
                page_pattern = r"\[\s*page_(\d{3})_start\s*\](.*?)\[\s*page_\1_end\s*\]"
                matches = re.findall(page_pattern, fixed_text, re.DOTALL | re.IGNORECASE)

                # Создаём словарь исправленного текста по номерам страниц
                fixed_by_page = {int(m[0]): m[1].strip() for m in matches}

                for page_data in chunk_pages:
                    page_num = page_data["page_num"]
                    text = fixed_by_page.get(page_num, page_data["text"])  # Fallback на исходный если не нашли
                    tables = page_data["tables"]

                    # Фиксим таблицы отдельно (они короче)
                    tables = [_llm_fix(t) for t in tables if t.strip()]

                    out_file = norm_dir / f"page_{page_num:03d}.json"
                    out_file.write_text(
                        json.dumps(
                            {"page": page_num, "text": text, "tables": tables},
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    print(f"  page {page_num}: {len(text)} chars, {len(tables)} tables")
    else:
        # Режим 2: каждая страница отдельно
        for page_data in pages_data:
            page_num = page_data["page_num"]
            text = page_data["text"]
            tables = page_data["tables"]

            if text:
                text = _llm_fix(text)
            tables = [_llm_fix(t) for t in tables if t.strip()]

            out_file = norm_dir / f"page_{page_num:03d}.json"
            out_file.write_text(
                json.dumps(
                    {"page": page_num, "text": text, "tables": tables},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"  page {page_num}: {len(text)} chars, {len(tables)} tables")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("doc_names", nargs="+", help="Имена папок документов")
    parser.add_argument("--batch", action="store_true", default=True,
                        help="Batch режим (default)")
    parser.add_argument("--per-page", action="store_true",
                        help="Фиксить каждую страницу отдельно")
    parser.add_argument("--chunk-size", type=int, default=10,
                        help="Обрабатывать по N страниц за раз в batch режиме (default 10)")
    args = parser.parse_args()

    batch_mode = not args.per_page
    for doc_name in args.doc_names:
        fix_document(doc_name, batch_fix=batch_mode, chunk_size=args.chunk_size if batch_mode else 0)
