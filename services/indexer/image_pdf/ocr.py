"""OCR для сканированных PDF через PaddleX PP-StructureV3.

Стратегия: рендерим каждую страницу в PNG, прогоняем через pipeline.
Рисунки детектируются PP-StructureV3 (label=figure), кропаются и сохраняются
на диск. Подписи (figure_caption) используются как текст чанка.
Vision LLM (VISION_DESCRIBE=1) добавляет описание поверх подписи — опционально.

Env:
    VISION_DESCRIBE  "1" включает LLM-описание рисунков (default "0")
    VISION_MODEL     модель Ollama с vision (default "llava:7b")
    IMG_MIN_PX       минимальный размер стороны кропа в пикселях (default 80)
    OLLAMA_URL       default http://ollama:11434
    LLM_FIX          "1" для LLM-коррекции OCR-текста (default "0")
    LLM_FIX_MODEL    default qwen3:8b
"""
from __future__ import annotations

import base64
import io
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
VISION_MODEL = os.getenv("VISION_MODEL", "llava:7b")
VISION_ENABLED = os.getenv("VISION_DESCRIBE", "0") == "1"
IMG_MIN_PX = int(os.getenv("IMG_MIN_PX", "80"))
NORM_OUT_BASE = Path(os.getenv("FIXED_OUT_DIR", "/data/02_normalized_text"))

_VISION_PROMPT = (
    "/no_think\n"
    "Ты описываешь рисунок из технической инструкции на русском языке. "
    "Опиши кратко, что изображено: схема, диаграмма, фотография, чертёж, "
    "и какие ключевые элементы или подписи видны. "
    "Ответ только на русском языке, одним абзацем, без списков."
)

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


def _inside_figure(bbox: list, raw_figs: list[dict], overlap_thresh: float = 0.6) -> bool:
    """True если bbox блока перекрывается с figure-регионом больше чем на overlap_thresh."""
    if not bbox or len(bbox) < 4:
        return False
    bx1, by1, bx2, by2 = bbox
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    if b_area == 0:
        return False
    for fig in raw_figs:
        if fig.get("type") != "figure":
            continue
        fb = fig.get("bbox") or []
        if len(fb) < 4:
            continue
        fx1, fy1, fx2, fy2 = fb
        ix1, iy1 = max(bx1, fx1), max(by1, fy1)
        ix2, iy2 = min(bx2, fx2), min(by2, fy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter / b_area >= overlap_thresh:
            return True
    return False


def _build_ocr_tokens(res: dict) -> list[tuple]:
    """Извлекает OCR-токены из overall_ocr_res: список (text, x1, y1, x2, y2)."""
    ocr = res.get("overall_ocr_res") or res.get("ocr_res") or {}
    texts = ocr.get("rec_texts") or []
    boxes = ocr.get("rec_boxes") or ocr.get("rec_polys") or []
    tokens = []
    for i, text in enumerate(texts):
        if i < len(boxes) and len(boxes[i]) >= 4:
            x1, y1, x2, y2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            tokens.append((text, x1, y1, x2, y2))
    return tokens


def _reconstruct_text(bbox: list, tokens: list[tuple], y_tol: int = 12) -> str:
    """Реконструирует текст блока из OCR-токенов внутри bbox.

    Токены группируются по строкам (близкие Y), внутри строки сортируются по X
    и соединяются пробелом. PP-StructureV3 теряет пробелы при сборке block_content
    в многоколоночных блоках — этот метод восстанавливает правильные пробелы.
    """
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
    lines: list[list] = [[inside[0]]]
    for token in inside[1:]:
        if abs(token[0] - lines[-1][-1][0]) <= y_tol:
            lines[-1].append(token)
        else:
            lines.append([token])
    result_lines = [" ".join(t[2] for t in sorted(line, key=lambda t: t[1])) for line in lines]
    return "\n".join(result_lines)


def _extract_content(parsed_items: list[dict]) -> tuple[str, list[str], list[dict]]:
    """Достаём текст, таблицы и рисунки из сырого ответа PP-StructureV3.

    Returns: (text, tables_markdown, figures)
    Каждый figure: {"bbox": [x0, y0, x1, y1], "caption": str, "order": int}
    """
    text_blocks: list[dict] = []
    tables: list[str] = []
    raw_figs: list[dict] = []   # figure и figure_caption блоки вперемешку

    _SKIP_LABELS = {"header", "footer", "number", "page_number"}

    for item in parsed_items:
        res = item.get("res", item) if isinstance(item, dict) else item
        if not isinstance(res, dict):
            continue

        for t in res.get("table_res_list", []) or []:
            html = t.get("pred_html") or t.get("html")
            if html:
                tables.append(html)

        ocr_tokens = _build_ocr_tokens(res)

        parsing = res.get("parsing_res_list") or []
        for block in parsing:
            label = block.get("block_label", "")
            bbox = block.get("block_bbox", [0, 0, 0, 0])
            raw_order = block.get("block_order")
            order = raw_order if raw_order is not None else 9999

            if label in ("figure", "image"):
                raw_figs.append({"order": order, "bbox": bbox, "type": "figure", "text": ""})
                continue

            # Реконструируем текст из OCR-токенов, чтобы избежать слипания слов
            # в block_content (баг PP-StructureV3 на многоколоночных блоках)
            if ocr_tokens and bbox and len(bbox) >= 4:
                content = _reconstruct_text(bbox, ocr_tokens)
            else:
                content = block.get("block_content", "").strip()

            if label in ("figure_caption", "figure_title"):
                raw_figs.append({"order": order, "bbox": bbox, "type": "caption", "text": content})
                continue

            if not content or label in _SKIP_LABELS:
                continue
            # Пропускаем текст внутри уже найденных figure-регионов
            if _inside_figure(bbox, raw_figs):
                continue
            if "title" in label:
                content = f"## {content}"
            text_blocks.append({
                "order": order,
                "y": (bbox[1] or 0) if len(bbox) >= 2 else 0,
                "text": content,
            })

        if not parsing:
            ocr = res.get("overall_ocr_res") or res.get("ocr_res") or {}
            rec_texts = ocr.get("rec_texts") or []
            if rec_texts:
                text_blocks.append({"order": 0, "y": 0, "text": "\n".join(rec_texts)})

    text_blocks.sort(key=lambda b: (b["order"] or 9999, b["y"] or 0))
    text = "\n\n".join(b["text"] for b in text_blocks).strip()

    # Матчим рисунки с подписями: идём по порядку, caption сразу после figure
    raw_figs.sort(key=lambda f: f["order"] or 9999)
    figures: list[dict] = []
    i = 0
    while i < len(raw_figs):
        if raw_figs[i]["type"] == "figure":
            caption = ""
            if i + 1 < len(raw_figs) and raw_figs[i + 1]["type"] == "caption":
                caption = raw_figs[i + 1]["text"]
                i += 2
            else:
                i += 1
            figures.append({"bbox": raw_figs[i - 1]["bbox"] if caption else raw_figs[i - 1]["bbox"],
                             "caption": caption,
                             "order": raw_figs[i - (2 if caption else 1)]["order"]})
        else:
            # caption без figure (перед рисунком) — запоминаем, применим к следующему
            pending_caption = raw_figs[i]["text"]
            if i + 1 < len(raw_figs) and raw_figs[i + 1]["type"] == "figure":
                figures.append({"bbox": raw_figs[i + 1]["bbox"],
                                 "caption": pending_caption,
                                 "order": raw_figs[i]["order"]})
                i += 2
            else:
                i += 1

    return text, tables, figures


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


def _describe_image_llm(img_bytes: bytes, source: str, page: int, idx: int) -> str:
    """Vision LLM описание — используется только если VISION_DESCRIBE=1."""
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
        print(f"  [warn] vision LLM {source} p{page} fig{idx}: {e}")
        return ""


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

            try:
                (out_dir / f"page_{page_num:03d}.json").write_text(
                    json.dumps(parsed_items, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"  [warn] dump json page {page_num}: {e}")

            text, tables, figures = _extract_content(parsed_items)
            tables = [_html_table_to_markdown(t) for t in tables]

            # Кропаем рисунки, сохраняем на диск, формируем описание
            images: list[dict] = []
            if figures:
                page_img = Image.open(img_path)
                for fig_idx, fig in enumerate(figures):
                    bbox = fig["bbox"]
                    if not bbox or len(bbox) < 4:
                        continue
                    try:
                        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        crop = page_img.crop((x0, y0, x1, y1))
                        if crop.width < IMG_MIN_PX or crop.height < IMG_MIN_PX:
                            continue

                        fig_path = out_dir / f"page_{page_num:03d}_fig{fig_idx:02d}.png"
                        crop.save(fig_path)

                        caption = fig.get("caption", "").strip()

                        # Vision LLM — опционально поверх подписи
                        llm_desc = ""
                        if VISION_ENABLED:
                            buf = io.BytesIO()
                            crop.save(buf, format="PNG")
                            llm_desc = _describe_image_llm(buf.getvalue(), pdf_path.name, page_num, fig_idx)

                        description = caption
                        if llm_desc:
                            description = f"{caption}\n{llm_desc}".strip() if caption else llm_desc
                        if not description:
                            description = f"Рисунок (стр. {page_num})"

                        images.append({
                            "description": description,
                            "caption": caption,
                            "path": str(fig_path),
                            "bbox": [x0, y0, x1, y1],
                        })
                        print(f"  p{page_num} fig{fig_idx}: {crop.width}x{crop.height}px → {fig_path.name}"
                              + (f" [{caption[:40]}]" if caption else ""))
                    except Exception as e:
                        print(f"  [warn] figure crop p{page_num} fig{fig_idx}: {e}")

            if LLM_FIX_ENABLED:
                if text:
                    text = _llm_fix(text)
                tables = [_llm_fix(t) for t in tables if t.strip()]

            norm_dir = NORM_OUT_BASE / pdf_path.stem
            norm_dir.mkdir(parents=True, exist_ok=True)
            (norm_dir / f"page_{page_num:03d}.json").write_text(
                json.dumps({"page": page_num, "text": text, "tables": tables, "images": images},
                           ensure_ascii=False),
                encoding="utf-8",
            )

            print(f"  page {page_num}: {len(text)} chars, {len(tables)} tables, {len(images)} figures")

            pages.append(Page(
                source=pdf_path.name,
                page=page_num,
                text=text,
                tables=tables,
                images=images,
                source_type="ocr",
            ))
    finally:
        doc.close()

    return pages
