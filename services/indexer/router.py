"""Эвристика: определяет тип PDF (text / ocr) по наличию текстового слоя.

Запуск:
    python router.py data/raw/                 # список всех PDF с их типом
    python router.py data/raw/instr.pdf        # тип одного файла

Правило: если >90% страниц не содержат извлекаемого текста — это скан.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pdfplumber


def detect_pdf_type(pdf_path: Path, threshold: float = 0.9) -> str:
    """Возвращает 'text' или 'ocr'."""
    empty = 0
    total = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            total += 1
            txt = (page.extract_text() or "").strip()
            if len(txt) < 20:
                empty += 1
    if total == 0:
        return "text"
    return "ocr" if (empty / total) >= threshold else "text"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: router.py <path-to-pdf-or-dir>", file=sys.stderr)
        sys.exit(1)

    target = Path(sys.argv[1])
    pdfs = sorted(target.glob("*.pdf")) if target.is_dir() else [target]

    for p in pdfs:
        try:
            kind = detect_pdf_type(p)
            print(f"{kind:5s}  {p.name}")
        except Exception as e:
            print(f"ERR    {p.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
