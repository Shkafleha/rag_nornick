"""PDF парсер для цифровых документов. Обёртка над pdfplumber.

TODO: портировать логику извлечения заголовков/breadcrumb из scripts/index_cen2.py.
Пока — минимальная версия: текст + таблицы со страницы.
"""
from __future__ import annotations

from pathlib import Path

import pdfplumber

from common.schema import Page


def parse_pdf(pdf_path: Path) -> list[Page]:
    pages: list[Page] = []
    source = pdf_path.name
    with pdfplumber.open(pdf_path) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            text = (p.extract_text() or "").strip()
            tables_md: list[str] = []
            for tbl in p.extract_tables() or []:
                rows = ["| " + " | ".join((c or "").strip() for c in row) + " |" for row in tbl]
                if rows:
                    sep = "|" + "|".join(["---"] * len(tbl[0])) + "|"
                    tables_md.append("\n".join([rows[0], sep, *rows[1:]]))
            pages.append(Page(source=source, page=i, text=text, tables=tables_md, source_type="text"))
    return pages
