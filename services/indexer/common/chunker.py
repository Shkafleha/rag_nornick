"""Чанкинг: берёт список Page → возвращает список Chunk.

Текст режется по заголовкам вида «1.2.3 ЗАГОЛОВОК» (русский, заглавная первая буква).
Таблицы и изображения становятся отдельными чанками с breadcrumb текущего раздела.
"""
from __future__ import annotations

import re

from .schema import Chunk, Page

HEADER_RE = re.compile(r"^(?:#+\s+)?(\d+(?:\.\d+)*)\s+([А-ЯЁ][А-Яа-яЁё \-,]{3,100})\.?$")
CHUNK_MAX_CHARS = 1500
CHUNK_MIN_CHARS = 250
OVERLAP_CHARS = 150


def _chunk_by_headers(text: str) -> list[dict]:
    """Разбивает текст на секции по заголовкам, длинные секции дробит с overlap."""
    lines = text.split("\n")
    raw_sections: list[tuple[list[str], str]] = []
    cur_lines: list[str] = []
    header_stack: list[tuple[int, str]] = []
    cur_path: list[str] = []

    def flush() -> None:
        if cur_lines:
            raw_sections.append((list(cur_path), "\n".join(cur_lines)))

    for line in lines:
        m = HEADER_RE.match(line.strip())
        if m:
            flush()
            num, title = m.group(1), m.group(2).strip()
            level = num.count(".") + 1
            full = f"{num} {title}"
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, full))
            cur_path = [h for _, h in header_stack]
            cur_lines = [line]
        else:
            cur_lines.append(line)
    flush()

    sections = [(p, b.strip()) for p, b in raw_sections if len(b.strip()) >= CHUNK_MIN_CHARS]

    chunks: list[dict] = []
    for header_path, section in sections:
        header = header_path[-1] if header_path else ""
        breadcrumb = " > ".join(header_path)
        prefix = f"[{breadcrumb}]\n" if breadcrumb else ""

        if len(section) <= CHUNK_MAX_CHARS:
            chunks.append({"text": f"{prefix}{section}", "header": header, "header_breadcrumb": breadcrumb})
            continue

        start = 0
        while start < len(section):
            end = min(start + CHUNK_MAX_CHARS, len(section))
            if end < len(section):
                cut = section.rfind("\n\n", start + CHUNK_MIN_CHARS, end)
                if cut == -1:
                    cut = section.rfind(" ", start + CHUNK_MIN_CHARS, end)
                if cut > start:
                    end = cut
            piece = section[start:end].strip()
            if len(piece) >= CHUNK_MIN_CHARS:
                chunks.append({"text": f"{prefix}{piece}", "header": header, "header_breadcrumb": breadcrumb})
            if end >= len(section):
                break
            start = max(end - OVERLAP_CHARS, start + 1)

    return chunks


def chunk_pages(pages: list[Page], start_chunk_id: int = 0) -> list[Chunk]:
    if not pages:
        return []

    source = pages[0].source
    source_type = pages[0].source_type

    # Pass 1: header-контекст на каждой странице (нужен для таблиц и картинок)
    header_stack: list[tuple[int, str]] = []
    page_header: dict[int, tuple[str, str]] = {}  # page → (header, breadcrumb)

    for p in pages:
        for line in (p.text or "").split("\n"):
            m = HEADER_RE.match(line.strip())
            if m:
                num, title = m.group(1), m.group(2).strip()
                level = num.count(".") + 1
                full = f"{num} {title}"
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, full))
        cur_path = [h for _, h in header_stack]
        page_header[p.page] = (
            cur_path[-1] if cur_path else "",
            " > ".join(cur_path),
        )

    # Pass 2: текстовые чанки по заголовкам (весь документ сразу)
    full_text = "\n\n".join((p.text or "").strip() for p in pages if (p.text or "").strip())
    text_sections = _chunk_by_headers(full_text)

    cid = start_chunk_id
    chunks: list[Chunk] = []

    for sec in text_sections:
        chunks.append(Chunk(
            chunk_id=cid,
            text=sec["text"],
            source=source,
            source_type=source_type,
            page=None,
            header=sec["header"],
            header_breadcrumb=sec["header_breadcrumb"],
            type="text",
        ))
        cid += 1

    # Pass 3: табличные чанки (с header-контекстом страницы)
    for p in pages:
        h, bc = page_header.get(p.page, ("", ""))
        for tbl in p.tables:
            tbl = (tbl or "").strip()
            if not tbl:
                continue
            chunks.append(Chunk(
                chunk_id=cid,
                text=tbl,
                source=source,
                source_type=source_type,
                page=p.page,
                header=h,
                header_breadcrumb=bc,
                type="table",
            ))
            cid += 1

    # Pass 4: чанки картинок (с header-контекстом страницы)
    for p in pages:
        h, bc = page_header.get(p.page, ("", ""))
        for img in (p.images or []):
            desc = (img.get("description") or "").strip()
            if not desc:
                continue
            chunks.append(Chunk(
                chunk_id=cid,
                text=desc,
                source=source,
                source_type=source_type,
                page=p.page,
                header=h,
                header_breadcrumb=bc,
                type="image",
            ))
            cid += 1

    return chunks
