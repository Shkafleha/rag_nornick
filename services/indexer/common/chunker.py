"""Чанкинг: берёт список Page → возвращает список Chunk.

TODO: портировать реальную логику из scripts/index_cen2.py.
Пока — минимальная заглушка: каждая страница = один чанк.
"""
from __future__ import annotations

from .schema import Chunk, Page


def chunk_pages(pages: list[Page], start_chunk_id: int = 0) -> list[Chunk]:
    chunks: list[Chunk] = []
    cid = start_chunk_id
    for p in pages:
        text = (p.text or "").strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=cid,
                text=text,
                source=p.source,
                source_type=p.source_type,
                page=p.page,
                type="text",
            )
        )
        cid += 1
        for tbl in p.tables:
            tbl = (tbl or "").strip()
            if not tbl:
                continue
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=tbl,
                    source=p.source,
                    source_type=p.source_type,
                    page=p.page,
                    type="table",
                )
            )
            cid += 1
    return chunks
