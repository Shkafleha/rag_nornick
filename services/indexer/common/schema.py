"""Общие схемы данных для text_pdf и image_pdf пайплайнов."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SourceType = Literal["text", "ocr"]
ChunkType = Literal["text", "table"]


@dataclass
class Page:
    """Одна страница PDF после парсинга (до чанкинга)."""
    source: str                 # имя файла, напр. "instr.pdf"
    page: int                   # 1-based
    text: str                   # извлечённый текст
    tables: list[str] = field(default_factory=list)  # таблицы в markdown
    source_type: SourceType = "text"


@dataclass
class Chunk:
    """Готовый к индексации чанк."""
    chunk_id: int
    text: str
    source: str
    source_type: SourceType
    page: int | None
    header: str = ""
    header_breadcrumb: str = ""
    type: ChunkType = "text"

    def to_payload(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "source_type": self.source_type,
            "page": self.page,
            "header": self.header,
            "header_breadcrumb": self.header_breadcrumb,
            "chunk_id": self.chunk_id,
            "type": self.type,
        }
