import json
import re
import sys

import pdfplumber
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

PDF_PATH = "data/raw/instr.pdf"
COLLECTION = "CEN2_all_pages"
OLLAMA_URL = "http://ollama:11434"
QDRANT_URL = "http://qdrant:6333"
EMBED_MODEL = "bge-m3:latest"
START_PAGE = 7
CHUNK_MIN_CHARS = 250
BATCH_SIZE = 10
EMBED_BATCH_SIZE = 32

HEADER_LINE_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+([А-ЯЁ][А-Яа-яЁё \-,]{3,100})$")
CHUNK_MAX_CHARS = 1500
OVERLAP_CHARS = 150


def words_to_lines(words, y_tol=3.0):
    if not words:
        return []
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines, cur, cur_top = [], [], words[0]["top"]
    for w in words:
        if abs(w["top"] - cur_top) <= y_tol:
            cur.append(w)
        else:
            cur = sorted(cur, key=lambda x: x["x0"])
            lines.append(" ".join(x["text"] for x in cur))
            cur = [w]
            cur_top = w["top"]
    cur = sorted(cur, key=lambda x: x["x0"])
    lines.append(" ".join(x["text"] for x in cur))
    return lines


def inside(w, bb, pad=1.5):
    x0, top, x1, bottom = bb
    cx = (w["x0"] + w["x1"]) / 2
    cy = (w["top"] + w["bottom"]) / 2
    return (x0 - pad) <= cx <= (x1 + pad) and (top - pad) <= cy <= (bottom + pad)


def clean(text):
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"Лист\s+\d+\s+Листов\s+\d+", "", text)
    text = re.sub(r"ТИ\s*305\.2-\d+-\d+", "", text)
    text = re.sub(r"Рисунок\s+\d+.*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path, start_page):
    all_text = []
    all_table_chunks = []
    header_stack = []  # [(level, "1.2 Title")]

    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        print(f"Всего страниц: {total}, парсим с {start_page} по {total - 1}")

        for page_idx in range(start_page, total):
            print(f"  -> page {page_idx}", flush=True)
            page = pdf.pages[page_idx]
            tables = page.find_tables()

            # Сначала текст — чтобы обновить стек заголовков до таблиц этой страницы
            page_text = ""
            words = page.extract_words()
            if words:
                table_bboxes = [t.bbox for t in tables] if tables else []
                text_words = [w for w in words if not any(inside(w, bb) for bb in table_bboxes)]
                if text_words:
                    lines = words_to_lines(text_words)
                    page_text = clean("\n".join(lines))
                    for line in page_text.split("\n"):
                        m = HEADER_LINE_RE.match(line.strip())
                        if m:
                            num, title = m.group(1), m.group(2).strip()
                            level = num.count(".") + 1
                            full = f"{num} {title}"
                            while header_stack and header_stack[-1][0] >= level:
                                header_stack.pop()
                            header_stack.append((level, full))

            cur_path = [h for _, h in header_stack]
            cur_breadcrumb = " > ".join(cur_path)
            cur_header = cur_path[-1] if cur_path else ""

            # Таблицы → один чанк на таблицу (markdown) с текущим заголовком
            for t in tables:
                rows = t.extract()
                if not rows:
                    continue
                # нормализуем ячейки и склеиваем continuation-строки (где первая ячейка пустая)
                norm = [[(str(c).strip().replace("\n", " ") if c else "") for c in r] for r in rows]
                merged = []
                for r in norm:
                    if merged and (not r[0]) and any(r):
                        merged[-1] = [
                            (a + " " + b).strip() if b else a
                            for a, b in zip(merged[-1], r + [""] * (len(merged[-1]) - len(r)))
                        ]
                    else:
                        merged.append(r)
                if len(merged) < 2:
                    continue
                ncols = max(len(r) for r in merged)
                merged = [r + [""] * (ncols - len(r)) for r in merged]
                header_row = merged[0]
                body = merged[1:]

                def render(rows_subset):
                    lines = ["| " + " | ".join(header_row) + " |",
                             "|" + "|".join(["---"] * ncols) + "|"]
                    for r in rows_subset:
                        lines.append("| " + " | ".join(r) + " |")
                    return "\n".join(lines)

                # дробим, если таблица большая, повторяя шапку
                chunk_rows, cur, cur_len = [], [], 0
                base_len = sum(len(c) for c in header_row) + ncols * 3
                for r in body:
                    rlen = sum(len(c) for c in r) + ncols * 3
                    if cur and cur_len + rlen + base_len > CHUNK_MAX_CHARS:
                        chunk_rows.append(cur)
                        cur, cur_len = [], 0
                    cur.append(r)
                    cur_len += rlen
                if cur:
                    chunk_rows.append(cur)

                for sub in chunk_rows:
                    text = render(sub)
                    if len(text.strip()) > 20:
                        all_table_chunks.append({
                            "text": text,
                            "type": "table",
                            "page": page_idx,
                            "header": cur_header,
                            "header_path": list(cur_path),
                            "header_breadcrumb": cur_breadcrumb,
                        })

            if page_text.strip():
                all_text.append(page_text)

            if (page_idx - start_page + 1) % 20 == 0:
                print(f"  Обработано страниц: {page_idx - start_page + 1}/{total - start_page}", flush=True)

    return all_text, all_table_chunks


def chunk_by_headers(text, min_chars=CHUNK_MIN_CHARS):
    """Режет текст по заголовкам, затем каждый раздел дополнительно
    дробит на куски до CHUNK_MAX_CHARS с overlap.
    Сохраняет иерархию заголовков (стек по номерам 1 / 1.2 / 1.2.3)."""
    print(f"  chunk_by_headers: split lines…", flush=True)
    lines = text.split("\n")
    print(f"  chunk_by_headers: {len(lines)} строк, поиск заголовков…", flush=True)

    # Группируем строки в секции и одновременно ведём стек заголовков
    raw_sections = []  # [(header_path, body_str)]
    cur_lines = []
    header_stack = []  # [(level:int, "1.2.3 Заголовок")]
    cur_path = []

    def flush():
        if cur_lines:
            body = "\n".join(cur_lines)
            raw_sections.append((list(cur_path), body))

    for line in lines:
        m = HEADER_LINE_RE.match(line.strip())
        if m:
            flush()
            num, title = m.group(1), m.group(2).strip()
            level = num.count(".") + 1
            full = f"{num} {title}"
            # pop всё, что не строго мельче нового уровня
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, full))
            cur_path = [h for _, h in header_stack]
            cur_lines = [line]
        else:
            cur_lines.append(line)
    flush()

    print(f"  chunk_by_headers: получено {len(raw_sections)} секций, фильтрация…", flush=True)
    sections = [(p, b.strip()) for p, b in raw_sections if len(b.strip()) >= min_chars]
    print(f"  chunk_by_headers: после фильтра {len(sections)} секций, дробление…", flush=True)

    chunks = []
    for header_path, section in sections:
        header = header_path[-1] if header_path else ""
        breadcrumb = " > ".join(header_path)

        prefix = f"[{breadcrumb}]\n" if breadcrumb else ""

        def make(body):
            return {
                "text": f"{prefix}{body}",
                "header": header,
                "header_path": list(header_path),
                "header_breadcrumb": breadcrumb,
            }

        if len(section) <= CHUNK_MAX_CHARS:
            chunks.append(make(section))
            continue

        # Раздел длиннее лимита — дробим с overlap
        start = 0
        while start < len(section):
            end = min(start + CHUNK_MAX_CHARS, len(section))
            if end < len(section):
                cut = section.rfind("\n\n", start + min_chars, end)
                if cut == -1:
                    cut = section.rfind(" ", start + min_chars, end)
                if cut > start:
                    end = cut
            piece = section[start:end].strip()
            if len(piece) >= min_chars:
                chunks.append(make(piece))
            if end >= len(section):
                break
            new_start = end - OVERLAP_CHARS
            start = max(new_start, start + 1)

    return chunks


def embed_batch(texts):
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=600,
    )
    r.raise_for_status()
    embs = r.json().get("embeddings")
    if not isinstance(embs, list) or len(embs) != len(texts):
        raise RuntimeError(f"Bad embed response: got {len(embs) if embs else 0}, expected {len(texts)}")
    return embs


def main():
    print("=== Парсинг PDF ===")
    all_text, table_chunks = parse_pdf(PDF_PATH, START_PAGE)

    print(f"Парсинг завершён. Текстовых страниц: {len(all_text)}, табличных строк: {len(table_chunks)}", flush=True)
    doc_text = "\n\n".join(p for p in all_text if p)
    print(f"Длина документа: {len(doc_text)} символов. Режу на чанки…", flush=True)
    text_chunks = chunk_by_headers(doc_text, CHUNK_MIN_CHARS)
    print(f"Текстовых чанков: {len(text_chunks)}", flush=True)
    print(f"Табличных чанков: {len(table_chunks)}", flush=True)

    # Объединяем все чанки
    all_chunks = (
        [
            {
                "text": c["text"],
                "header": c["header"],
                "header_path": c.get("header_path", []),
                "header_breadcrumb": c.get("header_breadcrumb", ""),
                "type": "text",
                "page": None,
            }
            for c in text_chunks
        ]
        + table_chunks
    )
    print(f"Итого чанков: {len(all_chunks)}")

    print("\n=== Эмбеддинги + загрузка в Qdrant ===", flush=True)
    qdrant = QdrantClient(url=QDRANT_URL)

    print("Пробный запрос к Ollama для определения размерности…", flush=True)
    sample_embs = embed_batch([all_chunks[0]["text"]])
    dim = len(sample_embs[0])
    print(f"Размерность: {dim}", flush=True)

    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
        print(f"Коллекция {COLLECTION} удалена")

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Коллекция {COLLECTION} создана")

    total = len(all_chunks)
    for batch_start in range(0, total, EMBED_BATCH_SIZE):
        batch = all_chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
        embs = embed_batch([c["text"] for c in batch])
        points = [
            PointStruct(
                id=batch_start + j,
                vector=embs[j],
                payload={
                    "text": chunk["text"],
                    "header": chunk.get("header", ""),
                    "header_path": chunk.get("header_path", []),
                    "header_breadcrumb": chunk.get("header_breadcrumb", ""),
                    "type": chunk["type"],
                    "page": chunk["page"],
                    "source": "instr.pdf",
                    "chunk_id": batch_start + j,
                },
            )
            for j, chunk in enumerate(batch)
        ]
        qdrant.upsert(collection_name=COLLECTION, points=points)
        print(f"  Загружено: {batch_start + len(batch)}/{total}", flush=True)

    print(f"\nГотово! Загружено {total} чанков в коллекцию {COLLECTION}")


if __name__ == "__main__":
    main()
