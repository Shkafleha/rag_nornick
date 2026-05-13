# indexer/

Offline-индексатор: `data/raw/*.pdf` → Qdrant коллекция `docs_v1`.

Два пайплайна под разные типы документов:

| Пайплайн | Для чего | Стек |
|---|---|---|
| `text_pdf/` | Цифровые PDF с текстовым слоем | pdfplumber, pymupdf |
| `image_pdf/` | Сканы, где текст — растровая картинка | Surya OCR (PyTorch, GPU) |

Каждый — **отдельный сервис** в `docker-compose.yml` под профилем `indexing` (не поднимается по умолчанию):

```bash
# Индексировать все цифровые PDF
docker compose --profile indexing run --rm indexer_text

# Индексировать все сканы
docker compose --profile indexing run --rm indexer_image

# Оба сразу
docker compose --profile indexing run --rm indexer_text && \
docker compose --profile indexing run --rm indexer_image
```

## Роутинг: text vs image

`router.py` определяет тип PDF по эвристике: для каждой страницы спрашивает у `pdfplumber`, есть ли текстовый слой. Если >90% страниц без текста → скан → идёт в `image_pdf`. Иначе → `text_pdf`.

Пока роутер **не вызывается автоматически**: каждый сервис получает список файлов через env-переменную или сам фильтрует `data/raw/`. Автороутинг — TODO.

## Коллекция и payload

**Одна общая коллекция** `docs_v1`. Каждый чанк имеет:

```python
{
    "text": str,
    "source": str,          # "instr.pdf"
    "source_type": str,     # "text" | "ocr"
    "page": int,
    "header": str,
    "header_breadcrumb": str,
    "chunk_id": int,
    "type": str,            # "text" | "table"
}
```

`rag_api` фильтрует по `source` / `source_type` через Qdrant payload filter.

## Структура

```
services/indexer/
├── common/                ← общий код для обоих пайплайнов
│   ├── chunker.py              чанкинг (одинаковый для text и OCR)
│   ├── embedder.py             обёртка над Ollama embed
│   ├── qdrant_loader.py        загрузка в Qdrant
│   └── schema.py               Chunk, Page, Config
│
├── text_pdf/
│   ├── Dockerfile              лёгкий образ без CUDA
│   ├── requirements.txt
│   ├── build_index.py          точка входа
│   └── parser.py               pdfplumber wrapper
│
├── image_pdf/
│   ├── Dockerfile              тяжёлый образ с CUDA + Surya
│   ├── requirements.txt
│   ├── build_index.py
│   └── ocr.py                  Surya wrapper
│
└── router.py              эвристика text vs image (TODO)
```

## Статус

- [x] Скелет директорий и README
- [ ] `common/` — портировать логику чанкинга из `scripts/index_cen2.py`
- [ ] `text_pdf/build_index.py` — рабочая версия для цифровых PDF
- [ ] `image_pdf/build_index.py` — Surya OCR для сканов
- [ ] `router.py` — авто-детект
- [ ] Добавить сервисы в `docker-compose.yml`
- [ ] Убрать `scripts/index_cen2.py` и `scripts/rag_simple.py`
