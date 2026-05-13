# 01_parsing — извлечение текста из PDF

## Цель
Получить чистый структурированный текст из исходного PDF (`data/raw/instr.pdf`) с максимальным сохранением:
- структуры заголовков (для `header_breadcrumb`)
- таблиц (сейчас в индексе много пустых таблиц `| | |` — это симптом плохого парсинга)
- многоколоночных страниц
- сносок и подписей

## Кандидаты

| Парсер | Сильные стороны | Слабые |
|---|---|---|
| `pymupdf` (fitz) | Быстрый, точные координаты | Слабо с таблицами |
| `unstructured` | Универсальный, есть OCR fallback | Медленный, тяжёлый |
| `docling` (IBM) | Отлично с таблицами, layout-aware | Новый, может ломаться |
| `marker` | LLM-aware, структурный markdown | Требует GPU |
| `pdfplumber` | Хорошо с таблицами | Только цифровые PDF |
| `tesseract` / `paddleocr` / `surya` | Для сканов | Медленнее |

## Метрика качества

Два уровня:

**1. Прямой (трудоёмкий, точный)** — для топ-10 страниц вручную размечен "идеальный текст". Считаем CER (Character Error Rate) и table-F1.

**2. Косвенный (быстрый)** — берём `golden_qa.jsonl`, прогоняем весь RAG-пайплайн на каждой версии парсинга, смотрим answer accuracy. Лучший парсер = лучший финальный ответ.

## Артефакты

```
data/experiments/<date>_<parser>/
├── pages.parquet      # колонки: page, type, header, text, bbox
├── chunks.jsonl       # после чанкинга (это уже задача 02_chunking)
├── stats.json         # n_pages, n_tables, avg_chunk_len, ...
└── meta.yaml          # версия парсера, параметры, hash исходника
```

## TODO

- [ ] Скрипт `parse.py --parser docling --input data/raw/instr.pdf --out data/experiments/`
- [ ] Базовая разметка 10 страниц → `eval/golden_pages/`
- [ ] Скрипт сравнения парсеров `compare.py`
