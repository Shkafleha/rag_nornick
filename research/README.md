# research/

Экспериментальная зона. Никакого прод-runtime — только исследования, бенчмарки и offline-эвалы.

## Принципы

1. **Research пишет файлы → prod их читает.** Артефакты экспериментов кладём в `data/experiments/<date>_<tag>/`. Прод (`services/rag_api`) подхватывает только зафиксированные версии из `data/processed/`.
2. **Каждый блок самодостаточный** — свой README, свои скрипты. Можно открыть и разобраться без остального проекта.
3. **Один golden dataset на всё** — `07_e2e_eval/datasets/golden_qa.jsonl`. Любой эксперимент сравнивается с ним.
4. **Tracking:** MLflow для офлайн-метрик ретривера/эмбеддеров, Langfuse — для онлайн-трейсов прода.

## Блоки

| Папка | Что внутри |
|---|---|
| `01_parsing/` | OCR и извлечение текста из PDF: pymupdf, unstructured, docling, marker |
| `02_chunking/` | Стратегии чанкинга: fixed, semantic, hierarchical, table-aware |
| `03_embeddings/` | Подбор embedder'а: bge-m3, e5-large, mxbai, nomic |
| `04_retrieval/` | Dense / BM25 / hybrid (RRF) / HyDE / multi-query |
| `05_reranking/` | Cross-encoders: bge-reranker-v2-m3, mxbai-rerank, jina |
| `06_llm/` | Бенчмарк LLM из Ollama: качество, скорость, VRAM |
| `07_e2e_eval/` | End-to-end оценка пайплайна через RAGAS, golden_qa, отчёты |

## Workflow эксперимента

1. Берёшь блок (например, `03_embeddings`).
2. Запускаешь скрипт — он логирует метрики в MLflow + кладёт артефакт в `data/experiments/<date>_<tag>/`.
3. Сравниваешь с предыдущим лучшим.
4. Если стало лучше — обновляешь прод: указываешь новый артефакт в `services/rag_api` env-переменных.
