# 03_embeddings — подбор embedder'а

## Цель
Найти embedding-модель, которая лучше всего работает **на твоих документах**, а не на MTEB.

## Кандидаты

| Модель | Источник | Размер | Языки |
|---|---|---|---|
| `bge-m3` | Ollama / HF | 1024 | Мультиязычная (текущая прод) |
| `mxbai-embed-large` | Ollama | 1024 | EN |
| `nomic-embed-text` | Ollama | 768 | EN |
| `intfloat/multilingual-e5-large` | HF (embeddings_service) | 1024 | Мультиязычная |
| `intfloat/multilingual-e5-base` | HF | 768 | Мультиязычная |
| `paraphrase-multilingual-MiniLM-L12-v2` | HF | 384 | Мультиязычная (быстрая) |

## Бенчмарк

Для каждой модели:
1. Эмбеддим один и тот же набор чанков (фикс. версия из `data/experiments/<chosen>/`)
2. Заливаем во временную коллекцию Qdrant
3. Прогоняем `golden_qa.jsonl` → считаем:
   - `recall@1`, `recall@5`, `recall@10`
   - `MRR` (Mean Reciprocal Rank)
   - `nDCG@10`
4. Логируем в MLflow

## Метрики помимо качества
- Время эмбеддинга (sec/chunk)
- Размер вектора (память Qdrant)
- VRAM при инференсе

## TODO

- [ ] `benchmark.py` — один скрипт, прогоняющий все модели по golden_qa
- [ ] `results/` — md-таблица + plot
