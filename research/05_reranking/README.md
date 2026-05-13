# 05_reranking — подбор реранкера

## Цель
Подобрать cross-encoder, который лучше всего сортирует кандидатов после первичного ретрива.

## Кандидаты

| Модель | Размер | Особенности |
|---|---|---|
| `BAAI/bge-reranker-v2-m3` | base | Текущий прод, мультиязычный |
| `BAAI/bge-reranker-v2-gemma` | 2B | Сильнее, медленнее |
| `BAAI/bge-reranker-v2-minicpm-layerwise` | 2.5B | Адаптивный |
| `mixedbread-ai/mxbai-rerank-large-v1` | large | EN |
| `jinaai/jina-reranker-v2-base-multilingual` | base | Мультиязычный, быстрый |

## Бенчмарк

Зафиксировать ретривер (top-20) → прогнать каждый реранкер → сравнить:
- recall@5 после реранка (растёт ли по сравнению с dense top-5)
- среднее изменение позиции "идеальных" чанков
- скорость (ms/query × 20 docs)

## TODO

- [ ] `benchmark.py`
- [ ] Эксперимент с `RERANKER_TOP_K` (5 vs 8 vs 10)
