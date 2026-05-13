# 04_retrieval — стратегии поиска

## Цель
Найти оптимальную комбинацию ретриверов поверх лучшего embedder'а.

## Стратегии

| Стратегия | Что даёт |
|---|---|
| `dense` | Текущий baseline (Qdrant cosine) |
| `bm25` | Точные совпадения, аббревиатуры, цифры, термины |
| `hybrid_rrf` | Reciprocal Rank Fusion (dense + BM25) |
| `hyde` | LLM генерирует "гипотетический ответ" → его эмбеддят |
| `multi_query` | LLM делает 3-5 переформулировок запроса, объединение |
| `query_rewrite_history` | Переписывание с учётом истории диалога — фиксит "расскажи подробнее" |

## Инструменты
- `llama-index-retrievers-bm25` — уже в requirements
- `llama_index.core.retrievers.QueryFusionRetriever` — RRF из коробки
- Qdrant 1.10+ — нативный sparse-vector indexing

## Метрики
Те же что в `03_embeddings`: recall@k, MRR, nDCG.

## TODO

- [ ] `strategies/` — по файлу на ретривер, единый интерфейс `retrieve(query, k) -> List[Hit]`
- [ ] `eval/run.py` — прогон всех стратегий по golden_qa
