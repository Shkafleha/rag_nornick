# 07_e2e_eval — end-to-end оценка пайплайна

## Цель
Единая площадка, где меняешь одну переменную (embedder, реранкер, LLM, чанкинг) и видишь, **улучшился ли финальный ответ**. Без этого все остальные эвалы условны.

## Golden dataset

`datasets/golden_qa.jsonl` — формат:
```jsonl
{"id": "q001", "question": "Какие параметры контролируются на пачуке №504?", "expected_answer": "Уровень, рН, температура, содержание соды", "expected_chunk_ids": [357], "tags": ["контроль", "карбонатный_передел"]}
```

Минимум 30 вопросов, оптимально 100+. Источники:
- Сам пишешь по инструкции
- Генерируешь через большую LLM ("прочитай раздел и сгенерируй 5 вопросов с ответами и номерами разделов")
- Логи Langfuse: `user_feedback = 0` → берёшь как тест-кейсы регрессии

## Метрики (RAGAS)

| Метрика | Что меряет |
|---|---|
| `faithfulness` | Не галлюцинирует ли ответ за пределами контекста |
| `answer_relevancy` | Релевантен ли ответ вопросу |
| `context_precision` | Доля релевантных чанков в контексте |
| `context_recall` | Найдены ли все нужные чанки |

Плюс свои:
- `chunk_recall@5` = `|expected_chunk_ids ∩ retrieved_top_5| / |expected_chunk_ids|`
- `exact_match_keywords` = простой матч ключевых слов из expected_answer

## Запуск

```bash
python research/07_e2e_eval/run.py \
    --api http://localhost:8000 \
    --dataset research/07_e2e_eval/datasets/golden_qa.jsonl \
    --report research/07_e2e_eval/reports/$(date +%Y-%m-%d).md
```

Скрипт также залогирует прогон в Langfuse как **dataset run** → можно сравнивать прогоны в UI Langfuse.

## TODO

- [ ] Заполнить `golden_qa.jsonl` (минимум 20 вопросов)
- [ ] `run.py` (см. ниже — каркас уже есть)
- [ ] `ragas_runner.py` — обёртка над RAGAS
