# 06_llm — бенчмарк LLM из Ollama

## Цель
Подобрать LLM для финальной генерации, балансируя качество, скорость и VRAM.

## Кандидаты

| Модель | Размер | Контекст | Заметки |
|---|---|---|---|
| `bambucha/saiga-llama3:8b-q4_K` | 8B | 8K | Текущий прод, RU |
| `qwen2.5:7b-instruct` | 7B | 32K | Сильный мультиязычный |
| `qwen2.5:14b-instruct` | 14B | 32K | Лучше, но 14ГБ VRAM |
| `llama3.1:8b` | 8B | 128K | Длинный контекст |
| `gemma2:9b` | 9B | 8K | |
| `phi-4` | 14B | 16K | Microsoft, сильна в STEM |
| `mistral-nemo` | 12B | 128K | |

## Метод

Для фикс. (вопрос + контекст) из `golden_qa`:

1. Прогнать каждую LLM
2. Оценить ответ:
   - **LLM-as-judge** — большая модель (gpt-4o / claude-opus локально нельзя, но можно qwen2.5:32b как judge)
   - **RAGAS metrics** — `faithfulness`, `answer_relevancy`
   - **Exact match** на фактах из эталона
3. Замерить tokens/sec, VRAM, latency

## Артефакт

```
results/<date>.md
| model              | faithfulness | relevance | tok/s | VRAM |
|--------------------|--------------|-----------|-------|------|
| qwen2.5:7b         | 0.85         | 0.91      | 45    | 6GB  |
| saiga-llama3:8b    | 0.78         | 0.87      | 38    | 7GB  |
```

## TODO

- [ ] `runner.py --models qwen2.5:7b,llama3.1:8b --questions golden_qa.jsonl`
- [ ] `judges/llm_judge.py`
- [ ] `judges/ragas_judge.py`
