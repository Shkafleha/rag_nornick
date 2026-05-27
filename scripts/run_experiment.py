#!/usr/bin/env python3
"""Прогоняет все вопросы из dataset через /ask и привязывает traces к items.

Запуск:
  # Полный прогон 85 вопросов (~40 мин при ~30с/вопрос)
  docker cp scripts/run_experiment.py rag_api:/tmp/r.py
  docker exec -e RUN_NAME=baseline-qwen3-8b rag_api python /tmp/r.py

  # Тестовый прогон 5 вопросов
  docker exec -e RUN_NAME=test -e LIMIT=5 rag_api python /tmp/r.py

Параметры через env:
  LANGFUSE_HOST       (из env контейнера)
  LANGFUSE_PUBLIC_KEY
  LANGFUSE_SECRET_KEY
  RAG_API_URL         (default: http://rag_api:8000)
  DATASET_NAME        (default: golden_questions)
  RUN_NAME            (обязательный — имя experiment run, по нему сравнивают версии)
  RUN_DESCRIPTION     (опц., короткое описание прогона)
  LIMIT               (опц., число вопросов для тестового прогона)
"""
import os
import sys
import time
import requests
from datetime import datetime

try:
    from langfuse import Langfuse
except ImportError:
    sys.exit("langfuse не установлен")

HOST = os.getenv("LANGFUSE_HOST")
PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
RAG_API_URL = os.getenv("RAG_API_URL", "http://rag_api:8000")
DATASET_NAME = os.getenv("DATASET_NAME", "golden_questions")
RUN_NAME = os.getenv("RUN_NAME") or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
RUN_DESCRIPTION = os.getenv("RUN_DESCRIPTION", "")
LIMIT = int(os.getenv("LIMIT", "0")) or None

if not (HOST and PUBLIC_KEY and SECRET_KEY):
    sys.exit("LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY должны быть заданы")

lf = Langfuse(host=HOST, public_key=PUBLIC_KEY, secret_key=SECRET_KEY)
ds = lf.get_dataset(DATASET_NAME)
items = ds.items[:LIMIT] if LIMIT else ds.items
print(f"→ Запуск '{RUN_NAME}' на {len(items)} вопросах из '{DATASET_NAME}'")
print(f"  API: {RAG_API_URL}/ask | Langfuse: {HOST}")
print()

ok = 0
errors = 0
total_t = time.perf_counter()

for i, item in enumerate(items, 1):
    q = (item.input or {}).get("question", "")
    expected = (item.expected_output or {}).get("answer", "")
    if not q:
        print(f"  [{i:>3}/{len(items)}] SKIP (пустой вопрос)")
        continue

    # Создаём собственный trace для эксперимента
    trace = lf.trace(
        name="experiment_ask",
        input={"question": q},
        metadata={
            "dataset_item_id": item.id,
            "run_name": RUN_NAME,
            "expected_answer": expected[:300],
        },
    )

    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{RAG_API_URL}/ask",
            json={"q": q, "history": []},
            timeout=600,
        )
        r.raise_for_status()
        data = r.json()
        answer = data.get("answer", "")
        timings = data.get("timings", {})
        api_trace_id = data.get("trace_id")
        elapsed = time.perf_counter() - t0

        trace.update(
            output={
                "answer": answer,
                "n_citations": len(data.get("citations", [])),
                "api_trace_id": api_trace_id,
            },
            metadata={"timings": timings, "api_trace_id": api_trace_id},
        )
        ok += 1
        print(f"  [{i:>3}/{len(items)}] ✓ {elapsed:>5.1f}s | {len(answer):>4}ch | {q[:60]}")
    except Exception as e:
        trace.update(output={"error": str(e)[:300]}, level="ERROR")
        errors += 1
        print(f"  [{i:>3}/{len(items)}] ✗ {type(e).__name__}: {e}")

    # Привязываем этот trace к dataset item под именем run'а
    try:
        item.link(trace, run_name=RUN_NAME, run_description=RUN_DESCRIPTION or None)
    except Exception as e:
        print(f"        ! link failed: {e}")

lf.flush()
total = time.perf_counter() - total_t
print()
print(f"✓ Готово: {ok} ok, {errors} errors, {total/60:.1f} мин")
print(f"  Откройте: {HOST}/project → Datasets → {DATASET_NAME} → Runs → {RUN_NAME}")
