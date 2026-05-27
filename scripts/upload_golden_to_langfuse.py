#!/usr/bin/env python3
"""Загружает golden dataset из data/04_golden_dataset/qa_pairs_v2_podpunkti_golden.jsonl
в Langfuse как dataset.

Запуск:
  - С хоста:        python scripts/upload_golden_to_langfuse.py
                    (нужен:  pip install langfuse python-dotenv)
  - Из контейнера:  docker cp scripts/upload_golden_to_langfuse.py rag_api:/tmp/u.py
                    docker cp data/04_golden_dataset rag_api:/tmp/golden
                    docker exec -e DATASET_PATH=/tmp/golden/qa_pairs_v2_podpunkti_golden.jsonl rag_api python /tmp/u.py

Параметры через env:
  LANGFUSE_HOST         (default: http://localhost:3000)
  LANGFUSE_PUBLIC_KEY   из .env
  LANGFUSE_SECRET_KEY   из .env
  DATASET_PATH          (default: data/04_golden_dataset/qa_pairs_v2_podpunkti_golden.jsonl)
  DATASET_NAME          (default: golden_questions)
"""
import json
import os
import sys
from pathlib import Path

# Подгружаем .env если есть
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from langfuse import Langfuse
except ImportError:
    sys.exit("langfuse не установлен. Сделайте: pip install langfuse python-dotenv")

HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
DATASET_PATH = Path(os.getenv("DATASET_PATH", "data/04_golden_dataset/qa_pairs_v2_podpunkti_golden.jsonl"))
DATASET_NAME = os.getenv("DATASET_NAME", "golden_questions")

if not PUBLIC_KEY or not SECRET_KEY:
    sys.exit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY не заданы (проверьте .env)")
if not DATASET_PATH.exists():
    sys.exit(f"Файл не найден: {DATASET_PATH}")

lf = Langfuse(host=HOST, public_key=PUBLIC_KEY, secret_key=SECRET_KEY)

# Создаём dataset (если уже есть — Langfuse вернёт существующий)
print(f"→ Создаю/обновляю dataset '{DATASET_NAME}' в {HOST}...")
lf.create_dataset(
    name=DATASET_NAME,
    description="Эталонные вопросы по ТИ 3-48200234 (очистка электролита)",
    metadata={"source": str(DATASET_PATH.name)},
)

# Загружаем элементы
created = 0
skipped = 0
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        qa = json.loads(line)
        q = (qa.get("question") or "").strip()
        a = (qa.get("answer") or "").strip()
        if not q or not a:
            skipped += 1
            continue
        lf.create_dataset_item(
            dataset_name=DATASET_NAME,
            input={"question": q},
            expected_output={"answer": a},
            metadata={
                "id": qa.get("id"),
                "type": qa.get("type"),
                "source_pages": qa.get("source_pages"),
                "source_chunks": qa.get("source_chunks"),
                "multi_hop_required": qa.get("multi_hop_required"),
            },
        )
        created += 1
        if created % 10 == 0:
            print(f"  загружено {created}…")

lf.flush()
print(f"✓ Готово: создано {created} элементов, пропущено {skipped}")
print(f"  Откройте: {HOST}/project → Datasets → {DATASET_NAME}")
