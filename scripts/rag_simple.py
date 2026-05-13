"""
Минимальный RAG-пайплайн в одном файле.
Без фреймворков — только requests + print, чтобы видеть каждый шаг.

Запуск (из контейнера notebook-gpu или локально при запущенных Ollama + Qdrant):
  python scripts/rag_simple.py "Что такое ОРФ?"

Или с переменными окружения:
  OLLAMA_URL=http://localhost:11434 QDRANT_URL=http://localhost:6333 \
  QDRANT_COLLECTION=orf_chunks python scripts/rag_simple.py "Ваш вопрос"
"""

import os
import sys

import requests

# ── Настройки ───────────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "orf_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:latest")       # модель эмбеддингов
LLM_MODEL = os.getenv("LLM_MODEL", "bambucha/saiga-llama3:8b-q4_K")  # модель генерации
TOP_K = 5  # сколько релевантных чанков брать


# ── Шаг 1: Эмбеддинг вопроса ───────────────────────────────────────────────
# Превращаем текстовый вопрос в числовой вектор (список из 1024 чисел).
# Тот же самый bge-m3, которым индексировали документы.

def embed(text: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


# ── Шаг 2: Поиск похожих чанков в Qdrant ───────────────────────────────────
# Отправляем вектор вопроса в Qdrant, он находит TOP_K ближайших
# векторов (по косинусному расстоянию) и возвращает их вместе с payload.

def search(query_vector: list[float]) -> list[dict]:
    r = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        json={
            "vector": query_vector,
            "limit": TOP_K,
            "with_payload": True,  # вернуть текст и метаданные
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["result"]  # список {id, score, payload}


# ── Шаг 3: Сборка контекста ────────────────────────────────────────────────
# Берём тексты найденных чанков и склеиваем в один блок.
# Это и есть «контекст», который LLM получит вместе с вопросом.

def build_context(hits: list[dict]) -> str:
    texts = []
    for h in hits:
        text = h["payload"]["text"]
        score = h["score"]
        texts.append(f"[score={score:.4f}]\n{text}")
    return "\n\n---\n\n".join(texts)


# ── Шаг 4: Генерация ответа через LLM ──────────────────────────────────────
# Формируем промпт: инструкция + вопрос + контекст.
# LLM должна отвечать ТОЛЬКО на основе контекста (не выдумывать).

def generate(question: str, context: str) -> str:
    prompt = (
        "Отвечай на вопрос, используя только контекст ниже.\n"
        "Если в контексте нет ответа — скажи 'Не найдено в базе'.\n\n"
        f"Вопрос:\n{question}\n\n"
        f"Контекст:\n{context}\n"
    )

    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"]


# ── Запуск всего пайплайна ──────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Использование: python rag_simple.py \"Ваш вопрос\"")
        sys.exit(1)

    question = sys.argv[1]
    print(f"Вопрос: {question}\n")

    # Шаг 1
    print("1. Эмбеддинг вопроса...")
    q_vector = embed(question)
    print(f"   Размерность вектора: {len(q_vector)}")

    # Шаг 2
    print(f"\n2. Поиск в Qdrant (коллекция={COLLECTION}, top_k={TOP_K})...")
    hits = search(q_vector)
    print(f"   Найдено: {len(hits)} чанков")
    for i, h in enumerate(hits):
        score = h["score"]
        chunk_id = h["payload"].get("chunk_id", h["id"])
        text_preview = h["payload"]["text"][:80].replace("\n", " ")
        print(f"   [{i+1}] score={score:.4f}  chunk={chunk_id}")
        print(f"       {text_preview}...")

    # Шаг 3
    print("\n3. Сборка контекста...")
    context = build_context(hits)
    print(f"   Длина контекста: {len(context)} символов")

    # Шаг 4
    print(f"\n4. Генерация ответа (модель={LLM_MODEL})...")
    answer = generate(question, context)
    print(f"\n{'='*60}")
    print(f"ОТВЕТ:\n{answer}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
