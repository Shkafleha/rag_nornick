#!/usr/bin/env python3
"""Тест реранкера"""
import requests
import json

RERANKER_URL = "http://localhost:8082"

def test_reranker():
    print(f"Testing reranker at {RERANKER_URL}...")

    # Проверка здоровья
    try:
        r = requests.get(f"{RERANKER_URL}/health", timeout=5)
        print(f"✓ Health: {r.status_code}")
    except Exception as e:
        print(f"✗ Health failed: {e}")
        return

    # Тестовый реранкинг
    query = "очистка электролита"
    docs = [
        "Электролит содержит примеси железа и цинка",
        "Процесс очистки включает несколько этапов",
        "Температура раствора 60 градусов Цельсия"
    ]

    payload = {
        "query": query,
        "docs": docs,
        "top_k": 2
    }

    print(f"\nSending request:")
    print(f"  Query: {query}")
    print(f"  Docs count: {len(docs)}")

    try:
        r = requests.post(
            f"{RERANKER_URL}/rerank",
            json=payload,
            timeout=30
        )
        print(f"\n✓ Response: {r.status_code}")
        print(f"  Response size: {len(r.text)} bytes")
        print(f"  Response body: {r.text[:200]}")

        if r.status_code == 200:
            data = r.json()
            print(f"\n✓ Parsed JSON:")
            print(json.dumps(data, indent=2))
    except requests.Timeout:
        print(f"✗ Timeout after 30 seconds")
    except Exception as e:
        print(f"✗ Request failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_reranker()
