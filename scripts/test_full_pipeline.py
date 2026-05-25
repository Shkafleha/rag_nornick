#!/usr/bin/env python3
"""Полный тест RAG-пайплайна: поиск и генерация"""
import requests
import json
import time

RAG_API_URL = "http://localhost:8000"

def test_search():
    """Тест только поиска (без генерации)"""
    print("=" * 70)
    print("TEST 1: /search (embed + retrieve + rerank, NO generation)")
    print("=" * 70)

    query = "очистка электролита"
    payload = {
        "q": query,
        "collections": []
    }

    print(f"Query: {query}\n")

    t0 = time.time()
    try:
        r = requests.post(
            f"{RAG_API_URL}/search",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - t0

        print(f"✓ Response: {r.status_code} ({elapsed:.1f}s)")
        if r.status_code == 200:
            data = r.json()
            print(f"  Citations: {len(data.get('citations', []))}")
            print(f"  Timings: {data.get('timings', {})}")
        else:
            print(f"  Error: {r.text[:200]}")
    except requests.Timeout:
        print(f"✗ Timeout after 60s")
    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {e}")

    print()


def test_ask():
    """Тест полного RAG (с генерацией)"""
    print("=" * 70)
    print("TEST 2: /ask (embed + retrieve + rerank + generate)")
    print("=" * 70)

    query = "очистка электролита"
    payload = {
        "q": query,
        "collections": [],
        "history": []
    }

    print(f"Query: {query}\n")

    t0 = time.time()
    try:
        r = requests.post(
            f"{RAG_API_URL}/ask",
            json=payload,
            timeout=120  # 2 minutes для генерации
        )
        elapsed = time.time() - t0

        print(f"✓ Response: {r.status_code} ({elapsed:.1f}s)")
        if r.status_code == 200:
            data = r.json()
            print(f"  Answer length: {len(data.get('answer', ''))}")
            print(f"  Citations: {len(data.get('citations', []))}")
            print(f"  Timings: {data.get('timings', {})}")
            print(f"\n  Answer preview:")
            answer = data.get('answer', '')
            print(f"  {answer[:300]}..." if len(answer) > 300 else f"  {answer}")
        else:
            print(f"  Error: {r.text[:200]}")
    except requests.Timeout:
        print(f"✗ Timeout after 120s - likely stuck on LLM generation")
    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {e}")

    print()


if __name__ == "__main__":
    print("\nRAG Pipeline Test\n")

    # Проверка здоровья
    try:
        r = requests.get(f"{RAG_API_URL}/health", timeout=5)
        if r.status_code != 200:
            print(f"✗ API not healthy: {r.status_code}")
            exit(1)
        print(f"✓ API is healthy\n")
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        exit(1)

    test_search()
    test_ask()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
If /search works but /ask hangs:
  → Issue is with LLM generation (check if qwen3:8b is downloaded in Ollama)

If /search hangs on reranking:
  → Issue is with reranker (check docker logs reranker)

If both hang:
  → Issue is earlier in pipeline (embedding or retrieval)
    """)
