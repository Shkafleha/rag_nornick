#!/usr/bin/env python3
"""Диагностика RAG-системы: проверка всех сервисов и подключений"""

import sys
import io

# Исправить кодировку для Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
from typing import Dict, Tuple

QDRANT_URL = "http://localhost:6333"
RAG_API_URL = "http://localhost:8000"
OLLAMA_URL = "http://localhost:11434"  # Локальная Ollama
RERANKER_URL = "http://localhost:8082"

def check_service(name: str, url: str, endpoint: str = "", expected_status=200) -> Tuple[bool, str]:
    """Проверяет доступность сервиса"""
    full_url = f"{url}{endpoint}"
    try:
        r = requests.get(full_url, timeout=5)
        if r.status_code == expected_status:
            return True, f"[OK] {name} доступен"
        else:
            return False, f"[FAIL] {name} вернул статус {r.status_code}"
    except requests.exceptions.Timeout:
        return False, f"[FAIL] {name} - timeout (не доступен)"
    except requests.exceptions.ConnectionError:
        return False, f"[FAIL] {name} - connection error (сервис не запущен)"
    except Exception as e:
        return False, f"[FAIL] {name} - {type(e).__name__}: {str(e)}"

def check_qdrant() -> Dict:
    """Проверяет Qdrant и коллекции"""
    status, msg = check_service("Qdrant", QDRANT_URL, "/collections")
    result = {"status": msg}

    if status:
        try:
            r = requests.get(f"{QDRANT_URL}/collections", timeout=5)
            collections = r.json().get("result", {}).get("collections", [])
            result["collections"] = [c.get("name", "?") for c in collections]
        except Exception as e:
            result["collections_error"] = str(e)

    return result

def check_rag_api() -> Dict:
    """Проверяет RAG API"""
    status, msg = check_service("RAG API", RAG_API_URL, "/health")
    result = {"status": msg}

    if not status:
        return result

    # Попробуй сделать тестовый запрос
    try:
        r = requests.post(
            f"{RAG_API_URL}/ask",
            json={"q": "test"},
            timeout=10
        )
        if r.status_code == 502:
            try:
                detail = r.json().get("detail", "unknown")
                result["error"] = f"502 Bad Gateway: {detail}"
            except:
                result["error"] = "502 Bad Gateway (нет деталей в ответе)"
        elif r.status_code == 200:
            result["test_request"] = "[OK] Тестовый запрос успешен"
        else:
            result["error"] = f"Статус {r.status_code}"
    except Exception as e:
        result["test_request_error"] = str(e)

    return result

def check_ollama() -> Dict:
    """Проверяет Ollama"""
    status, msg = check_service("Ollama", OLLAMA_URL, "/api/tags")
    result = {"status": msg}

    if status:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            models = r.json().get("models", [])
            result["models"] = [m.get("name", "?") for m in models]
        except Exception as e:
            result["models_error"] = str(e)

    return result

def check_reranker() -> Dict:
    """Проверяет Reranker"""
    status, msg = check_service("Reranker", RERANKER_URL, "/health")
    return {"status": msg}

if __name__ == "__main__":
    print("=" * 70)
    print("RAG-СИСТЕМА: ДИАГНОСТИКА")
    print("=" * 70)
    print()

    services = {
        "Qdrant (Vector DB)": check_qdrant,
        "RAG API (FastAPI)": check_rag_api,
        "Ollama (LLM)": check_ollama,
        "Reranker (Cross-encoder)": check_reranker,
    }

    for service_name, check_func in services.items():
        print(f"\n[{service_name}]")
        result = check_func()
        for key, value in result.items():
            if key == "status":
                print(f"  {value}")
            elif isinstance(value, list):
                print(f"  {key}: {', '.join(value)}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИИ:")
    print("=" * 70)
    print("""
1. Если Ollama показывает ❌:
   - Проверьте, запущена ли Ollama на 192.168.31.241:11434
   - Обновите OLLAMA_URL в этом скрипте и docker-compose.yml

2. Если RAG API показывает 502:
   - Это обычно означает, что одна из зависимостей недоступна (Ollama, Qdrant, Reranker)
   - Смотрите детали в разделе "RAG API (FastAPI)"

3. Если все зелёные ✅:
   - Проверьте логи: docker logs rag_api
   - Попробуйте отправить запрос через curl:
     curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"q":"test"}'
    """)
