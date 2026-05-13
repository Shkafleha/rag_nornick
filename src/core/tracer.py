"""Интеграция LangSmith для трассирования LLM цепочек."""
import os
from typing import Optional

from langsmith import Client

# Глобальный клиент LangSmith
_langsmith_client: Optional[Client] = None


def setup_tracer() -> Client:
    """
    Инициализировать LangSmith клиент.

    Использует переменные окружения:
    - LANGSMITH_ENDPOINT: LangSmith сервер (http://langsmith:8000 для Docker)
    - LANGSMITH_API_KEY: API ключ
    - LANGSMITH_PROJECT: Имя проекта
    """
    global _langsmith_client

    endpoint = os.getenv("LANGSMITH_ENDPOINT", "http://localhost:8000")
    api_key = os.getenv("LANGSMITH_API_KEY", "default-key")
    project = os.getenv("LANGSMITH_PROJECT", "rag-project")

    os.environ["LANGSMITH_ENDPOINT"] = endpoint
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGSMITH_TRACING_V2"] = "true"

    _langsmith_client = Client(api_url=endpoint, api_key=api_key)

    return _langsmith_client


def get_tracer() -> Optional[Client]:
    """Получить LangSmith клиент."""
    if _langsmith_client is None:
        setup_tracer()
    return _langsmith_client


def get_trace_id() -> Optional[str]:
    """Получить текущий ID trace из контекста."""
    try:
        from langsmith.run_trees import get_run_tree
        run_tree = get_run_tree()
        return run_tree.id if run_tree else None
    except Exception:
        return None
