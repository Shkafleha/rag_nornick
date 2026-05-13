"""Работа с PostgreSQL БД для RAG системы."""
import os
import json
import logging
from typing import Any, Dict, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/llm_platform"
)

# Пул соединений для производительности
_db_pool: Optional[SimpleConnectionPool] = None


def init_db_pool(minconn: int = 2, maxconn: int = 10) -> SimpleConnectionPool:
    """Инициализировать пул соединений с БД."""
    global _db_pool
    try:
        _db_pool = SimpleConnectionPool(minconn, maxconn, DATABASE_URL)
        logger.info("Database pool initialized successfully")
        _create_tables()
        return _db_pool
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise


def get_db_connection() -> psycopg2.extensions.connection:
    """Получить соединение из пула."""
    if _db_pool is None:
        init_db_pool()
    return _db_pool.getconn()


def return_db_connection(conn: psycopg2.extensions.connection) -> None:
    """Вернуть соединение в пул."""
    if _db_pool is not None:
        _db_pool.putconn(conn)


def _create_tables() -> None:
    """Создать таблицы если их нет."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # Таблица для логирования запросов RAG
        cursor.execute("""
            CREATE SCHEMA IF NOT EXISTS rag_data;

            CREATE TABLE IF NOT EXISTS rag_data.queries (
                id SERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                model VARCHAR(255),
                created_at TIMESTAMP DEFAULT NOW(),
                langsmith_trace_id VARCHAR(255)
            );

            CREATE INDEX IF NOT EXISTS idx_queries_created ON rag_data.queries(created_at);
            CREATE INDEX IF NOT EXISTS idx_queries_trace ON rag_data.queries(langsmith_trace_id);

            CREATE TABLE IF NOT EXISTS rag_data.responses (
                id SERIAL PRIMARY KEY,
                query_id INTEGER REFERENCES rag_data.queries(id) ON DELETE CASCADE,
                answer TEXT,
                citations JSONB,
                response_time_ms FLOAT,
                model VARCHAR(255),
                created_at TIMESTAMP DEFAULT NOW(),
                langsmith_trace_id VARCHAR(255)
            );

            CREATE INDEX IF NOT EXISTS idx_responses_query ON rag_data.responses(query_id);
            CREATE INDEX IF NOT EXISTS idx_responses_created ON rag_data.responses(created_at);

            CREATE TABLE IF NOT EXISTS rag_data.extracted_data (
                id SERIAL PRIMARY KEY,
                query_id INTEGER REFERENCES rag_data.queries(id) ON DELETE CASCADE,
                extracted_content JSONB,
                source VARCHAR(255),
                confidence FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_extracted_query ON rag_data.extracted_data(query_id);
        """)

        conn.commit()
        logger.info("Database tables created/verified")
    except psycopg2.Error as e:
        conn.rollback()
        logger.debug(f"Table creation: {e}")
    finally:
        cursor.close()
        return_db_connection(conn)


def save_query(
    query: str,
    model: str,
    langsmith_trace_id: Optional[str] = None,
) -> int:
    """Сохранить запрос в БД. Вернуть ID запроса."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO rag_data.queries (query, model, langsmith_trace_id)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (query, model, langsmith_trace_id),
        )
        query_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"Query saved: id={query_id}, trace_id={langsmith_trace_id}")
        return query_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving query: {e}")
        raise
    finally:
        cursor.close()
        return_db_connection(conn)


def save_response(
    query_id: int,
    answer: str,
    citations: list,
    response_time_ms: float,
    model: str,
    langsmith_trace_id: Optional[str] = None,
) -> int:
    """Сохранить ответ в БД. Вернуть ID ответа."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO rag_data.responses
            (query_id, answer, citations, response_time_ms, model, langsmith_trace_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                query_id,
                answer,
                json.dumps(citations),
                response_time_ms,
                model,
                langsmith_trace_id,
            ),
        )
        response_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"Response saved: id={response_id}, query_id={query_id}")
        return response_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving response: {e}")
        raise
    finally:
        cursor.close()
        return_db_connection(conn)


def get_query_history(limit: int = 100) -> list:
    """Получить историю запросов."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT id, query, model, created_at, langsmith_trace_id
            FROM rag_data.queries
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error fetching query history: {e}")
        return []
    finally:
        cursor.close()
        return_db_connection(conn)


def close_db_pool() -> None:
    """Закрыть пул соединений."""
    global _db_pool
    if _db_pool is not None:
        _db_pool.closeall()
        logger.info("Database pool closed")
