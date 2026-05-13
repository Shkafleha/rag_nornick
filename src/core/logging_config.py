"""Конфигурация логирования для RAG системы."""
import json
import logging
import logging.handlers
import os
import socket
import sys
from datetime import datetime
from typing import Any, Dict

try:
    import pythonjsonlogger.jsonlogger
    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False


class JSONFormatter(logging.Formatter):
    """JSON форматер для структурированных логов."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "hostname": socket.gethostname(),
            "service": os.getenv("SERVICE_NAME", "rag-system"),
        }

        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        if hasattr(record, "extra_data"):
            log_obj["extra"] = record.extra_data

        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(
    service_name: str = "rag-system",
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_logstash: bool = False,
    logstash_host: str = "localhost",
    logstash_port: int = 5000,
) -> logging.Logger:
    """
    Настроить логирование для приложения.

    Args:
        service_name: Имя сервиса
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        enable_console: Писать ли логи в консоль
        enable_file: Писать ли логи в файл
        enable_logstash: Отправлять ли логи в Logstash
        logstash_host: Хост Logstash
        logstash_port: Порт Logstash

    Returns:
        Настроенный logger
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Удаляем старые обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = JSONFormatter()

    # Логирование в консоль
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Логирование в файл с ротацией
    if enable_file:
        log_dir = os.getenv("LOG_DIR", "./logs")
        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, f"{service_name}.log"),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=10,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Отправка логов в Logstash (если включено)
    if enable_logstash and os.getenv("LOGSTASH_ENABLED", "false").lower() == "true":
        try:
            logstash_handler = logging.handlers.SocketHandler(
                logstash_host,
                logstash_port,
            )
            logstash_handler.setFormatter(formatter)
            root_logger.addHandler(logstash_handler)
        except Exception as e:
            root_logger.warning(f"Failed to connect to Logstash: {e}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Получить логгер для модуля."""
    return logging.getLogger(name)
