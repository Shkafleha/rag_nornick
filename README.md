# OLLAMA Container Setup

Конфигурация для запуска OLLAMA контейнера с сервисом embeddings для работы с моделями sentence-transformers.

## Структура проекта (Cookiecutter Data Science)

```
├── data/
│   ├── raw/           # исходные данные (PDF, JSONL)
│   └── processed/     # обработанные данные (chunks, embeddings)
├── notebooks/         # Jupyter ноутбуки
├── src/
│   ├── data/          # скрипты обработки данных (qdrant_dual_index.py)
│   ├── features/
│   ├── models/
│   └── visualization/
├── models/            # сохранённые модели (LoRA, etc)
├── reports/
├── tests/
├── services/          # сервисы (см. ниже)
└── docker-compose.yml
```

## Docker-сервисы

```
services/
├── notebook/          # CUDA Jupyter + requirements
├── embeddings/        # embeddings API (в includes src/embeddings_api.py)
└── rag_api/           # FastAPI + LangGraph
```

## Модели для загрузки

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `intfloat/multilingual-e5-base`
- `intfloat/multilingual-e5-large`
- `BAAI/bge-m3`

## Запуск

```bash
docker-compose up -d
```

## Сервисы

### Ollama
- **Порт**: `11434`
- **URL**: `http://localhost:11434`
- Работает с моделями в формате GGUF

### Embeddings API
- **Порт**: `8080`
- **URL**: `http://localhost:8080`
- API сервис для работы с моделями embeddings из HuggingFace

### Jupyter Notebook
- **Порт**: `8888`
- **URL**: `http://localhost:8888`
- Ноутбуки запускаются в Docker; проект монтируется в `/workspace`, API embeddings доступен по `http://embeddings_service:8080`

### Jupyter Notebook с CUDA (GPU)
- **Порт**: `8889` (внутри контейнера 8888)
- **URL**: `http://localhost:8889`
- Образ собран из `services/notebook/Dockerfile` (nvidia/cuda + Jupyter + datascience-стек). На хосте нужен **nvidia-container-toolkit**. Запуск только этого сервиса: `docker-compose up -d notebook-gpu`

## Использование Embeddings API

### Проверка здоровья сервиса
```bash
curl http://localhost:8080/health
```

### Список загруженных моделей
```bash
curl http://localhost:8080/models
```

### Генерация embeddings для одного текста
```bash
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Привет, мир!"}'
```

### Генерация embeddings с указанием модели
```bash
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Привет, мир!",
    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  }'
```

### Пакетная обработка
```bash
curl -X POST http://localhost:8080/embed/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Текст 1", "Текст 2", "Текст 3"],
    "model": "intfloat/multilingual-e5-base"
  }'
```

### Пример на Python
```python
import requests

# Генерация embeddings
response = requests.post(
    'http://localhost:8080/embed',
    json={
        'text': 'Пример текста для обработки',
        'model': 'BAAI/bge-m3'  # опционально
    }
)

data = response.json()
embedding = data['embedding']
print(f"Размерность: {data['dimension']}")
```

## Запуск ноутбука через Docker

1. Запустите все сервисы:
   ```bash
   docker-compose up -d
   ```

2. Узнайте токен Jupyter (в логах контейнера):
   ```bash
   docker logs llm_notebook
   ```
   В выводе будет строка вида: `http://127.0.0.1:8888/?token=...` — скопируйте полный URL.

3. Откройте в браузере: **http://localhost:8888** и вставьте токен из логов (или перейдите по скопированному URL, заменив 127.0.0.1 на localhost).

4. В проекте откройте ноутбук `notebooks/v2_cobalt_local.ipynb`. PDF должен лежать в `data/raw/instr.pdf` (структура Cookiecutter Data Science).

## Примечания

- Модели загружаются автоматически при старте контейнера
- Первая загрузка моделей может занять время (зависит от размера модели)
- Кэш моделей сохраняется в volume `embeddings_cache`
- Если модель не указана, используется первая доступная из списка
