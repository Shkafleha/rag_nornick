# Notebooks — Этапы разработки RAG системы

## 📁 Структура папок

### 📸 `ocr/` — OCR и парсинг документов
Ноутбуки для работы с извлечением и парсингом текста из PDF:
- **01_v2_cobalt_local.ipynb** — Парсинг PDF (Cobalt, локально через Docker)
- **03_inspect_chunks.ipynb** — Визуализация разбивки документа на chunks с breadcrumbs

### 🔬 `eval/` — Оценка метрик и качества
- **02_eval_ocr_pre_llm.ipynb** — Оценка сырых результатов OCR (layout-боксы, разметка)
- **04_eval_ocr_post_llm.ipynb** — Анализ нормализованного текста (структура, breadcrumbs)
- **05_eval_search.ipynb** — Бенчмарк embedding-моделей (BGE, E5 и т.д.)

### 🧪 `experiments/` — Разработка и экспименты
- **06_v3_rag.ipynb** — RAG система с тестированием эмбеддингов
- **07_v4_langgraph.ipynb** — Финальный граф LangGraph
- **08_qlora_min.ipynb** — Эксперимент с fine-tuning (LoRA)

## 🚀 Как использовать

1. **OCR парсинг**: `01_v2_cobalt_local.ipynb` — извлечение текста из PDF
2. **Оценка ДО обработки**: `02_eval_ocr_pre_llm.ipynb` — сырые результаты OCR
3. **Анализ chunks**: `03_inspect_chunks.ipynb` — разбивка на chunks с breadcrumbs
4. **Оценка ПОСЛЕ обработки**: `04_eval_ocr_post_llm.ipynb` — нормализованный текст
5. **Поиск и embeddings**: `05_eval_search.ipynb` — бенчмарк моделей
6. **RAG разработка**: `06_v3_rag.ipynb` → `07_v4_langgraph.ipynb` → `08_qlora_min.ipynb`
