"""
Простой UI для RAG-системы на Streamlit.

Что делает:
  1. Берёт список коллекций из Qdrant (сайдбар с чекбоксами)
  2. Принимает вопрос от пользователя (чат-инпут)
  3. Отправляет вопрос в RAG API (POST /ask)
  4. Показывает ответ + найденные чанки
"""

import os

import requests
import streamlit as st

# ── Настройки ───────────────────────────────────────────────────────────────
RAG_API_URL = os.getenv("RAG_API_URL", "http://rag_api:8000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

# ── Страница ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Docs", page_icon="📄", layout="wide")
st.title("📄 RAG по документам")

# ── Сайдбар: выбор коллекций из Qdrant ──────────────────────────────────────
with st.sidebar:
    st.header("Коллекции")
    try:
        r = requests.get(f"{QDRANT_URL}/collections", timeout=5)
        all_collections = [c["name"] for c in r.json()["result"]["collections"]]
    except Exception:
        all_collections = []

    if all_collections:
        selected = [c for c in all_collections if st.checkbox(c, value=(c == "CEN2_all_pages"), key=f"col_{c}")]
    else:
        st.warning("Нет доступных коллекций")
        selected = []

# ── История чата (хранится в session_state между перерисовками) ──────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_chunk(c):
    header = c.get("header_breadcrumb") or c.get("header") or (c["text"].split("\n", 1)[0] if c.get("text") else "")
    st.markdown(f"**{header}**")
    meta = [f"chunk {c['chunk_id']}", f"score {c['score']}"]
    if c.get("rerank_score") is not None:
        meta.append(f"rerank {c['rerank_score']}")
    if c.get("type"):
        meta.append(f"тип: {c['type']}")
    if c.get("page") is not None:
        meta.append(f"стр. {c['page']}")
    if c.get("source"):
        meta.append(c["source"])
    st.markdown(" · ".join(meta))
    st.caption(c["text"])
    st.divider()


def render_citations(items, title="Найденные чанки", expanded=True):
    if not items:
        return
    with st.expander(f"{title} ({len(items)})", expanded=expanded):
        for c in items:
            _render_chunk(c)

# Показываем предыдущие сообщения
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("timings"):
            order = ["embed_query", "retrieve", "rerank", "build_context", "generate", "total"]
            parts = [f"**{k}**: {msg['timings'][k]/1000:.1f} с" for k in order if k in msg["timings"]]
            st.caption(" · ".join(parts))
        render_citations(msg.get("citations_pre_rerank") or [], title="До реранкера", expanded=False)
        render_citations(msg.get("citations") or [], title="После реранкера", expanded=True)
        tid = msg.get("trace_id")
        if tid and msg["role"] == "assistant":
            scored_key = f"scored_{tid}"
            if st.session_state.get(scored_key):
                st.caption(f"✓ Оценено: {st.session_state[scored_key]}")
            else:
                c1, c2, _ = st.columns([1, 1, 8])
                if c1.button("👍", key=f"up_{tid}"):
                    try:
                        requests.post(f"{RAG_API_URL}/score",
                                      json={"trace_id": tid, "value": 1.0}, timeout=10)
                        st.session_state[scored_key] = "👍"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Score failed: {e}")
                if c2.button("👎", key=f"down_{tid}"):
                    try:
                        requests.post(f"{RAG_API_URL}/score",
                                      json={"trace_id": tid, "value": 0.0}, timeout=10)
                        st.session_state[scored_key] = "👎"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Score failed: {e}")

# ── Примеры вопросов ────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "Расскажи как работает карбонатный передел",
]
cols = st.columns(len(EXAMPLE_QUESTIONS))
preset_prompt = None
for i, q in enumerate(EXAMPLE_QUESTIONS):
    if cols[i].button(q, key=f"example_{i}"):
        preset_prompt = q

# ── Ввод вопроса ────────────────────────────────────────────────────────────
chat_prompt = st.chat_input("Задай вопрос по документу…")
prompt = preset_prompt or chat_prompt
if prompt:
    # Показываем вопрос пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Отправляем в RAG API и показываем ответ
    with st.chat_message("assistant"):
        with st.spinner("Думаю…"):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]  # без только что добавленного user-сообщения
                    if m["role"] in ("user", "assistant")
                ]
                r = requests.post(
                    f"{RAG_API_URL}/ask",
                    json={"q": prompt, "collections": selected, "history": history},
                    timeout=600,
                )
                
                r.raise_for_status()
                result = r.json()
                answer = result["answer"]
                citations = result.get("citations", [])
                citations_pre = result.get("citations_pre_rerank", [])
                timings = result.get("timings", {})
                trace_id = result.get("trace_id")
            except Exception as e:
                answer = f"Ошибка: {e}"
                citations = []
                citations_pre = []
                timings = {}
                trace_id = None

        st.markdown(answer)
        if timings:
            order = ["embed_query", "retrieve", "rerank", "build_context", "generate", "total"]
            parts = [f"**{k}**: {timings[k]/1000:.1f} с" for k in order if k in timings]
            st.caption(" · ".join(parts))
        render_citations(citations_pre, title="До реранкера", expanded=False)
        render_citations(citations, title="После реранкера", expanded=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "citations_pre_rerank": citations_pre,
        "timings": timings,
        "trace_id": trace_id,
    })
    st.rerun()
