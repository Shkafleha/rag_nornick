import os
import logging
import requests
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

RAG_API_URL = os.getenv("RAG_API_URL", "http://rag_api:8000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

st.set_page_config(page_title="RAG Docs", page_icon="📄", layout="wide")
st.title("📄 RAG по документам")

with st.sidebar:
    st.header("Коллекции")
    try:
        r = requests.get(f"{QDRANT_URL}/collections", timeout=5)
        all_collections = [c["name"] for c in r.json()["result"]["collections"]]
    except:
        all_collections = []

    selected = [c for c in all_collections if st.checkbox(c, key=f"col_{c}")] if all_collections else all_collections[0]

if "messages" not in st.session_state:
    st.session_state.messages = []

def _render_chunk(c):
    header = c.get("header_breadcrumb") or c.get("header") or (c["text"].split("\n", 1)[0] if c.get("text") else "")
    st.markdown(f"**{header}**")
    meta = [f"chunk {c['chunk_id']}", f"score {c['score']}"]
    if c.get("rerank_score"):
        meta.append(f"rerank {c['rerank_score']}")
    if c.get("page"):
        meta.append(f"стр. {c['page']}")
    st.markdown(" · ".join(meta))
    st.caption(c["text"])
    st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

EXAMPLES = ["Расскажи как работает карбонатный передел"]
cols = st.columns(len(EXAMPLES))
preset = None
for i, q in enumerate(EXAMPLES):
    if cols[i].button(q, key=f"ex_{i}"):
        preset = q

prompt = preset or st.chat_input("Вопрос…")
if prompt:
    logger.info(f"🔵 Новый вопрос: {prompt[:50]}...")

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages[:-1] if m["role"] in ("user", "assistant")]

    with st.chat_message("assistant"):
        # Этап 1: поиск чанков
        citations = []
        citations_pre_rerank = []
        search_timings = {}
        with st.spinner("🔍 Ищу релевантные фрагменты…"):
            try:
                r = requests.post(
                    f"{RAG_API_URL}/search",
                    json={"q": prompt, "collections": selected},
                    timeout=120,
                )
                r.raise_for_status()
                sd = r.json()
                citations = sd.get("citations", [])
                citations_pre_rerank = sd.get("citations_pre_rerank", [])
                search_timings = sd.get("timings", {})
                logger.info(f"✓ /search: pre={len(citations_pre_rerank)} → post={len(citations)}")
            except Exception as e:
                logger.error(f"❌ /search: {type(e).__name__}: {e}")
                st.error(f"Ошибка поиска: {str(e)[:200]}")

        if citations_pre_rerank:
            with st.expander(f"🔎 До реранка ({len(citations_pre_rerank)})", expanded=False):
                for c in citations_pre_rerank:
                    _render_chunk(c)
        if citations:
            with st.expander(f"📄 После реранка ({len(citations)})", expanded=False):
                for c in citations:
                    _render_chunk(c)

        # Этап 2: генерация ответа (передаём готовые citations — без повторного поиска)
        answer = ""
        timings = dict(search_timings)
        trace_id = None
        with st.spinner("🤖 Генерирую ответ…"):
            try:
                r = requests.post(
                    f"{RAG_API_URL}/ask",
                    json={
                        "q": prompt,
                        "collections": selected,
                        "history": history,
                        "citations": citations,
                    },
                    timeout=600,
                )
                r.raise_for_status()
                data = r.json()
                answer = data.get("answer", "")
                timings.update(data.get("timings", {}))
                trace_id = data.get("trace_id")
                logger.info(f"✓ /ask: {len(answer)} символов")
            except Exception as e:
                logger.error(f"❌ /ask: {type(e).__name__}: {e}")
                answer = f"❌ Ошибка: {str(e)[:200]}"

        st.markdown(answer)
        if timings:
            parts = [f"**{k}**: {timings[k]/1000:.1f}с"
                     for k in ["embed_query", "retrieve", "rerank", "build_context", "generate", "total"]
                     if k in timings]
            st.caption(" · ".join(parts))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "trace_id": trace_id,
    })
