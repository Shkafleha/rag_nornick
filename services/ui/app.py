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

    selected = [c for c in all_collections if st.checkbox(c, key=f"col_{c}")] if all_collections else []

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
    logger.info(f"Selected collections: {selected}")

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages[:-1] if m["role"] in ("user", "assistant")]

    logger.info(f"History length: {len(history)}")
    logger.info(f"API URL: {RAG_API_URL}/ask")

    with st.spinner("🔍 Ищу и генерирую ответ…"):
        try:
            logger.info("📤 Отправляю запрос к /ask...")
            payload = {"q": prompt, "collections": selected, "history": history}
            logger.info(f"Payload: {payload}")

            r = requests.post(
                f"{RAG_API_URL}/ask",
                json=payload,
                timeout=600
            )

            logger.info(f"✅ Ответ получен: статус {r.status_code}")
            r.raise_for_status()
            data = r.json()
            logger.info(f"Data keys: {data.keys()}")

            answer = data.get("answer", "")
            citations = data.get("citations", [])
            timings = data.get("timings", {})
            trace_id = data.get("trace_id")

            logger.info(f"✓ Ответ: {len(answer)} символов, {len(citations)} чанков")
        except Exception as e:
            logger.error(f"❌ ОШИБКА: {type(e).__name__}: {str(e)}")
            answer = f"❌ Ошибка: {str(e)[:200]}"
            citations = []
            timings = {}
            trace_id = None

    with st.chat_message("assistant"):
        st.markdown(answer)
        if timings:
            parts = [f"**{k}**: {timings[k]/1000:.1f}с" for k in ["embed_query", "retrieve", "rerank", "build_context", "generate", "total"] if k in timings]
            st.caption(" · ".join(parts))
        if citations:
            with st.expander(f"📄 Чанки ({len(citations)})"):
                for c in citations:
                    _render_chunk(c)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "timings": timings,
        "trace_id": trace_id
    })
    st.rerun()
