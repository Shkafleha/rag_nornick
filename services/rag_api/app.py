import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, TypedDict

import requests
from fastapi import FastAPI, HTTPException, Query
from langgraph.graph import END, StateGraph

from pydantic import BaseModel
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse_client = Langfuse()

api = FastAPI()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", OLLAMA_URL)
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "CEN2_all_pages")

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "bambucha/saiga-llama3:8b-q4_K")

RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8080")
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "20"))



class State(TypedDict, total=False):
    query: str
    collections: List[str]
    history: List[Dict[str, str]]   # NEW: [{role, content}, ...]
    q_embedding: List[float]
    hits: List[Dict[str, Any]]
    context: str
    answer: str
    citations: List[Dict[str, Any]]
    citations_pre_rerank: List[Dict[str, Any]]
    timings: Dict[str, float]

class AskRequest(BaseModel):
    q: str
    collections: List[str] | None = None
    history: List[Dict[str, str]] | None = None  # [{"role":"user|assistant","content":"..."}]


class ScoreRequest(BaseModel):
    trace_id: str
    value: float            # 1.0 = 👍, 0.0 = 👎
    comment: str | None = None



def timed(name: str):
    """Декоратор: пишет длительность узла (в мс) в state['timings'][name]."""
    def deco(fn):
        @wraps(fn)
        def wrapper(state: State) -> State:
            t0 = time.perf_counter()
            try:
                return fn(state)
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000
                timings = state.setdefault("timings", {})
                timings[name] = round(dt_ms, 1)
        return wrapper
    return deco


@timed("embed_query")
@observe(name="embed_query", capture_input=False, capture_output=False)
def embed_query(state: State) -> State:
    try:
        langfuse_context.update_current_observation(input={"query": state.get("query", "")})
    except Exception:
        pass
    try:
        r = requests.post(
            f"{OLLAMA_EMBED_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": state["query"]},
            timeout=60,
        )
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embeddings failed: {e}")

    emb = (r.json().get("embeddings") or [None])[0]
    if not isinstance(emb, list) or not emb:
        raise HTTPException(status_code=502, detail="Embeddings response missing 'embeddings'")

    state["q_embedding"] = emb
    return state


@timed("retrieve")
@observe(name="retrieve", capture_input=False, capture_output=False)
def retrieve(state: State) -> State:
    collections = state.get("collections") or [QDRANT_COLLECTION]
    all_hits = []

    for col in collections:
        try:
            r = requests.post(
                f"{QDRANT_URL}/collections/{col}/points/search",
                json={"vector": state["q_embedding"], "limit": RETRIEVE_TOP_K, "with_payload": True},
                timeout=60,
            )
            r.raise_for_status()
            hits = r.json().get("result", [])
            for h in hits:
                h["_collection"] = col
            all_hits.extend(hits)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Qdrant search failed ({col}): {e}")

    # Сортируем по score и берём RETRIEVE_TOP_K лучших — реранкер потом сократит
    all_hits.sort(key=lambda h: h.get("score", 0), reverse=True)
    state["hits"] = all_hits[:RETRIEVE_TOP_K]
    try:
        langfuse_context.update_current_observation(
            input={"collections": collections, "limit": RETRIEVE_TOP_K},
            output={
                "n_hits": len(state["hits"]),
                "hits": [
                    {
                        "chunk_id": (h.get("payload") or {}).get("chunk_id", h.get("id")),
                        "score": round(h.get("score", 0), 4),
                        "header": (h.get("payload") or {}).get("header_breadcrumb", ""),
                        "page": (h.get("payload") or {}).get("page"),
                        "text": ((h.get("payload") or {}).get("text") or "")[:500],
                    }
                    for h in state["hits"]
                ],
            },
        )
    except Exception:
        pass
    return state


def _hit_to_citation(h: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(h, dict):
        return None
    payload = h.get("payload") or {}
    t = (payload.get("text") or "").strip()
    if not t:
        return None
    return {
        "text": t,
        "header": payload.get("header", ""),
        "header_breadcrumb": payload.get("header_breadcrumb", ""),
        "page": payload.get("page"),
        "type": payload.get("type", ""),
        "source": payload.get("source", "unknown"),
        "chunk_id": payload.get("chunk_id", h.get("id")),
        "score": round(h.get("score", 0), 4),
        "rerank_score": round(h["rerank_score"], 4) if "rerank_score" in h else None,
        "collection": h.get("_collection", ""),
    }


@timed("rerank")
@observe(name="rerank", capture_input=False, capture_output=False)
def rerank(state: State) -> State:
    hits = state.get("hits", [])
    # Снимок «до реранка» — для отладки в UI
    state["citations_pre_rerank"] = [c for c in (_hit_to_citation(h) for h in hits) if c]
    if not hits:
        return state
    docs = [(h.get("payload") or {}).get("text", "") for h in hits]
    try:
        r = requests.post(
            f"{RERANKER_URL}/rerank",
            json={"query": state["query"], "docs": docs, "top_k": RERANKER_TOP_K},
            timeout=300,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Rerank failed: {e}")

    reranked = []
    for item in results:
        h = hits[item["index"]]
        h["rerank_score"] = item["score"]
        reranked.append(h)
    state["hits"] = reranked
    try:
        langfuse_context.update_current_observation(
            input={"query": state.get("query", ""), "n_in": len(docs), "top_k": RERANKER_TOP_K},
            output={
                "n_out": len(reranked),
                "hits": [
                    {
                        "chunk_id": (h.get("payload") or {}).get("chunk_id", h.get("id")),
                        "score": round(h.get("score", 0), 4),
                        "rerank_score": round(h.get("rerank_score", 0), 4),
                        "header": (h.get("payload") or {}).get("header_breadcrumb", ""),
                        "page": (h.get("payload") or {}).get("page"),
                        "text": ((h.get("payload") or {}).get("text") or "")[:500],
                    }
                    for h in reranked
                ],
            },
        )
    except Exception:
        pass
    return state

@timed("build_context")
@observe(name="build_context", capture_input=False, capture_output=False)
def build_context(state: State) -> State:
    citations: List[Dict[str, Any]] = []
    texts: List[str] = []
    for h in state.get("hits", []):
        c = _hit_to_citation(h)
        if c:
            citations.append(c)
            texts.append(c["text"])

    state["context"] = "\n\n---\n\n".join(texts)
    state["citations"] = citations
    try:
        langfuse_context.update_current_observation(
            output={"n_chunks": len(citations), "context_chars": len(state["context"])},
        )
    except Exception:
        pass
    return state


@timed("generate")
@observe(name="generate", as_type="generation", capture_input=False, capture_output=False)
def generate(state: State) -> State:
    history = state.get("history") or []
    history = history[-6:]  # последние 3 пары user/assistant
    history_block = ""
    if history:
        lines = []
        for m in history:
            role = "Пользователь" if m.get("role") == "user" else "Ассистент"
            lines.append(f"{role}: {m.get('content','').strip()}")
        history_block = "Предыдущий диалог:\n" + "\n".join(lines) + "\n\n"

    prompt = (
        "Ты — ассистент по технологической документации. "
        "Отвечай на вопрос, опираясь на приведённый контекст из инструкции. "
        "Контекст состоит из нескольких фрагментов, разделённых '---'. "
        "Цитируй и пересказывай только то, что есть в контексте. "
        "Если ни в одном фрагменте действительно нет нужной информации — ответь "
        "'Не найдено в базе' и предложи уточнение. Не отказывай, если ответ есть хотя бы частично.\n\n"
        f"{history_block}"
        f"Вопрос: {state['query']}\n\n"
        f"Контекст:\n{state.get('context','')}\n\n"
        "Ответ:"
    )

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_ctx": 8192, "temperature": 0.2},
            },
            timeout=600,
        )
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")

    resp = r.json().get("response")
    if not isinstance(resp, str):
        raise HTTPException(status_code=502, detail="Generation response missing 'response'")
    state["answer"] = resp
    try:
        langfuse_context.update_current_observation(
            model=LLM_MODEL,
            input=prompt,
            output=resp,
        )
    except Exception:
        pass
    return state


graph = StateGraph(State)
graph.add_node("embed_query", embed_query)
graph.add_node("retrieve", retrieve)
graph.add_node("rerank", rerank)
graph.add_node("build_context", build_context)
graph.add_node("generate", generate)

graph.set_entry_point("embed_query")
graph.add_edge("embed_query", "retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "build_context")
graph.add_edge("build_context", "generate")
graph.add_edge("generate", END)

app = graph.compile()


@api.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@api.on_event("shutdown")
def _flush_langfuse() -> None:
    try:
        langfuse_context.flush()
    except Exception:
        pass

@api.post("/ask")
@observe(name="ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    q = (req.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    cols = req.collections or [QDRANT_COLLECTION]
    try:
        langfuse_context.update_current_trace(
            input={"query": q, "collections": cols, "history_len": len(req.history or [])},
            metadata={"llm_model": LLM_MODEL, "embed_model": EMBED_MODEL},
        )
    except Exception:
        pass
    t0 = time.perf_counter()
    result = app.invoke({"query": q, "collections": cols, "history": req.history or []})
    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    timings = result.get("timings", {}) or {}
    timings["total"] = total_ms
    try:
        langfuse_context.update_current_trace(
            output={"answer": result.get("answer", ""), "timings": timings},
        )
    except Exception:
        pass
    trace_id = None
    try:
        trace_id = langfuse_context.get_current_trace_id()
    except Exception:
        pass
    return {
        "trace_id": trace_id,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "citations_pre_rerank": result.get("citations_pre_rerank", []),
        "timings": timings,
    }


@api.post("/score")
def score(req: ScoreRequest) -> Dict[str, str]:
    try:
        langfuse_client.score(
            trace_id=req.trace_id,
            name="user_feedback",
            value=req.value,
            data_type="NUMERIC",
            comment=req.comment,
        )
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Score failed: {e}")