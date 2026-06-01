import asyncio
import json
import os
import re
import time
import logging
from functools import wraps
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import requests
from fastapi import FastAPI, HTTPException, Query
from langgraph.graph import END, StateGraph
from fastmcp import Client as McpClient

from pydantic import BaseModel
from langfuse import Langfuse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Заглушаем болтливые родные логи сторонних сервисов/библиотек на каждый запрос
# (HTTP-вызовы, телеметрия Langfuse, внутренние логи MCP-клиента и т.п.),
# оставляя только наши собственные INFO-сообщения.
for _noisy in ("httpx", "httpcore", "langfuse", "qdrant_client", "fastmcp", "mcp", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

langfuse_client = Langfuse()

api = FastAPI()

MCP_DB_URL = os.getenv("MCP_DB_URL", "http://mcp_db:8090/mcp")


async def _call_mcp_async(tool: str, args: dict) -> str:
    async with McpClient(MCP_DB_URL) as client:
        result = await client.call_tool(tool, args)
        if result and hasattr(result, 'content') and result.content:
            return result.content[0].text
        return ""


def call_mcp_tool(tool: str, args: dict) -> str:
    """Synchronous wrapper to call an MCP tool on the mcp_db service."""
    try:
        return asyncio.run(_call_mcp_async(tool, args))
    except Exception as e:
        logger.warning(f"MCP call failed ({tool}): {e}")
        return f"MCP ERROR: {e}"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", OLLAMA_URL)
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "CEN2_all_pages")

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "bambucha/saiga-llama3:8b-q4_K")

RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8080")
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "20"))


# ── Ollama client ─────────────────────────────────────────────────────────────
#
# OOP-концепция здесь: ИНКАПСУЛЯЦИЯ.
# Класс держит вместе СОСТОЯНИЕ (url, модель, дефолтный таймаут) и ПОВЕДЕНИЕ
# (generate / embed) над этим состоянием. Раньше эти данные были разбросаны по
# глобальным переменным, а вызовы requests.post дублировались в 4 местах.
#
# Важное проектное решение: клиент НЕ знает про FastAPI. При ошибке он бросает
# свой собственный OllamaError, а уже вызывающий узел решает, как реагировать
# (превратить в HTTPException, залогировать и вернуть fallback и т.п.).
# Это — разделение ответственности: клиент отвечает за «поговорить с Ollama»,
# а не за то, как ошибка выглядит для пользователя API.

class OllamaError(RuntimeError):
    """Поднимается, когда запрос к Ollama не удался или вернул мусор."""


class OllamaClient:
    def __init__(self, url: str, model: str, *, timeout: int = 60):
        # Это атрибуты экземпляра (instance attributes): у каждого объекта
        # OllamaClient — свои значения. self ссылается на конкретный объект.
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        num_ctx: int,
        temperature: float = 0.0,
        timeout: Optional[int] = None,
    ) -> str:
        """Один вызов /api/generate. Возвращает строку ответа модели."""
        try:
            r = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_ctx": num_ctx, "temperature": temperature},
                },
                timeout=timeout or self.timeout,
            )
            r.raise_for_status()
        except Exception as e:
            raise OllamaError(f"generate failed: {e}") from e

        resp = r.json().get("response")
        if not isinstance(resp, str):
            raise OllamaError("response missing 'response' field")
        return resp

    def embed(self, text: str) -> List[float]:
        """Один вызов /api/embed. Возвращает вектор эмбеддинга."""
        try:
            r = requests.post(
                f"{self.url}/api/embed",
                json={"model": self.model, "input": text},
                timeout=self.timeout,
            )
            r.raise_for_status()
        except Exception as e:
            raise OllamaError(f"embed failed: {e}") from e

        emb = (r.json().get("embeddings") or [None])[0]
        if not isinstance(emb, list) or not emb:
            raise OllamaError("response missing 'embeddings'")
        return emb


# Два отдельных экземпляра — наглядная иллюстрация того, что класс это «шаблон»,
# а объекты независимы: LLM на GPU и embeddings на CPU-инстансе с разными моделями.
llm = OllamaClient(OLLAMA_URL, LLM_MODEL)
embedder = OllamaClient(OLLAMA_EMBED_URL, EMBED_MODEL)


# ── Shared state ────────────────────────────────────────────────────────────

def _merge_timings(left: Dict[str, float], right: Dict[str, float]) -> Dict[str, float]:
    """Reducer: merge timings from parallel branches instead of clobbering."""
    return {**(left or {}), **(right or {})}


class State(TypedDict, total=False):
    query: str
    collections: List[str]
    history: List[Dict[str, str]]
    q_embedding: List[float]
    hits: List[Dict[str, Any]]
    context: str
    db_context: str
    use_db: bool
    tag: Optional[str]
    db_mode: str                # "raw" | "aggregate"
    date_from: Optional[str]
    date_to: Optional[str]
    agents: List[str]           # ["doc"] | ["db"] | ["doc", "db"]
    answer: str
    citations: List[Dict[str, Any]]
    citations_pre_rerank: List[Dict[str, Any]]
    timings: Annotated[Dict[str, float], _merge_timings]

class AskRequest(BaseModel):
    q: str
    collections: List[str] | None = None
    history: List[Dict[str, str]] | None = None
    citations: List[Dict[str, Any]] | None = None


class ScoreRequest(BaseModel):
    trace_id: str
    value: float
    comment: str | None = None


def timed(name: str):
    """Time a node and record it under `name`.

    Works whether the node returns the full state or a partial update dict:
    the timing is written into the returned mapping so LangGraph's `timings`
    reducer merges it (safe across parallel branches).
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(*args) -> State:
            # Поддерживаем и функции-узлы wrapper(state), и методы wrapper(self, state):
            # состояние — всегда последний позиционный аргумент.
            state = args[-1]
            t0 = time.perf_counter()
            result = None
            try:
                result = fn(*args)
                return result
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000
                target = result if isinstance(result, dict) else state
                timings = dict(target.get("timings") or {})
                timings[name] = round(dt_ms, 1)
                target["timings"] = timings
        return wrapper
    return deco


# ── Doc-agent nodes ──────────────────────────────────────────────────────────

@timed("embed_query")
def embed_query(state: State) -> State:
    try:
        state["q_embedding"] = embedder.embed(state["query"])
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=f"Embeddings failed: {e}")
    return state


@timed("retrieve")
def retrieve(state: State) -> State:
    collections = state.get("collections") or [QDRANT_COLLECTION]
    all_hits = []
    PREDEL_ALIASES = {
        "карбонатный": ["карбонатный", "карбонат", "карбоната никеля"],
        "железоочистка": ["железоочистка", "очистка от железа", "железо"],
        "медеочистка": ["медеочистка", "очистка от меди", "медь"],
        "кобальтоочистка": ["кобальтоочистка", "очистка от кобальта", "кобальт"],
        "ОПУ ХВ": ["опу хв", "хлорное выщелачивание", "хлоридное выщелачивание"],
        "сульфатный": ["сульфатный", "прием кислот", "приём кислот", "кислотный раствор"],
    }


    def detect_predel(query: str) -> Optional[str]:
        q = query.lower().replace("ё", "е")

        for predel, aliases in PREDEL_ALIASES.items():
            for alias in aliases:
                if alias.lower().replace("ё", "е") in q:
                    return predel

        return None


    def boost_hits_by_predel(hits: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        detected_predel = detect_predel(query)

        if not detected_predel:
            return hits

        for h in hits:
            payload = h.get("payload") or {}
            predel = payload.get("predel")
            base_score = h.get("score", 0.0)

            h["score_before_predel_boost"] = base_score
            h["detected_predel"] = detected_predel

            if predel == detected_predel:
                h["score"] = base_score * 1.25
                h["predel_boost"] = True
            else:
                h["predel_boost"] = False

        return hits
    
    
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

    all_hits = boost_hits_by_predel(all_hits, state["query"])
    all_hits.sort(key=lambda h: h.get("score", 0), reverse=True)
    state["hits"] = all_hits[:RETRIEVE_TOP_K]
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
        "section_title": payload.get("section_title", ""),
        "section_breadcrumb": payload.get("section_breadcrumb", ""),
        "page": payload.get("page"),
        "page_range": payload.get("page_range", ""),
        "type": payload.get("type", ""),
        "source": payload.get("source", "unknown"),
        "chunk_id": payload.get("chunk_id", h.get("id")),
        "score": round(h.get("score", 0), 4),
        "rerank_score": round(h["rerank_score"], 4) if "rerank_score" in h else None,
        "collection": h.get("_collection", ""),
        "predel": payload.get("predel", ""),
        "predel_boost": h.get("predel_boost", False),
        "detected_predel": h.get("detected_predel"),
        "score_before_predel_boost": round(h.get("score_before_predel_boost", 0), 4)
            if "score_before_predel_boost" in h else None,
    }


@timed("rerank")
def rerank(state: State) -> State:
    hits = state.get("hits", [])
    state["citations_pre_rerank"] = [c for c in (_hit_to_citation(h) for h in hits) if c]
    if not hits:
        return state
    docs = [(h.get("payload") or {}).get("text", "") for h in hits]
    try:
        r = requests.post(
            f"{RERANKER_URL}/rerank",
            json={"query": state["query"], "docs": docs, "top_k": RERANKER_TOP_K},
            timeout=60,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Rerank failed: {e}")

    reranked = []
    for item in results:
        idx = item.get("index")
        if idx is None:
            continue
        h = hits[idx]
        h["rerank_score"] = item.get("score", 0)
        reranked.append(h)
    state["hits"] = reranked
    return state


@timed("build_context")
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
    return state


# ── Tag resolver ──────────────────────────────────────────────────────────────
#
# OOP-концепция здесь: КОМПОЗИЦИЯ («has-a»).
# TagResolver не НАСЛЕДУЕТ OllamaClient — он его ИСПОЛЬЗУЕТ (держит как поле
# self._llm). Это связь «использует/состоит из», а не «является». Правило:
# наследование — для «X является Y», композиция — для «X использует/имеет Y».
#
# Резолвинг тега разбит: mcp_db делает словарный поиск (resolve_tag), а LLM здесь
# только выбирает из кандидатов. TagResolver инкапсулирует оба шага.

_TAG_PICK_PROMPT = """\
Ты — система поиска тегов лабораторных показателей.
Пользователь задал вопрос. Из списка кандидатов выбери один тег, который лучше всего соответствует запросу.
Если ни один не подходит — ответь null.

Ответь ТОЛЬКО валидным JSON без пояснений:
{{"tag": "2141_3052_28"}}  или  {{"tag": null}}

Запрос: {query}

Кандидаты (тег: описание):
{candidates}

JSON:"""


class TagResolver:
    def __init__(self, llm: OllamaClient):
        self._llm = llm  # композиция: резолвер ИСПОЛЬЗУЕТ клиента, а не наследует

    def resolve(self, query: str) -> Optional[str]:
        """Точный матч из mcp_db, иначе — выбор тега LLM из кандидатов."""
        raw = call_mcp_tool("resolve_tag", {"query": query})
        try:
            resolved = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"  [db] resolve_tag bad response: {raw!r}")
            return None

        exact = resolved.get("exact")
        if exact:
            logger.info(f"  [db] тег найден точно: {exact}")
            return exact

        candidates = resolved.get("candidates", [])
        logger.info(f"  [db] точного тега нет, {len(candidates)} кандидатов → LLM")
        tag = self._pick(query, candidates)
        logger.info(f"  [db] LLM выбрал: {tag}")
        return tag

    def _pick(self, query: str, candidates: List[Dict[str, Any]]) -> Optional[str]:
        """Спросить LLM, какой тег из списка кандидатов подходит."""
        if not candidates:
            return None
        valid = {c["tag"] for c in candidates}
        candidates_text = "\n".join(f"  {c['tag']}: {c['desc']}" for c in candidates)
        prompt = _TAG_PICK_PROMPT.format(query=query, candidates=candidates_text)
        try:
            # mcp_db отдаёт весь словарь (~130 тегов ≈ 5k токенов), поэтому контекст
            # должен вместить всех кандидатов, а не top-K срез.
            raw = self._llm.generate(prompt, num_ctx=8192, temperature=0.0).strip()
        except OllamaError as e:
            logger.warning(f"  [db] tag picker LLM failed: {e}")
            return None

        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if not json_match:
            return None
        try:
            tag = json.loads(json_match.group()).get("tag")
            return tag if isinstance(tag, str) and tag in valid else None
        except json.JSONDecodeError:
            return None


tag_resolver = TagResolver(llm)


# ── Orchestrator ─────────────────────────────────────────────────────────────

_ORCHESTRATOR_PROMPT = """\
Ты — оркестратор мульти-агентной системы промышленной документации.
Определи, каких агентов нужно запустить для ответа на запрос пользователя.

Доступные агенты:
- doc: поиск по документации, инструкциям, регламентам
- db:  запрос реальных данных из БД датчиков (нужен тег вида 1234_5678_90 или явный вопрос о значениях/временных рядах)

Ответь ТОЛЬКО валидным JSON без пояснений:
{{"agents": ["doc"]}}          — только документация
{{"agents": ["db"]}}           — только БД
{{"agents": ["doc", "db"]}}    — оба источника

Запрос: {query}
JSON:"""


@timed("orchestrator")
def orchestrator(state: State) -> State:
    """Decide which agents to run by asking the LLM to choose doc / db / both."""
    query = state.get("query", "")

    prompt = _ORCHESTRATOR_PROMPT.format(query=query)
    try:
        raw = llm.generate(prompt, num_ctx=512, temperature=0.0).strip()
    except OllamaError as e:
        logger.warning(f"Orchestrator LLM failed, defaulting to doc+db: {e}")
        raw = '{"agents": ["doc", "db"]}'

    # Extract JSON from response (LLM may add extra text)
    json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
    agents = ["doc", "db"]
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            agents = parsed.get("agents", ["doc", "db"])
            if not isinstance(agents, list):
                agents = ["doc", "db"]
            agents = [a for a in agents if a in ("doc", "db")]
            if not agents:
                agents = ["doc", "db"]
        except json.JSONDecodeError:
            logger.warning(f"Orchestrator JSON parse failed: {raw!r}")

    logger.info(f"Orchestrator decision: {agents}")
    state["agents"] = agents
    return state


# ── Agents ────────────────────────────────────────────────────────────────────
#
# OOP-концепции здесь: АБСТРАКЦИЯ + ПОЛИМОРФИЗМ.
#
# Agent — абстрактный базовый класс (ABC): он задаёт КОНТРАКТ (имя + метод run),
# но не реализацию. Создать Agent() напрямую нельзя — Python запретит из-за
# @abstractmethod. Это и есть смысл абстракции: «вот что должен уметь любой агент».
#
# DocAgent и DbAgent — конкретные реализации. Полиморфизм в том, что оркестратор
# работает со списком Agent одинаково (agent.run(state)), не зная, какой именно
# это агент. Поведение разное, интерфейс один.
#
# Контракт run(): вернуть ЧАСТИЧНЫЙ апдейт состояния (а не полный state) — иначе
# сломается параллельный fan-out и reducer timings.

from abc import ABC, abstractmethod

_AGG_HINTS = ("средн", "минимум", "максимум", "макс", "мин", "динамик", "тренд",
              "сколько в среднем", "разброс", "статистик", "уровень", "типичн")


def _infer_db_mode(query: str) -> str:
    ql = query.lower()
    return "aggregate" if any(h in ql for h in _AGG_HINTS) else "raw"


class Agent(ABC):
    name: str  # каждый подкласс обязан задать своё имя ("doc" / "db")

    @abstractmethod
    def run(self, state: State) -> dict:
        """Выполнить агента, вернуть частичный апдейт состояния."""


class DocAgent(Agent):
    name = "doc"

    def run(self, state: State) -> dict:
        logger.info("  → [doc] поиск по документации...")
        doc_state = doc_app.invoke({
            "query": state["query"],
            "collections": state.get("collections") or [QDRANT_COLLECTION],
            "history": state.get("history") or [],
        })
        timings = {f"doc.{k}": v for k, v in (doc_state.get("timings") or {}).items()}
        logger.info(f"  ✓ [doc] найдено {len(doc_state.get('citations', []))} цитат")
        return {
            "context": doc_state.get("context", ""),
            "citations": doc_state.get("citations", []),
            "citations_pre_rerank": doc_state.get("citations_pre_rerank", []),
            "timings": timings,
        }


class DbAgent(Agent):
    name = "db"

    def __init__(self, resolver: TagResolver):
        # Композиция: DbAgent ИСПОЛЬЗУЕТ TagResolver (который, в свою очередь,
        # использует OllamaClient). Цепочка «has-a», а не наследование.
        self._resolver = resolver
        # Агент ВЛАДЕЕТ своим под-графом (симметрично doc-агенту): узлы под-графа —
        # это методы самого агента. Под-граф строится один раз в конструкторе.
        self._app = self._build_graph()

    def _build_graph(self):
        g = StateGraph(State)
        g.add_node("resolve_tag", self._node_resolve_tag)
        g.add_node("fetch", self._node_fetch)
        g.set_entry_point("resolve_tag")
        g.add_edge("resolve_tag", "fetch")
        g.add_edge("fetch", END)
        return g.compile()

    def run(self, state: State) -> dict:
        logger.info("  → [db] запрос к БД датчиков...")
        db_state = self._app.invoke({
            "query": state["query"],
            "date_from": state.get("date_from"),
            "date_to": state.get("date_to"),
        })
        timings = {f"db.{k}": v for k, v in (db_state.get("timings") or {}).items()}
        logger.info(f"  ✓ [db] получено: {str(db_state.get('db_context',''))[:80]!r}")
        return {
            "db_context": db_state.get("db_context", ""),
            "tag": db_state.get("tag"),
            "db_mode": db_state.get("db_mode", "raw"),
            "timings": timings,
        }

    @timed("resolve_tag")
    def _node_resolve_tag(self, state: State) -> State:
        state["use_db"] = True
        state["db_mode"] = _infer_db_mode(state.get("query", ""))
        state["tag"] = self._resolver.resolve(state.get("query", ""))
        return state

    @timed("fetch")
    def _node_fetch(self, state: State) -> State:
        """Достать сырые значения или сводную статистику из mcp_db."""
        tag = state.get("tag")
        if not tag:
            logger.warning("  [db] тег не найден — пропускаю запрос к БД")
            state["db_context"] = "Не найдено обозначение тега в запросе."
            return state

        args: Dict[str, Any] = {"tag": tag}
        if state.get("date_from"):
            args["date_from"] = state["date_from"]
        if state.get("date_to"):
            args["date_to"] = state["date_to"]

        if state.get("db_mode") == "aggregate":
            logger.info(f"  [db] MCP aggregate {args}")
            result = call_mcp_tool("aggregate", args)
            label = "Сводная статистика из MCP/БД"
        else:
            args["limit"] = 50
            logger.info(f"  [db] MCP values_between {args}")
            result = call_mcp_tool("values_between", args)
            label = "Данные из MCP/БД (временные ряды)"

        state["db_context"] = f"{label}:\n{result}"
        return state


# Реестр агентов по имени — полиморфная диспетчеризация: оркестратор и узлы
# графа берут агента по строке-имени и вызывают .run(), не зная конкретный класс.
AGENTS: Dict[str, Agent] = {a.name: a for a in (DocAgent(), DbAgent(tag_resolver))}


# ── Agent nodes (graph fan-out) ──────────────────────────────────────────────
# doc and db are independent, so the orchestrator fans out to both via conditional
# edges and they run as parallel branches. Each writes disjoint state keys, so
# there are no concurrent-write conflicts on merge. The node functions are thin
# wrappers that delegate to the Agent instances in AGENTS.

def _route_agents(state: State) -> List[str]:
    """Conditional edge: dispatch to the selected agent branches."""
    return list(state.get("agents", ["doc", "db"]))


@timed("doc")
def doc_agent(state: State) -> dict:
    return AGENTS["doc"].run(state)


@timed("db")
def db_agent(state: State) -> dict:
    return AGENTS["db"].run(state)


def join_agents(state: State) -> State:
    """Fan-in barrier: both branches have completed; nothing to merge here."""
    return state


# ── Generate ──────────────────────────────────────────────────────────────────

@timed("generate")
def generate(state: State) -> State:
    history = state.get("history") or []
    history = history[-6:]
    history_block = ""
    if history:
        lines = []
        for m in history:
            role = "Пользователь" if m.get("role") == "user" else "Ассистент"
            lines.append(f"{role}: {m.get('content','').strip()}")
        history_block = "Предыдущий диалог:\n" + "\n".join(lines) + "\n\n"

    db_block = ""
    if state.get("db_context"):
        db_block = f"{state.get('db_context', '')}\n\n"

    prompt = (
        "Ты — ассистент по технологической документации. "
        "Отвечай на вопрос, опираясь на приведённый контекст из инструкции и, если есть, "
        "на данные MCP/БД. "
        "Контекст состоит из нескольких фрагментов, разделённых '---'. "
        "Если используешь данные MCP/БД, явно скажи, что это фактические данные из системы. "
        "Не выдумывай значения. "
        "Если ни в документации, ни в MCP/БД нет нужной информации — ответь "
        "'Не найдено в базе' и предложи уточнение. Не отказывай, если ответ есть хотя бы частично.\n\n"
        f"{history_block}"
        f"Вопрос: {state['query']}\n\n"
        f"Контекст документации:\n{state.get('context','')}\n\n"
        f"{db_block}"
        "Ответ:"
    )

    try:
        # Генерация ответа длиннее прочих вызовов → свой таймаут.
        state["answer"] = llm.generate(
            prompt, num_ctx=4096, temperature=0.1, timeout=600
        )
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")
    return state


# ── Doc sub-graph ─────────────────────────────────────────────────────────────

doc_graph = StateGraph(State)
doc_graph.add_node("embed_query", embed_query)
doc_graph.add_node("retrieve", retrieve)
doc_graph.add_node("rerank", rerank)
doc_graph.add_node("build_context", build_context)

doc_graph.set_entry_point("embed_query")
doc_graph.add_edge("embed_query", "retrieve")
doc_graph.add_edge("retrieve", "rerank")
doc_graph.add_edge("rerank", "build_context")
doc_graph.add_edge("build_context", END)

doc_app = doc_graph.compile()

# ── Main orchestrator graph ───────────────────────────────────────────────────
#
# orchestrator ─┬─(doc)─→ doc_agent ─┐
#               └─(db)──→ db_agent ──┴─→ join_agents → generate → END
# doc_agent and db_agent run as parallel branches; join_agents is the fan-in
# barrier before generation.

graph = StateGraph(State)
graph.add_node("orchestrator", orchestrator)
graph.add_node("doc_agent", doc_agent)
graph.add_node("db_agent", db_agent)
graph.add_node("join_agents", join_agents)
graph.add_node("generate", generate)

graph.set_entry_point("orchestrator")
graph.add_conditional_edges(
    "orchestrator",
    _route_agents,
    {"doc": "doc_agent", "db": "db_agent"},
)
graph.add_edge("doc_agent", "join_agents")
graph.add_edge("db_agent", "join_agents")
graph.add_edge("join_agents", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# ── Reuse-citations graph ─────────────────────────────────────────────────────
# Used by /ask when the caller already has doc citations (the UI does /search then
# /ask). Skips doc retrieval; the orchestrator still decides whether the DB agent
# is needed, then generation runs on the supplied context.

reuse_graph = StateGraph(State)
reuse_graph.add_node("orchestrator", orchestrator)
reuse_graph.add_node("db_agent", db_agent)
reuse_graph.add_node("generate", generate)


def _route_reuse(state: State) -> str:
    """Run the DB agent only if the orchestrator asked for it; else generate."""
    return "db_agent" if "db" in state.get("agents", []) else "generate"


reuse_graph.set_entry_point("orchestrator")
reuse_graph.add_conditional_edges(
    "orchestrator",
    _route_reuse,
    {"db_agent": "db_agent", "generate": "generate"},
)
reuse_graph.add_edge("db_agent", "generate")
reuse_graph.add_edge("generate", END)

reuse_app = reuse_graph.compile()


# ── API endpoints ─────────────────────────────────────────────────────────────

@api.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@api.post("/search")
def search(req: AskRequest) -> Dict[str, Any]:
    """Только поиск без генерации ответа"""
    q = (req.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    cols = req.collections or [QDRANT_COLLECTION]

    trace = langfuse_client.trace(
        name="search",
        input={"query": q, "collections": cols},
        metadata={"embed_model": EMBED_MODEL},
    )
    logger.info(f"🔵 LF trace created: id={trace.id} name=search")
    logger.info(f"━━━ /search START: {q[:50]}... ━━━")
    t0 = time.perf_counter()
    try:
        result = doc_app.invoke({"query": q, "collections": cols, "history": []})
    except HTTPException:
        langfuse_client.flush()
        raise
    except Exception as e:
        logger.exception(f"Search failed: {e}")
        trace.update(output={"error": str(e)})
        langfuse_client.flush()
        raise HTTPException(status_code=502, detail=f"Search error: {str(e)}")

    search_ms = round((time.perf_counter() - t0) * 1000, 1)
    timings = {
        "embed_query": result.get("timings", {}).get("embed_query", 0),
        "retrieve": result.get("timings", {}).get("retrieve", 0),
        "rerank": result.get("timings", {}).get("rerank", 0),
        "build_context": result.get("timings", {}).get("build_context", 0),
        "search_total": search_ms,
    }
    trace.update(output={"n_citations": len(result.get("citations", [])), "timings": timings})
    langfuse_client.flush()
    logger.info(f"🔵 LF trace flushed: id={trace.id}")
    logger.info(f"━━━ /search DONE: {search_ms}ms, {len(result.get('citations', []))} citations ━━━")

    return {
        "query": q,
        "citations": result.get("citations", []),
        "citations_pre_rerank": result.get("citations_pre_rerank", []),
        "timings": timings,
    }


@api.on_event("shutdown")
def _flush_langfuse() -> None:
    langfuse_client.flush()


@api.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    q = (req.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    cols = req.collections or [QDRANT_COLLECTION]

    trace = langfuse_client.trace(
        name="ask",
        input={"query": q, "collections": cols, "history_len": len(req.history or [])},
        metadata={"llm_model": LLM_MODEL, "embed_model": EMBED_MODEL},
    )
    logger.info(f"━━━ /ask START: {q[:50]}... (history={len(req.history or [])} msgs) ━━━")
    t0 = time.perf_counter()
    try:
        if req.citations:
            context = "\n\n---\n\n".join((c.get("text") or "").strip() for c in req.citations if c.get("text"))
            result = reuse_app.invoke({
                "query": q,
                "history": req.history or [],
                "context": context,
                "citations": req.citations,
            })
        else:
            result = app.invoke({"query": q, "collections": cols, "history": req.history or []})
    except HTTPException:
        langfuse_client.flush()
        raise
    except Exception as e:
        logger.exception(f"Graph execution failed: {e}")
        trace.update(output={"error": str(e)})
        langfuse_client.flush()
        raise HTTPException(status_code=502, detail=f"Graph error: {str(e)}")

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"━━━ /ask DONE: {total_ms}ms, answer={len(result.get('answer', ''))} chars, reused_citations={bool(req.citations)} ━━━")
    timings = result.get("timings", {}) or {}
    timings["total"] = total_ms
    trace.update(
        output={"answer": result.get("answer", ""), "timings": timings},
    )
    langfuse_client.flush()
    return {
        "trace_id": trace.id,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "citations_pre_rerank": result.get("citations_pre_rerank", []),
        "timings": timings,
    }


@api.post("/score")
def score(req: ScoreRequest) -> Dict[str, str]:
    langfuse_client.score(
        trace_id=req.trace_id,
        name="user_feedback",
        value=req.value,
        data_type="NUMERIC",
        comment=req.comment,
    )
    return {"status": "ok"}
