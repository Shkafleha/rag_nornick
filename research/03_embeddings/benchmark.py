"""
Бенчмарк embedder'ов на твоих чанках.

Что делает:
  1. Берёт чанки из существующей Qdrant-коллекции (default: CEN2_all_pages)
     — они уже распарсены и разбиты, от модели зависит только embedding.
  2. Для каждой модели из MODELS:
       - загружает через sentence_transformers
       - эмбеддит все чанки
       - создаёт временную коллекцию bench_<slug> в Qdrant
       - прогоняет вопросы из golden_qa.jsonl
       - считает recall@1, recall@5, recall@10, MRR
  3. Печатает сравнительную таблицу + пишет в results/<date>.md.

Запуск (из контейнера notebook-gpu, где есть torch/sentence-transformers):

    docker compose exec notebook-gpu python /workspace/research/03_embeddings/benchmark.py

Либо локально, если установлены torch + sentence-transformers + qdrant-client:

    python research/03_embeddings/benchmark.py

Env:
    QDRANT_URL          default http://qdrant:6333 (из контейнера) / http://localhost:6333 (хост)
    SOURCE_COLLECTION   default CEN2_all_pages   — откуда брать чанки
    GOLDEN_QA           default research/07_e2e_eval/datasets/golden_qa.jsonl
    MODELS              CSV моделей HF; если пусто — берутся дефолты ниже
    TOP_K               default 10
    KEEP_COLLECTIONS    "1" чтобы не удалять bench_* после прогона
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("sentence-transformers не установлен. Запусти внутри notebook-gpu или `pip install sentence-transformers`.")


# ── Конфиг ──────────────────────────────────────────────────────────────────

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
SOURCE_COLLECTION = os.getenv("SOURCE_COLLECTION", "CEN2_all_pages")
GOLDEN_QA = Path(os.getenv("GOLDEN_QA", "research/07_e2e_eval/datasets/golden_qa.jsonl"))
TOP_K = int(os.getenv("TOP_K", "10"))
KEEP_COLLECTIONS = os.getenv("KEEP_COLLECTIONS", "0") == "1"

DEFAULT_MODELS = [
    "BAAI/bge-m3",                                              # текущий прод, мультиязычный SOTA
    "deepvk/USER-bge-m3",                                       # русский fine-tune BGE-M3
    "intfloat/multilingual-e5-large",                           # проверенная мультиязычная
    "sergeyzh/LaBSE-ru-turbo",                                  # быстрая русская
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # лёгкий baseline
]

models_env = os.getenv("MODELS", "").strip()
MODELS = [m.strip() for m in models_env.split(",") if m.strip()] if models_env else DEFAULT_MODELS


# ── Qdrant helpers ──────────────────────────────────────────────────────────

def scroll_all_chunks(collection: str) -> list[dict]:
    """Выгружает все точки коллекции с payload."""
    out: list[dict] = []
    next_offset = None
    while True:
        body: dict = {"limit": 512, "with_payload": True, "with_vector": False}
        if next_offset is not None:
            body["offset"] = next_offset
        r = requests.post(f"{QDRANT_URL}/collections/{collection}/points/scroll", json=body, timeout=60)
        r.raise_for_status()
        data = r.json().get("result", {})
        pts = data.get("points", [])
        out.extend(pts)
        next_offset = data.get("next_page_offset")
        if next_offset is None:
            break
    return out


def recreate_collection(name: str, vector_size: int) -> None:
    requests.delete(f"{QDRANT_URL}/collections/{name}", timeout=30)
    r = requests.put(
        f"{QDRANT_URL}/collections/{name}",
        json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        timeout=30,
    )
    r.raise_for_status()


def upload_points(name: str, ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None:
    batch = 256
    for i in range(0, len(ids), batch):
        points = [
            {"id": int(ids[j]), "vector": vectors[j], "payload": payloads[j]}
            for j in range(i, min(i + batch, len(ids)))
        ]
        r = requests.put(
            f"{QDRANT_URL}/collections/{name}/points?wait=true",
            json={"points": points},
            timeout=300,
        )
        r.raise_for_status()


def search(name: str, vec: list[float], top_k: int) -> list[dict]:
    r = requests.post(
        f"{QDRANT_URL}/collections/{name}/points/search",
        json={"vector": vec, "limit": top_k, "with_payload": True},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("result", [])


def delete_collection(name: str) -> None:
    requests.delete(f"{QDRANT_URL}/collections/{name}", timeout=30)


# ── Метрики ─────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    model: str
    dim: int
    n_chunks: int
    embed_sec: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float


def compute_metrics(
    hits_per_q: list[list[int]],
    expected_per_q: list[list[int]],
    top_k: int,
) -> tuple[float, float, float, float]:
    def recall_at(k: int) -> float:
        if not expected_per_q:
            return 0.0
        vals = []
        for hits, exp in zip(hits_per_q, expected_per_q):
            if not exp:
                continue
            top = set(hits[:k])
            vals.append(len(set(exp) & top) / len(set(exp)))
        return sum(vals) / len(vals) if vals else 0.0

    def mrr() -> float:
        vals = []
        for hits, exp in zip(hits_per_q, expected_per_q):
            exp_set = set(exp)
            rr = 0.0
            for rank, h in enumerate(hits, start=1):
                if h in exp_set:
                    rr = 1.0 / rank
                    break
            vals.append(rr)
        return sum(vals) / len(vals) if vals else 0.0

    return recall_at(1), recall_at(5), recall_at(min(10, top_k)), mrr()


# ── Основной прогон ─────────────────────────────────────────────────────────

def slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def load_golden_qa(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def evaluate_model(
    model_name: str,
    chunks_text: list[str],
    chunks_payload: list[dict],
    chunks_ids: list[int],
    qa: list[dict],
) -> EvalResult:
    print(f"\n=== {model_name} ===")
    print("  loading model...")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    print(f"  embedding {len(chunks_text)} chunks...")
    t0 = time.perf_counter()
    embs = model.encode(
        chunks_text,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embed_sec = time.perf_counter() - t0
    dim = int(embs.shape[1])
    print(f"  done in {embed_sec:.1f}s, dim={dim}")

    coll = f"bench_{slug(model_name)}"
    recreate_collection(coll, vector_size=dim)
    upload_points(coll, chunks_ids, [e.tolist() for e in embs], chunks_payload)

    print(f"  running {len(qa)} golden_qa questions...")
    q_embs = model.encode(
        [q["question"] for q in qa],
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    hits_per_q: list[list[int]] = []
    expected_per_q: list[list[int]] = []
    for q, qe in zip(qa, q_embs):
        res = search(coll, qe.tolist(), TOP_K)
        retrieved_ids = [int(h.get("payload", {}).get("chunk_id", h.get("id"))) for h in res]
        hits_per_q.append(retrieved_ids)
        expected_per_q.append([int(x) for x in q.get("expected_chunk_ids", [])])

    r1, r5, r10, mrr = compute_metrics(hits_per_q, expected_per_q, TOP_K)

    if not KEEP_COLLECTIONS:
        delete_collection(coll)

    return EvalResult(
        model=model_name,
        dim=dim,
        n_chunks=len(chunks_text),
        embed_sec=round(embed_sec, 1),
        recall_at_1=round(r1, 3),
        recall_at_5=round(r5, 3),
        recall_at_10=round(r10, 3),
        mrr=round(mrr, 3),
    )


def write_report(results: list[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Embedder benchmark — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"- Source collection: `{SOURCE_COLLECTION}`",
        f"- Chunks: {results[0].n_chunks if results else 0}",
        f"- Questions: loaded from `{GOLDEN_QA}`",
        f"- Top-K: {TOP_K}",
        "",
        "| model | dim | embed_s | R@1 | R@5 | R@10 | MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: -x.mrr):
        lines.append(
            f"| `{r.model}` | {r.dim} | {r.embed_sec} | "
            f"{r.recall_at_1} | {r.recall_at_5} | {r.recall_at_10} | {r.mrr} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not GOLDEN_QA.exists():
        sys.exit(f"golden_qa не найден: {GOLDEN_QA}")

    qa = load_golden_qa(GOLDEN_QA)
    print(f"Loaded {len(qa)} golden QA items")

    print(f"Scrolling {SOURCE_COLLECTION} chunks from Qdrant...")
    points = scroll_all_chunks(SOURCE_COLLECTION)
    if not points:
        sys.exit(f"Коллекция {SOURCE_COLLECTION} пуста или не существует.")

    chunks_text: list[str] = []
    chunks_payload: list[dict] = []
    chunks_ids: list[int] = []
    for i, p in enumerate(points):
        payload = p.get("payload") or {}
        text = (payload.get("text") or "").strip()
        if not text:
            continue
        chunks_text.append(text)
        chunks_payload.append(payload)
        # Используем последовательный int id для bench-коллекций (Qdrant не любит строковые id без настройки)
        chunks_ids.append(i)
    print(f"Got {len(chunks_text)} non-empty chunks")

    results: list[EvalResult] = []
    for model_name in MODELS:
        try:
            res = evaluate_model(model_name, chunks_text, chunks_payload, chunks_ids, qa)
            results.append(res)
            print(
                f"  -> R@1={res.recall_at_1}  R@5={res.recall_at_5}  "
                f"R@10={res.recall_at_10}  MRR={res.mrr}"
            )
        except Exception as e:
            print(f"  !! failed: {e}")

    if not results:
        sys.exit("Нет результатов.")

    report_path = Path("research/03_embeddings/results") / f"{datetime.now().strftime('%Y-%m-%d_%H%M')}.md"
    write_report(results, report_path)

    print("\n=== Summary (sorted by MRR) ===")
    print(f"{'model':<55} {'dim':>5} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6}")
    for r in sorted(results, key=lambda x: -x.mrr):
        print(f"{r.model:<55} {r.dim:>5} {r.recall_at_1:>6} {r.recall_at_5:>6} {r.recall_at_10:>6} {r.mrr:>6}")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
