"""
Бенчмарк reranker'ов.

Что делает:
  1. Берёт чанки из Qdrant-коллекции (CEN2_all_pages).
  2. Для каждого вопроса из golden_qa.jsonl:
       - dense retrieval через текущий embedder (bge-m3 из Ollama) → top-RETRIEVE_K
       - каждый reranker пере-ранжирует эти кандидаты
  3. Считает recall@3/5 и MRR ДО и ПОСЛЕ реранка.
  4. Печатает таблицу + пишет отчёт в results/<date>.md.

Запуск:
    docker compose exec -e HF_TOKEN=... notebook-gpu python -u /workspace/research/04_retrieval/reranker_benchmark.py

Env:
    QDRANT_URL          default http://qdrant:6333
    SOURCE_COLLECTION   default CEN2_all_pages
    GOLDEN_QA           default research/07_e2e_eval/datasets/golden_qa.jsonl
    OLLAMA_URL          default http://ollama:11434
    EMBED_MODEL         default bge-m3  (Ollama model name)
    RETRIEVE_K          default 20  (сколько кандидатов для реранка)
    RERANK_K            default 5   (сколько оставить после реранка)
    MODELS              CSV моделей HF; если пусто — берутся дефолты
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    sys.exit("sentence-transformers не установлен. Запусти внутри notebook-gpu.")


# ── Конфиг ──────────────────────────────────────────────────────────────────

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
SOURCE_COLLECTION = os.getenv("SOURCE_COLLECTION", "CEN2_all_pages")
GOLDEN_QA = Path(os.getenv("GOLDEN_QA", "research/07_e2e_eval/datasets/golden_qa.jsonl"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "20"))
RERANK_K = int(os.getenv("RERANK_K", "5"))

DEFAULT_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "DiTy/cross-encoder-russian-msmarco",
    "amberoad/bert-multilingual-passage-reranking-msmarco",
]

models_env = os.getenv("MODELS", "").strip()
MODELS = [m.strip() for m in models_env.split(",") if m.strip()] if models_env else DEFAULT_MODELS


# ── Helpers ─────────────────────────────────────────────────────────────────

def embed_query_ollama(text: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embeddings"][0]


def search_qdrant(vec: list[float], top_k: int) -> list[dict]:
    r = requests.post(
        f"{QDRANT_URL}/collections/{SOURCE_COLLECTION}/points/search",
        json={"vector": vec, "limit": top_k, "with_payload": True},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("result", [])


def load_golden_qa(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ── Метрики ─────────────────────────────────────────────────────────────────

def recall_at_k(ranked_ids: list[int], expected: list[int], k: int) -> float:
    if not expected:
        return 0.0
    top = set(ranked_ids[:k])
    return len(set(expected) & top) / len(set(expected))


def mrr(ranked_ids: list[int], expected: list[int]) -> float:
    exp_set = set(expected)
    for rank, h in enumerate(ranked_ids, start=1):
        if h in exp_set:
            return 1.0 / rank
    return 0.0


@dataclass
class RerankerResult:
    model: str
    load_sec: float
    avg_rerank_sec: float
    recall_at_3_before: float
    recall_at_3_after: float
    recall_at_5_before: float
    recall_at_5_after: float
    mrr_before: float
    mrr_after: float


# ── Основной прогон ─────────────────────────────────────────────────────────

def retrieve_candidates(qa: list[dict]) -> list[dict]:
    """Для каждого вопроса получаем top-RETRIEVE_K от dense retriever."""
    results = []
    for q in qa:
        vec = embed_query_ollama(q["question"])
        hits = search_qdrant(vec, RETRIEVE_K)
        candidates = []
        for h in hits:
            payload = h.get("payload", {})
            candidates.append({
                "chunk_id": payload.get("chunk_id", h.get("id")),
                "text": payload.get("text", ""),
                "score": h.get("score", 0.0),
            })
        results.append({
            "question": q["question"],
            "expected_chunk_ids": [int(x) for x in q.get("expected_chunk_ids", [])],
            "candidates": candidates,
        })
    return results


def evaluate_reranker(model_name: str, retrieval_results: list[dict]) -> RerankerResult:
    print(f"\n=== {model_name} ===")

    print("  loading model...")
    t0 = time.perf_counter()
    reranker = CrossEncoder(model_name, trust_remote_code=True)
    load_sec = time.perf_counter() - t0
    print(f"  loaded in {load_sec:.1f}s")

    r3_before_list, r5_before_list, mrr_before_list = [], [], []
    r3_after_list, r5_after_list, mrr_after_list = [], [], []
    rerank_times = []

    for item in retrieval_results:
        question = item["question"]
        expected = item["expected_chunk_ids"]
        candidates = item["candidates"]

        # before rerank — порядок от dense retriever
        before_ids = [int(c["chunk_id"]) for c in candidates]
        r3_before_list.append(recall_at_k(before_ids, expected, 3))
        r5_before_list.append(recall_at_k(before_ids, expected, 5))
        mrr_before_list.append(mrr(before_ids, expected))

        # rerank
        pairs = [[question, c["text"]] for c in candidates]
        t0 = time.perf_counter()
        scores = reranker.predict(pairs)
        rerank_times.append(time.perf_counter() - t0)

        # sort by reranker score desc
        scored = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        after_ids = [int(c["chunk_id"]) for c, _ in scored]

        r3_after_list.append(recall_at_k(after_ids, expected, 3))
        r5_after_list.append(recall_at_k(after_ids, expected, 5))
        mrr_after_list.append(mrr(after_ids, expected))

    def avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    result = RerankerResult(
        model=model_name,
        load_sec=round(load_sec, 1),
        avg_rerank_sec=round(sum(rerank_times) / len(rerank_times), 3) if rerank_times else 0.0,
        recall_at_3_before=avg(r3_before_list),
        recall_at_3_after=avg(r3_after_list),
        recall_at_5_before=avg(r5_before_list),
        recall_at_5_after=avg(r5_after_list),
        mrr_before=avg(mrr_before_list),
        mrr_after=avg(mrr_after_list),
    )
    print(
        f"  R@3: {result.recall_at_3_before} -> {result.recall_at_3_after}  "
        f"R@5: {result.recall_at_5_before} -> {result.recall_at_5_after}  "
        f"MRR: {result.mrr_before} -> {result.mrr_after}  "
        f"avg rerank: {result.avg_rerank_sec:.3f}s"
    )
    return result


def write_report(results: list[RerankerResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Reranker benchmark — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"- Source collection: `{SOURCE_COLLECTION}`",
        f"- Retriever: Ollama `{EMBED_MODEL}` → dense top-{RETRIEVE_K}",
        f"- Rerank top-K: {RERANK_K}",
        f"- Questions: `{GOLDEN_QA}`",
        "",
        "| model | load_s | rerank_s | R@3 before | R@3 after | R@5 before | R@5 after | MRR before | MRR after |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: -x.mrr_after):
        lines.append(
            f"| `{r.model}` | {r.load_sec} | {r.avg_rerank_sec} | "
            f"{r.recall_at_3_before} | {r.recall_at_3_after} | "
            f"{r.recall_at_5_before} | {r.recall_at_5_after} | "
            f"{r.mrr_before} | {r.mrr_after} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not GOLDEN_QA.exists():
        sys.exit(f"golden_qa не найден: {GOLDEN_QA}")

    qa = load_golden_qa(GOLDEN_QA)
    print(f"Loaded {len(qa)} golden QA items")

    print(f"Retrieving top-{RETRIEVE_K} candidates via {EMBED_MODEL}...")
    retrieval_results = retrieve_candidates(qa)
    print(f"Got candidates for {len(retrieval_results)} questions")

    results: list[RerankerResult] = []
    for model_name in MODELS:
        try:
            res = evaluate_reranker(model_name, retrieval_results)
            results.append(res)
        except Exception as e:
            print(f"  !! failed: {e}")

    if not results:
        sys.exit("Нет результатов.")

    report_path = Path("research/04_retrieval/results") / f"rerankers_{datetime.now().strftime('%Y-%m-%d_%H%M')}.md"
    write_report(results, report_path)

    print(f"\n=== Summary (sorted by MRR after) ===")
    header = f"{'model':<55} {'R@3':>6} {'->':>4} {'R@3':>6} {'R@5':>6} {'->':>4} {'R@5':>6} {'MRR':>6} {'->':>4} {'MRR':>6}"
    print(header)
    for r in sorted(results, key=lambda x: -x.mrr_after):
        print(
            f"{r.model:<55} {r.recall_at_3_before:>6} {'->':>4} {r.recall_at_3_after:>6} "
            f"{r.recall_at_5_before:>6} {'->':>4} {r.recall_at_5_after:>6} "
            f"{r.mrr_before:>6} {'->':>4} {r.mrr_after:>6}"
        )
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
