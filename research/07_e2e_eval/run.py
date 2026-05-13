"""
End-to-end эвал RAG-пайплайна по golden_qa.jsonl.

Что делает:
  1. Читает датасет вопросов с эталонными ответами и chunk_id
  2. Дёргает /ask на каждом вопросе
  3. Считает простые метрики: chunk_recall@k, наличие ключевых слов в ответе, латентность
  4. Печатает таблицу + markdown-отчёт

Пока без RAGAS — это первый каркас. RAGAS добавится в ragas_runner.py.

Запуск:
    python research/07_e2e_eval/run.py \\
        --api http://localhost:8000 \\
        --dataset research/07_e2e_eval/datasets/golden_qa.jsonl \\
        --report research/07_e2e_eval/reports/latest.md
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


def load_dataset(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def ask(api_url: str, question: str) -> dict[str, Any]:
    r = requests.post(
        f"{api_url}/ask",
        json={"q": question, "history": []},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def chunk_recall_at_k(retrieved_chunk_ids: list[int], expected: list[int], k: int) -> float:
    if not expected:
        return 1.0
    top_k = set(retrieved_chunk_ids[:k])
    return len(set(expected) & top_k) / len(set(expected))


def keyword_overlap(answer: str, expected_answer: str) -> float:
    """Доля слов длиннее 4 символов из эталона, встретившихся в ответе."""
    def words(s: str) -> set[str]:
        return {w.lower() for w in re.findall(r"\w+", s) if len(w) > 4}
    exp = words(expected_answer)
    if not exp:
        return 1.0
    got = words(answer)
    return len(exp & got) / len(exp)


def evaluate(api_url: str, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for item in dataset:
        t0 = time.perf_counter()
        try:
            resp = ask(api_url, item["question"])
            error = None
        except Exception as e:
            resp = {"answer": "", "citations": [], "citations_pre_rerank": []}
            error = str(e)
        latency = round(time.perf_counter() - t0, 2)

        retrieved_after = [c.get("chunk_id") for c in resp.get("citations", [])]
        retrieved_before = [c.get("chunk_id") for c in resp.get("citations_pre_rerank", [])]
        expected = item.get("expected_chunk_ids", [])

        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": resp.get("answer", ""),
            "trace_id": resp.get("trace_id"),
            "error": error,
            "latency_s": latency,
            "recall@5_after_rerank": round(chunk_recall_at_k(retrieved_after, expected, 5), 3),
            "recall@10_before_rerank": round(chunk_recall_at_k(retrieved_before, expected, 10), 3),
            "keyword_overlap": round(keyword_overlap(resp.get("answer", ""), item.get("expected_answer", "")), 3),
        })
    return results


def write_report(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(results)

    def avg(key: str) -> float:
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    md = [
        "# E2E eval report",
        "",
        f"- Вопросов: **{n}**",
        f"- Ошибок: **{sum(1 for r in results if r['error'])}**",
        f"- Средний `recall@5` (after rerank): **{avg('recall@5_after_rerank')}**",
        f"- Средний `recall@10` (before rerank): **{avg('recall@10_before_rerank')}**",
        f"- Средний `keyword_overlap`: **{avg('keyword_overlap')}**",
        f"- Средняя латентность: **{avg('latency_s')} s**",
        "",
        "## По вопросам",
        "",
        "| id | recall@5 | kw | lat | trace |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        md.append(
            f"| {r['id']} | {r['recall@5_after_rerank']} | "
            f"{r['keyword_overlap']} | {r['latency_s']}s | "
            f"`{(r['trace_id'] or '')[:8]}` |"
        )
    path.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--dataset", default="research/07_e2e_eval/datasets/golden_qa.jsonl")
    p.add_argument("--report", default="research/07_e2e_eval/reports/latest.md")
    args = p.parse_args()

    dataset = load_dataset(Path(args.dataset))
    print(f"Loaded {len(dataset)} questions")

    results = evaluate(args.api, dataset)
    write_report(results, Path(args.report))

    print("\nSummary:")
    for r in results:
        mark = "OK" if not r["error"] else "ERR"
        print(f"  [{mark}] {r['id']}  recall@5={r['recall@5_after_rerank']}  kw={r['keyword_overlap']}  {r['latency_s']}s")
    print(f"\nReport: {args.report}")


if __name__ == "__main__":
    main()
