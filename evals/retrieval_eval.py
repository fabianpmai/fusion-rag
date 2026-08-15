"""Evaluate the four retrieval variants on hit-rate@5 and MRR@5 against
evals/ground_truth.csv -> evals/results/retrieval.csv.

Because chunks overlap 50%, the adjacent chunk (same episode, index +-1)
contains the same answer text and counts as a hit.

    uv run --env-file .env evals/retrieval_eval.py
"""

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fusionrag.search import Search

EVALS = Path(__file__).resolve().parent
K = 5
VARIANTS = ["keyword", "vector", "hybrid", "rerank"]

search = Search()


def is_match(result_id, truth_id):
    ep_r, idx_r = result_id.split("-")
    ep_t, idx_t = truth_id.split("-")
    return ep_r == ep_t and abs(int(idx_r) - int(idx_t)) <= 1


def reciprocal_rank(fn, question, truth_id):
    results = fn(question, k=K)
    for rank, chunk in enumerate(results):
        if is_match(chunk["id"], truth_id):
            return 1 / (rank + 1)
    return 0.0


def main():
    with (EVALS / "ground_truth.csv").open() as f:
        truth = list(csv.DictReader(f))
    print(f"{len(truth)} questions")

    rows = []
    for name in VARIANTS:
        fn = getattr(search, name)
        with ThreadPoolExecutor(max_workers=4) as ex:
            rrs = list(
                ex.map(lambda t: reciprocal_rank(fn, t["question"], t["chunk_id"]), truth)
            )
        hit_rate = sum(rr > 0 for rr in rrs) / len(rrs)
        mrr = sum(rrs) / len(rrs)
        rows.append({"variant": name, "hit_rate@5": round(hit_rate, 3), "mrr@5": round(mrr, 3)})
        print(f"{name:10s}  hit-rate@5 {hit_rate:.3f}  mrr@5 {mrr:.3f}")

    out = EVALS / "results" / "retrieval.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
