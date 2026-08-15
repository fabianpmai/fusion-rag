"""Keyword, vector, hybrid (RRF) and LLM-reranked search over data/.

Chunks overlap 50%, so every variant skips adjacent chunks (same episode,
index +-1) when filling top-k — results are distinct passages.
"""

import json
import os
from pathlib import Path

import numpy as np
from minsearch import Index
from openai import OpenAI

from fusionrag import adjacent
from fusionrag.embedder import Embedder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RRF_K = 60
RERANK_MODEL = "gpt-5.4-mini"


def _dedup_adjacent(ranked, k):
    picked = []
    for c in ranked:
        if any(adjacent(p["id"], c["id"]) for p in picked):
            continue
        picked.append(c)
        if len(picked) == k:
            break
    return picked


def _llm_scores(query, candidates):
    passages = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(candidates))
    prompt = (
        "Score how well each passage answers the question, 0 (irrelevant) "
        "to 10 (directly answers it). Passages are podcast transcript "
        f"excerpts.\n\nQuestion: {query}\n\n{passages}\n\n"
        f'Reply with JSON: {{"scores": [{len(candidates)} numbers, one per passage in order]}}'
    )
    resp = OpenAI().chat.completions.create(
        model=RERANK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    try:
        scores = [float(s) for s in json.loads(resp.choices[0].message.content)["scores"]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(scores) != len(candidates):
        return None
    return scores


class Search:
    def __init__(self):
        self.chunks = json.loads((DATA / "chunks.json").read_text())
        self.by_id = {c["id"]: c for c in self.chunks}
        npz = np.load(DATA / "embeddings.npz")
        self.embeddings = npz["embeddings"]
        self.ids = [str(cid) for cid in npz["chunk_ids"]]
        self.index = Index(text_fields=["text", "title", "guest"]).fit(self.chunks)
        self.embedder = Embedder()

    def keyword(self, query, k=5):
        docs = self.index.search(query, num_results=6 * k)
        return _dedup_adjacent(docs, k)

    def vector_with_scores(self, query, k=5):
        """Top-k chunks plus their cosine scores, in rank order."""
        scores = self.embeddings @ self.embedder.encode(query)
        order = np.argsort(-scores)[: 6 * k]
        picked = _dedup_adjacent([self.by_id[self.ids[i]] for i in order], k)
        by_id_score = {self.ids[i]: float(scores[i]) for i in order}
        return picked, [by_id_score[c["id"]] for c in picked]

    def vector(self, query, k=5):
        return self.vector_with_scores(query, k)[0]

    def hybrid(self, query, k=5):
        rrf = {}
        for rank, doc in enumerate(self.index.search(query, num_results=30)):
            rrf[doc["id"]] = rrf.get(doc["id"], 0) + 1 / (RRF_K + rank)
        scores = self.embeddings @ self.embedder.encode(query)
        for rank, i in enumerate(np.argsort(-scores)[:30]):
            cid = self.ids[i]
            rrf[cid] = rrf.get(cid, 0) + 1 / (RRF_K + rank)
        ranked = sorted(rrf, key=rrf.get, reverse=True)
        return _dedup_adjacent([self.by_id[cid] for cid in ranked], k)

    def rerank(self, query, k=5):
        candidates = self.hybrid(query, k=10)
        scores = _llm_scores(query, candidates)
        if scores is None:
            return candidates[:k]
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        return [candidates[i] for i in order[:k]]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "how do you keep the plasma from touching the wall"
    search = Search()
    variants = ["keyword", "vector", "hybrid"]
    if os.environ.get("OPENAI_API_KEY"):
        variants.append("rerank")
    for name in variants:
        print(f"== {name}: {query!r}")
        for c in getattr(search, name)(query, k=5):
            print(f"  {c['id']}  {c['youtube_url']}")
            print(f"    {c['text'][:100]}")
        print()
