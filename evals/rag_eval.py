"""Evaluate full RAG answers for two prompt styles with an LLM judge
(relevance 0-2, judge sees question + answer + source chunk).
-> evals/results/rag.csv (summary) and rag_details.csv (per question).

    uv run --env-file .env evals/rag_eval.py
"""

import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from fusionrag.rag import ANSWER_PROMPTS, MODEL, RAG

EVALS = Path(__file__).resolve().parent
SAMPLE_SIZE = 50
SEED = 42

JUDGE_PROMPT = """\
Judge a Q&A system's answer.

Question: {question}

Answer given by the system:
{answer}

Transcript excerpt the question was generated from (the expected source):
{source}

Score the answer's relevance to the question:
2 = relevant, answers the question
1 = partially relevant or incomplete
0 = irrelevant or fails to answer

Reply with JSON: {{"score": <0|1|2>}}"""

client = OpenAI()
rag = RAG()
by_id = rag.search.by_id


def judge(question, answer, truth_id):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question, answer=answer, source=by_id[truth_id]["text"]
            ),
        }],
        response_format={"type": "json_object"},
    )
    return int(json.loads(resp.choices[0].message.content)["score"])


def evaluate(style, t):
    out = rag.ask(t["question"], style=style)
    score = judge(t["question"], out["answer"], t["chunk_id"])
    return {
        "style": style,
        "question": t["question"],
        "chunk_id": t["chunk_id"],
        "score": score,
        "cost_usd": round(out["cost_usd"], 5),
    }


def main():
    with (EVALS / "ground_truth.csv").open() as f:
        truth = list(csv.DictReader(f))
    sample = random.Random(SEED).sample(truth, SAMPLE_SIZE)

    details, summary = [], []
    for style in ANSWER_PROMPTS:
        with ThreadPoolExecutor(max_workers=4) as ex:
            rows = list(ex.map(lambda t: evaluate(style, t), sample))
        details.extend(rows)
        scores = [r["score"] for r in rows]
        summary.append({
            "style": style,
            "mean_relevance": round(sum(scores) / len(scores), 3),
            "share_2": round(scores.count(2) / len(scores), 3),
            "share_1": round(scores.count(1) / len(scores), 3),
            "share_0": round(scores.count(0) / len(scores), 3),
        })
        print(summary[-1])

    results = EVALS / "results"
    results.mkdir(exist_ok=True)
    for name, rows in [("rag.csv", summary), ("rag_details.csv", details)]:
        with (results / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"-> {results}/rag.csv, rag_details.csv")


if __name__ == "__main__":
    main()
