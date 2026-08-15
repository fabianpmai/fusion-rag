"""Generate the retrieval ground truth: sample chunks, have the LLM write
3 questions each chunk answers -> evals/ground_truth.csv (question, chunk_id).

    uv run --env-file .env evals/generate_ground_truth.py
"""

import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from fusionrag.rag import MODEL
from fusionrag.search import DATA

EVALS = Path(__file__).resolve().parent
SAMPLE_SIZE = 100
SEED = 42

PROMPT = """\
Below is an excerpt from a podcast transcript about nuclear fusion. Write 3
questions a listener could ask whose answers are contained in this excerpt.
Use natural wording, don't copy phrases from the excerpt, and make each
question understandable on its own (no "he", "this", "the excerpt").

Excerpt:
{text}

Reply with JSON: {{"questions": ["...", "...", "..."]}}"""

client = OpenAI()


def questions_for(chunk):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(text=chunk["text"])}],
        response_format={"type": "json_object"},
    )
    try:
        qs = json.loads(resp.choices[0].message.content)["questions"]
    except (KeyError, TypeError):
        print(f"  skipping {chunk['id']}: malformed reply")
        return []
    return [(q, chunk["id"]) for q in qs[:3]]


def main():
    chunks = json.loads((DATA / "chunks.json").read_text())
    pool = [c for c in chunks if len(c["text"]) > 500]
    sample = random.Random(SEED).sample(pool, SAMPLE_SIZE)

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [row for rows in ex.map(questions_for, sample) for row in rows]

    out = EVALS / "ground_truth.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "chunk_id"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} questions for {SAMPLE_SIZE} chunks -> {out}")


if __name__ == "__main__":
    main()
