"""RAG pipeline: query rewrite -> hybrid+rerank retrieval -> grounded answer
with [guest @ hh:mm:ss](url) citations. ask() returns the answer plus usage
metadata (tokens, latency, cost) for logging."""

import time

from openai import OpenAI

from fusionrag.search import Search

MODEL = "gpt-5.4-mini"
PRICE_IN = 0.75 / 1e6  # USD per token, gpt-5.4-mini (verified 2026-08-15)
PRICE_OUT = 4.50 / 1e6

REWRITE_PROMPT = """\
Rewrite the user's message as a short standalone search query over spoken
podcast transcripts about nuclear fusion. Keep it close to the original,
resolve pronouns, and spell technical terms the way captions transcribe
speech — acronyms and names often appear as plain words (SPARC -> spark,
Whyte -> white). Include both spellings when unsure. Reply with the query
text only.

User message: {question}"""

STRICT_PROMPT = """\
You answer questions about nuclear fusion using excerpts from Lex Fridman
podcast episodes ({episodes}). Rules:
- Use only the excerpts below; if they don't contain the answer, say so.
- Cite every claim with the matching excerpt's guest, timestamp and link,
  e.g. [Ian Hutchinson @ 01:23:45](https://www.youtube.com/watch?v=xyz&t=5025s)
  — no braces or brackets inside the link text.
- Transcripts are unpunctuated speech; quote meaning, not verbatim text.

Question: {question}

Excerpts:
{excerpts}"""

EXPLANATORY_PROMPT = """\
You answer questions about nuclear fusion using excerpts from Lex Fridman
podcast episodes ({episodes}). Rules:
- Base the answer on the excerpts below; you may add brief background to
  make it understandable, but never contradict or go beyond them on facts.
  If the excerpts don't cover the question, say so.
- Cite every excerpt-based claim with the matching excerpt's guest,
  timestamp and link, e.g.
  [Ian Hutchinson @ 01:23:45](https://www.youtube.com/watch?v=xyz&t=5025s)
  — no braces or brackets inside the link text.
- Transcripts are unpunctuated speech; quote meaning, not verbatim text.

Question: {question}

Excerpts:
{excerpts}"""

ANSWER_PROMPTS = {"strict": STRICT_PROMPT, "explanatory": EXPLANATORY_PROMPT}
# winner of evals/rag_eval.py (mean relevance 1.84 vs 1.80)
DEFAULT_STYLE = "explanatory"


def _hms(sec):
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _format_excerpts(chunks):
    return "\n\n".join(
        f"[{i + 1}] {c['guest']} @ {_hms(c['start_sec'])} | {c['youtube_url']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )


class RAG:
    def __init__(self, search=None):
        self.search = search or Search()
        self.client = OpenAI()
        self.episodes = ", ".join(
            f"{guest} #{ep}"
            for ep, guest in sorted({(c["episode"], c["guest"]) for c in self.search.chunks})
        )

    def _complete(self, prompt):
        resp = self.client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip(), resp.usage

    def ask(self, question, k=5, style=DEFAULT_STYLE):
        t0 = time.time()
        rewritten, usage_rw = self._complete(
            REWRITE_PROMPT.format(question=question)
        )
        # vector won evals/retrieval_eval.py (hit-rate@5 0.787, mrr@5 0.623 —
        # beats keyword/hybrid/rerank on this corpus)
        chunks, scores = self.search.vector_with_scores(rewritten, k=k)
        answer, usage_ans = self._complete(
            ANSWER_PROMPTS[style].format(
                episodes=self.episodes,
                question=question,
                excerpts=_format_excerpts(chunks),
            )
        )
        usages = [usage_rw, usage_ans]
        prompt_tokens = sum(u.prompt_tokens for u in usages)
        completion_tokens = sum(u.completion_tokens for u in usages)
        return {
            "question": question,
            "rewritten_question": rewritten,
            "answer": answer,
            "model": MODEL,
            "chunk_ids": [c["id"] for c in chunks],
            "chunks": chunks,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": prompt_tokens * PRICE_IN + completion_tokens * PRICE_OUT,
            "latency_s": time.time() - t0,
            "avg_retrieval_score": sum(scores) / len(scores) if scores else None,
        }


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "why is helium-3 a good fusion fuel?"
    result = RAG().ask(question)
    print(f"Q: {question}")
    print(f"rewritten: {result['rewritten_question']}\n")
    print(result["answer"])
    print(
        f"\n[{result['prompt_tokens']}+{result['completion_tokens']} tok, "
        f"${result['cost_usd']:.4f}, {result['latency_s']:.1f}s, "
        f"chunks {', '.join(result['chunk_ids'])}]"
    )
