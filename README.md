# fusion-rag

Ask three fusion experts anything. A retrieval-augmented generation (RAG)
application over the transcripts of three Lex Fridman podcast episodes on
nuclear fusion — every answer cites the exact moment in the episode, so you
can jump into the video and hear the expert say it.

## Why I built this

I spend a good part of my free time with science podcasts — Lex Fridman
above all — and the topics I keep returning to are physics and the future
of energy. Nuclear fusion sits exactly at that intersection, and it's at a
fascinating moment: private companies are promising grid electricity within
a decade, while the old joke that "fusion is always 30 years away" refuses
to die.

The three episodes in this corpus are, to me, the best publicly available
snapshot of that tension. They span three very different
vantage points: an MIT plasma physicist explaining the fundamentals (2020),
the director of MIT's Plasma Science & Fusion Center as the private fusion
wave took off (2023), and the CEO of Helion — a startup betting on a
completely different reactor design (2025). Together that's nearly eight
hours of substantive, sometimes contradicting, state-of-the-art views
spanning five years of a fast-moving field.

The problem: eight hours of audio is where knowledge goes to hibernate.
When I want to recall what was *actually said* — how Whyte explained
SPARC's high-field magnets, or what Kirtley claims about direct energy
recovery from helium-3 — I'd have to scrub through video. This project
turns those conversations into something I can query: ask in plain
language, get an answer grounded in what the guests really said, with
timestamped links as evidence instead of vibes.

## The corpus

| Episode | Guest | Perspective |
|---|---|---|
| [#112 (2020)](https://www.youtube.com/watch?v=pDSEjaDCtOU) | Ian Hutchinson | plasma physicist, MIT — the fundamentals |
| [#353 (2023)](https://www.youtube.com/watch?v=aJoRMFWn2Jk) | Dennis Whyte | director, MIT Plasma Science & Fusion Center — tokamaks, SPARC |
| [#485 (2025)](https://www.youtube.com/watch?v=m_CFCyc2Shs) | David Kirtley | CEO, Helion Energy — pulsed fusion, commercialization |

~7.7 hours of conversation, 422 searchable chunks after cleaning.

## Architecture

```mermaid
flowchart TB
    subgraph ingestion["Ingestion — one-off, outputs committed to the repo"]
        A["YouTube transcripts<br/>(youtube-transcript-api)"] --> B["Sponsor/ad removal<br/>(SponsorBlock API + marker cut)"]
        B --> C["Chunking<br/>(~2000 chars, 50% overlap,<br/>timestamps preserved)"]
        C --> D["Embedding<br/>(ONNX all-MiniLM-L6-v2)"]
        C --> E[("data/chunks.json")]
        D --> F[("data/embeddings.npz")]
    end

    subgraph query["Query path — per question"]
        Q["User question"] --> RW["LLM query rewrite<br/>(caption-spelling aware)"]
        RW --> VS["Vector search<br/>(numpy cosine, top-5,<br/>overlap-aware dedup)"]
        VS --> P["Grounded prompt<br/>(top-5 excerpts)"]
        P --> ANS["Answer with timestamped<br/>guest @ hh:mm:ss citations"]
    end

    E -.-> VS
    F -.-> VS

    subgraph monitoring["Monitoring — Streamlit pages"]
        PG[("Postgres<br/>conversations + feedback")]
        DASH["dashboard<br/>(stat tiles + 6 charts)"]
        TBL["database<br/>(raw table view)"]
    end

    ANS --> PG
    FB["Feedback 👍 / 👎"] --> PG
    PG --> DASH
    PG --> TBL
```

Design choices worth naming:

- **No vector database.** 422 chunks × 384 dimensions is a single numpy
  matmul — exact, instant, zero infrastructure. Embeddings run through ONNX
  (no torch, no GPU), which keeps the Docker image small.
- **Four retrieval variants implemented** — keyword (minsearch), vector
  (cosine), hybrid (reciprocal rank fusion), LLM rerank — and the app uses
  the measured winner, not the assumed one (see Evaluation).
- **Query rewriting for caption reality.** Auto-captions write "spark" for
  SPARC and "Dennis White" for Whyte; an LLM rewrite maps the user's
  wording to how the transcripts actually spell things.
- **Overlap-aware retrieval.** Chunks overlap 50% for answer continuity, so
  result lists skip adjacent chunks — the top-5 are five distinct passages.
- **LLM**: OpenAI `gpt-5.4-mini` for rewrite, answer, rerank, and the
  evaluation judges. A typical question costs ~$0.004 and takes ~4 s.

## How to run

Prerequisites: Docker (with compose), an OpenAI API key.

```bash
git clone https://github.com/fabianpmai/fusion-rag.git && cd fusion-rag
cp .env.example .env          # put your OPENAI_API_KEY in .env
docker compose up --build
```

Open http://localhost:8501 — `app` is the chat, `dashboard` is monitoring,
`database` shows the raw tables. Postgres starts first (healthcheck), the
app waits for it, tables are created automatically, and the embedding model
is baked into the image — nothing downloads at runtime.

Local development without Docker (needs [uv](https://docs.astral.sh/uv/)
and a Postgres on localhost:5432):

```bash
uv run --env-file .env streamlit run app.py
```

The ingested data (`data/`) is committed, so nothing needs to be fetched
from YouTube. To reproduce it from scratch anyway:

```bash
uv run -m fusionrag.ingest
```

The script is idempotent: raw transcripts and SponsorBlock responses are
cached in `data/raw/` and reused offline. A health check of the whole
pipeline (data integrity, all search variants, one live cited answer):

```bash
uv run --env-file .env smoke.py
```

## Evaluation

### Retrieval

Ground truth: 100 randomly sampled chunks × 3 LLM-generated questions each
(`evals/ground_truth.csv`, 300 pairs). A retrieval counts as a hit if the
source chunk (or its 50%-overlapping neighbor) appears in the top 5.
Reproduce: `uv run --env-file .env evals/retrieval_eval.py`.

| Variant | Hit-rate@5 | MRR@5 |
|---|---|---|
| keyword (minsearch) | 0.553 | 0.465 |
| **vector (cosine)** | **0.787** | **0.623** |
| hybrid (RRF) | 0.740 | 0.557 |
| hybrid + LLM rerank | 0.757 | 0.593 |

**Vector wins and is the app default.** The expected winner was
hybrid+rerank; measurement said otherwise. Keyword search is weak on this
corpus (unpunctuated, misspelling-prone auto-captions vs. naturally worded
questions), and fusing its list in drags hybrid below pure vector — while
vector alone is also the cheapest and fastest variant. The vector-vs-rerank
gap (~3 points at n=300) is within noise; "at least as good, one LLM call
cheaper" decided it.

### Answer quality (LLM-as-judge)

50 ground-truth questions, full RAG run with two prompt styles, judged 0–2
for relevance (judge sees question, answer, and source chunk).
Reproduce: `uv run --env-file .env evals/rag_eval.py`.

| Prompt style | Mean relevance | fully relevant | partial | irrelevant |
|---|---|---|---|---|
| strict grounded | 1.80 | 82% | 16% | 2% |
| **explanatory** | **1.84** | **84%** | 16% | 0% |

The two styles are tied within noise; explanatory (may add brief
background, never beyond the excerpts on facts) is the default.

**Honest limitations**: LLM-generated questions share the source chunk's
topic vocabulary, so absolute numbers flatter all variants — the
*comparison* stays fair since every variant faces the same questions. The
judge is the same model family as the answerer (mild self-preference).
Instructing question generation to avoid the chunk's phrasing handicaps
keyword search specifically; real users typing exact jargon would narrow
the keyword–vector gap.

## Monitoring

Every conversation is logged to Postgres — question, rewrite, answer,
retrieved chunks, tokens, cost, latency, retrieval score — and every answer
has 👍/👎 feedback buttons. Two Streamlit pages read it live:

- **dashboard** — 4 stat tiles (questions, total cost, median latency,
  feedback split) and 6 charts: questions over time, feedback split,
  latency distribution, cost per day, episodes cited, retrieval score
  distribution, plus a recent-conversations table.
- **database** — the raw `conversations` and `feedback` tables, so the
  collected data is inspectable without psql.

## Project layout

```
fusionrag/
  ingest.py     fetch → de-ad → chunk → embed → data/ (idempotent)
  embedder.py   ONNX all-MiniLM wrapper (downloads model on first use)
  search.py     keyword / vector / hybrid / rerank + adjacent-chunk dedup
  rag.py        rewrite → retrieve → cited answer + usage metadata
  db.py         Postgres schema (auto-created) + logging helpers
app.py          Streamlit chat UI with feedback buttons
pages/          dashboard (monitoring) + database (raw tables)
evals/          ground truth generator, retrieval + RAG evals, results/
data/           committed transcripts, chunks.json, embeddings.npz
smoke.py        pre-commit pipeline health check
```

## For reviewers (evaluation criteria pointers)

- **Problem description** — this README, top two sections
- **Retrieval flow** — knowledge base + LLM: `fusionrag/search.py`, `fusionrag/rag.py`
- **Retrieval evaluation** — 4 variants compared, winner used: `evals/retrieval_eval.py`, table above
- **LLM evaluation** — 2 prompts × LLM judge, winner used: `evals/rag_eval.py`, table above
- **Interface** — Streamlit UI: `app.py`
- **Ingestion pipeline** — automated script: `fusionrag/ingest.py`
- **Monitoring** — feedback + dashboard with 6 charts: `pages/dashboard.py`
- **Containerization** — everything in docker-compose (app + Postgres with healthcheck): `docker-compose.yml`, `Dockerfile`
- **Reproducibility** — pinned deps (`uv.lock`), committed data, eval scripts re-runnable
- **Best practices** — hybrid search (RRF) evaluated, LLM re-ranking evaluated, query rewriting in the app path

Extras that don't fit a rubric row: sponsor-segment removal during ingest
(SponsorBlock API + transcript-marker fallback, verified by grepping for
the ad brands), timestamped deep-link citations into the videos,
overlap-aware retrieval, the raw-database page, and the embedding model
baked into the Docker image for offline startup.

## Attribution

Sponsor segment timestamps come from [SponsorBlock](https://sponsor.ajay.app)
(CC BY-NC-SA 4.0). This project is non-commercial. Transcripts are fetched
with [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/);
embeddings use [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2).
