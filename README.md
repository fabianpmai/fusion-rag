# fusion-rag

RAG app over three Lex Fridman nuclear-fusion episodes:
#112 (Ian Hutchinson), #353 (Dennis Whyte), #485 (David Kirtley).

Work in progress — capstone project for LLM Zoomcamp 2026.

## Ingest

Fetch transcripts, strip sponsor segments, chunk, embed:

```bash
uv run -m fusionrag.ingest
```

Outputs are committed under `data/`, so you don't need to run this to use
the app. The script is idempotent: raw transcripts and SponsorBlock
responses are cached in `data/raw/` and reused on re-runs.

## Attribution

Sponsor segment timestamps come from [SponsorBlock](https://sponsor.ajay.app)
(CC BY-NC-SA 4.0). This project is non-commercial.
