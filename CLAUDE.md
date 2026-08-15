# fusion-rag

RAG app over three Lex Fridman nuclear-fusion episodes — capstone project
for LLM Zoomcamp 2026. Public repo; everything here gets peer-reviewed.

## Context lives in the Obsidian vault

Read these before changing anything:

- `../vault/capstone/03-spec.md` — the design, source of truth for this build
- `../vault/capstone/01-brief.md` — rubric checklist we're scoring against
- `../llm-zoomcamp/project.md` — official evaluation criteria (the grader's view)
- `../vault/schedule-and-deadlines.md` — submission deadlines (authoritative,
  don't trust dates hard-coded anywhere else)

Log every non-trivial decision to `../vault/capstone/04-build-log.md` as you go.

## Rules

- uv only: `uv add` for deps, `uv run` for anything executable — never bare
  pip/python, locally or in Docker
- README is written for reviewers who never saw the course
- Reproducibility is sacred: pinned deps, committed data, docker-compose must
  work on a fresh clone
- This ships in days, not weeks — cut scope before cutting quality
