"""Pre-commit smoke check: data is clean, every search variant returns
chunks, and rag() answers with a timestamped citation. Needs OPENAI_API_KEY.

    uv run smoke.py
"""

import json

from fusionrag.rag import RAG
from fusionrag.search import DATA, Search


def main():
    chunks = json.loads((DATA / "chunks.json").read_text())
    assert len(chunks) > 400, f"only {len(chunks)} chunks"
    assert {c["episode"] for c in chunks} == {112, 353, 485}
    text = " ".join(c["text"].lower() for c in chunks)
    for brand in ["sunbasket", "powerdot"]:
        assert brand not in text, f"sponsor leak: {brand}"

    search = Search()
    for name in ["keyword", "vector", "hybrid"]:
        result = getattr(search, name)("how does a tokamak confine plasma")
        assert len(result) == 5, f"{name} returned {len(result)}"
    print("data + search variants OK")

    out = RAG(search).ask("why is helium-3 a good fusion fuel?")
    assert "youtube.com/watch" in out["answer"], "no citation link in answer"
    assert out["cost_usd"] > 0 and out["chunk_ids"]
    print(f"rag OK: {out['prompt_tokens']}+{out['completion_tokens']} tok, "
          f"${out['cost_usd']:.4f}, {out['latency_s']:.1f}s")
    print(out["answer"][:300])


if __name__ == "__main__":
    main()
