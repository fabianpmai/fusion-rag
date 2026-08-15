"""Fetch transcripts, drop sponsor segments, chunk, embed, write data/.

Idempotent: raw transcripts and SponsorBlock responses are cached in
data/raw/ and reused on re-runs, so the script works offline once fetched.
"""

import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import requests
from youtube_transcript_api import YouTubeTranscriptApi

from fusionrag.embedder import Embedder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"

CHUNK_SIZE = 2000
CHUNK_STEP = 1000
SPONSORBLOCK_API = "https://sponsor.ajay.app/api/skipSegments"
# Lex's ad-block-ending phrase: "And now, dear friends, here's <guest>."
MARKER = "dear friends"

EPISODES = [
    {
        "episode": 112,
        "video_id": "pDSEjaDCtOU",
        "guest": "Ian Hutchinson",
        "title": "Ian Hutchinson: Nuclear Fusion, Plasma Physics, and Religion | Lex Fridman Podcast #112",
    },
    {
        "episode": 353,
        "video_id": "aJoRMFWn2Jk",
        "guest": "Dennis Whyte",
        "title": "Dennis Whyte: Nuclear Fusion and the Future of Energy | Lex Fridman Podcast #353",
    },
    {
        "episode": 485,
        "video_id": "m_CFCyc2Shs",
        "guest": "David Kirtley",
        "title": "David Kirtley: Nuclear Fusion, Plasma Physics, and the Future of Energy | Lex Fridman Podcast #485",
    },
]


def fetch_transcript(video_id):
    path = RAW / f"{video_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    snippets = YouTubeTranscriptApi().fetch(video_id).to_raw_data()
    path.write_text(json.dumps(snippets, ensure_ascii=False, indent=1))
    return snippets


def fetch_sponsor_segments(video_id):
    """Crowdsourced [start, end] sponsor ranges from SponsorBlock (may be empty)."""
    path = RAW / f"{video_id}.sponsorblock.json"
    if path.exists():
        return json.loads(path.read_text())
    resp = requests.get(
        SPONSORBLOCK_API,
        params={"videoID": video_id, "category": "sponsor"},
        timeout=30,
    )
    if resp.status_code == 404:
        segments = []
    else:
        resp.raise_for_status()
        segments = [item["segment"] for item in resp.json()]
    path.write_text(json.dumps(segments))
    return segments


def drop_sponsor_snippets(snippets, segments):
    return [
        s
        for s in snippets
        if not any(
            s["start"] < end and s["start"] + s["duration"] > start
            for start, end in segments
        )
    ]


def merge_snippets(snippets):
    """Concatenate snippet texts, keeping (char_offset, start_sec) pairs.

    Manual captions (#485) roll over: a snippet often repeats the tail of
    the previous one, so exact boundary overlaps of 15+ chars are trimmed.
    """
    parts, offsets = [], []
    pos = 0
    tail = ""
    for s in snippets:
        text = " ".join(s["text"].split())
        low_tail, low_text = tail.lower(), text.lower()
        for k in range(min(len(tail), len(text), 80), 14, -1):
            if low_tail[-k:] == low_text[:k]:
                text = text[k:].lstrip()
                break
        if not text:
            continue
        offsets.append((pos, s["start"]))
        parts.append(text)
        pos += len(text) + 1
        tail = (tail + " " + text)[-80:]
    return " ".join(parts), offsets


def marker_cut(text, guest):
    """Fallback for videos without SponsorBlock data: everything before
    Lex's ad-block-ending phrase (which names the guest) is intro/ads."""
    low = text.lower()
    marker_at = low.find(MARKER)
    if marker_at == -1:
        raise RuntimeError(f"ad-block marker {MARKER!r} not found for {guest}")
    # match the first name only: auto captions misspell last names
    # (e.g. "Dennis White" for Whyte), then skip the last-name word
    firstname = guest.split()[0].lower()
    name_at = low.find(firstname, marker_at, marker_at + 200)
    if name_at == -1:
        raise RuntimeError(f"guest name {firstname!r} not found after marker")
    cut = name_at + len(firstname)
    while cut < len(text) and text[cut] == " ":
        cut += 1
    while cut < len(text) and text[cut] not in " .,":
        cut += 1
    while cut < len(text) and text[cut] in " .,:;-":
        cut += 1
    return cut


def chunk_episode(ep, text, offsets, start_char):
    chunk_starts = [off for off, _ in offsets]
    starts_sec = [sec for _, sec in offsets]
    url = f"https://www.youtube.com/watch?v={ep['video_id']}"
    chunks = []
    i = start_char
    while True:
        start_sec = int(starts_sec[bisect_right(chunk_starts, i) - 1])
        chunks.append(
            {
                "id": f"{ep['episode']}-{len(chunks):04d}",
                "episode": ep["episode"],
                "guest": ep["guest"],
                "title": ep["title"],
                "start_sec": start_sec,
                "youtube_url": f"{url}&t={start_sec}s",
                "text": text[i : i + CHUNK_SIZE],
            }
        )
        if i + CHUNK_SIZE >= len(text):
            return chunks
        i += CHUNK_STEP


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    for ep in EPISODES:
        snippets = fetch_transcript(ep["video_id"])
        segments = fetch_sponsor_segments(ep["video_id"])
        kept = drop_sponsor_snippets(snippets, segments)
        text, offsets = merge_snippets(kept)
        start_char = 0 if segments else marker_cut(text, ep["guest"])
        chunks = chunk_episode(ep, text, offsets, start_char)
        all_chunks.extend(chunks)
        print(
            f"#{ep['episode']}: {len(snippets)} snippets, "
            f"{len(segments)} sponsor segments"
            f"{' (marker cut)' if not segments else ''}, {len(chunks)} chunks"
        )

    (DATA / "chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=1)
    )

    embedder = Embedder()
    vectors = [
        embedder.encode_batch([c["text"] for c in all_chunks[i : i + 64]])
        for i in range(0, len(all_chunks), 64)
    ]
    embeddings = np.vstack(vectors).astype(np.float32)
    np.savez_compressed(
        DATA / "embeddings.npz",
        embeddings=embeddings,
        chunk_ids=np.array([c["id"] for c in all_chunks]),
    )
    print(f"wrote {len(all_chunks)} chunks, embeddings {embeddings.shape}")


if __name__ == "__main__":
    main()
