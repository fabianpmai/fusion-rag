"""Postgres logging: conversations + feedback. One module-level connection,
re-established when it goes stale (Postgres restart, idle timeout). Tables
are created on first connect if missing."""

import os

import pandas as pd
import psycopg

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://fusionrag:fusionrag@localhost:5432/fusionrag"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    question TEXT NOT NULL,
    rewritten_question TEXT,
    answer TEXT NOT NULL,
    model TEXT,
    prompt_tokens INT,
    completion_tokens INT,
    cost_usd DOUBLE PRECISION,
    latency_s DOUBLE PRECISION,
    retrieved_chunk_ids TEXT[],
    avg_retrieval_score DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS feedback (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES conversations(id),
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    thumbs_up BOOLEAN NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS feedback_conversation_uq
    ON feedback (conversation_id);
"""

_conn = None


def connect():
    """Return a live connection, reconnecting if the cached one is dead."""
    global _conn
    if _conn is not None and not _conn.closed:
        try:
            _conn.execute("SELECT 1")
            return _conn
        except psycopg.OperationalError:
            pass
    _conn = psycopg.connect(DB_URL, autocommit=True)
    with _conn.cursor() as cur:
        cur.execute(SCHEMA)
    return _conn


def query(sql):
    with connect().cursor() as cur:
        cur.execute(sql)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def log_conversation(result):
    with connect().cursor() as cur:
        cur.execute(
            """INSERT INTO conversations (question, rewritten_question, answer,
                   model, prompt_tokens, completion_tokens, cost_usd, latency_s,
                   retrieved_chunk_ids, avg_retrieval_score)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (
                result["question"],
                result["rewritten_question"],
                result["answer"],
                result["model"],
                result["prompt_tokens"],
                result["completion_tokens"],
                result["cost_usd"],
                result["latency_s"],
                result["chunk_ids"],
                result["avg_retrieval_score"],
            ),
        )
        return cur.fetchone()[0]


def log_feedback(conversation_id, thumbs_up):
    """One row per conversation: changing a vote updates it."""
    with connect().cursor() as cur:
        cur.execute(
            """INSERT INTO feedback (conversation_id, thumbs_up)
               VALUES (%s, %s)
               ON CONFLICT (conversation_id)
               DO UPDATE SET thumbs_up = EXCLUDED.thumbs_up, ts = now()""",
            (conversation_id, thumbs_up),
        )


def delete_feedback(conversation_id):
    with connect().cursor() as cur:
        cur.execute(
            "DELETE FROM feedback WHERE conversation_id = %s", (conversation_id,)
        )
