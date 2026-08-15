"""Postgres logging: conversations + feedback. Tables are created on
connect if missing."""

import os

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
"""


def connect():
    conn = psycopg.connect(DB_URL, autocommit=True)
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    return conn


def log_conversation(conn, result):
    with conn.cursor() as cur:
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


def log_feedback(conn, conversation_id, thumbs_up):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO feedback (conversation_id, thumbs_up) VALUES (%s, %s)",
            (conversation_id, thumbs_up),
        )
