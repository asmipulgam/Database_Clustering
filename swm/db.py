"""Postgres access shared by every pipeline stage. Connection settings come from .env."""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

_DOCUMENTS_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    corpus     TEXT NOT NULL,
    text       TEXT NOT NULL,
    label      TEXT,              -- ground truth, evaluation only
    split      TEXT,
    n_tokens   INTEGER NOT NULL,
    meta       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def connect() -> psycopg.Connection:
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "swm"),
        user=os.getenv("POSTGRES_USER", "swm"),
        password=os.getenv("POSTGRES_PASSWORD", "swm"),
        connect_timeout=5,
    )


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(_DOCUMENTS_DDL)
    conn.execute("CREATE INDEX IF NOT EXISTS documents_corpus_idx ON documents (corpus)")


def write_documents(conn: psycopg.Connection, corpus: str, docs: list[dict]) -> int:
    """Replace all rows for `corpus` with `docs` in one transaction (safe to re-run)."""
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE corpus = %s", (corpus,))
        with cur.copy(
            "COPY documents (id, corpus, text, label, split, n_tokens, meta) FROM STDIN"
        ) as copy:
            for d in docs:
                copy.write_row(
                    (d["id"], corpus, d["text"], d.get("label"), d.get("split"),
                     d["n_tokens"], json.dumps(d.get("meta", {})))
                )
    return len(docs)


def read_documents(conn: psycopg.Connection, corpus: str) -> list[dict]:
    """All documents for `corpus`, ordered by id so embedding rows line up across runs."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, corpus, text, label, split, n_tokens, meta FROM documents "
            "WHERE corpus = %s ORDER BY id",
            (corpus,),
        )
        return cur.fetchall()
