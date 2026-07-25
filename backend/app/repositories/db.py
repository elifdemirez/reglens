"""SQLite connection handling and schema definition."""

import sqlite3
from pathlib import Path

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT NOT NULL,
    stored_name   TEXT NOT NULL UNIQUE,
    file_type     TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    status        TEXT NOT NULL,              -- pending | indexing | ready | failed
    doc_kind      TEXT NOT NULL DEFAULT 'general',  -- mdr | ivdr | regulation | general
    short_label   TEXT,                       -- e.g. "MDR", "IVDR"
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    page_count    INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    created_at    TEXT NOT NULL,
    indexed_at    TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,
    embedding     TEXT,                       -- JSON array of floats
    page          INTEGER,
    article       TEXT,                       -- e.g. "Article 10"
    article_num   INTEGER,
    paragraph     TEXT,                       -- e.g. "2"
    section_path  TEXT,                       -- e.g. "Chapter II > Section 1 > Article 10"
    heading       TEXT,
    chunk_kind    TEXT NOT NULL DEFAULT 'body' -- definition | obligation | list | body | annex
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_article ON chunks(document_id, article_num);

CREATE TABLE IF NOT EXISTS query_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    mode          TEXT NOT NULL,              -- ask | compare
    answer_path   TEXT NOT NULL,              -- direct | synthesis
    confidence    REAL NOT NULL,
    sources_json  TEXT NOT NULL,
    elapsed_ms    INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys and row access by name."""
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Create tables if they do not exist yet."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
