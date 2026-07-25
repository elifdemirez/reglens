"""Data access for the ``documents`` table."""

import sqlite3
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create(
    conn: sqlite3.Connection,
    *,
    filename: str,
    stored_name: str,
    file_type: str,
    size_bytes: int,
    doc_kind: str = "general",
    short_label: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO documents
            (filename, stored_name, file_type, size_bytes, status, doc_kind, short_label, created_at)
        VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
        """,
        (filename, stored_name, file_type, size_bytes, doc_kind, short_label, _now()),
    )
    conn.commit()
    return int(cur.lastrowid)


def mark_indexing(conn: sqlite3.Connection, document_id: int) -> None:
    conn.execute("UPDATE documents SET status = 'indexing' WHERE id = ?", (document_id,))
    conn.commit()


def mark_ready(
    conn: sqlite3.Connection,
    document_id: int,
    *,
    chunk_count: int,
    page_count: int,
    doc_kind: str,
    short_label: str | None,
) -> None:
    conn.execute(
        """
        UPDATE documents
           SET status = 'ready', chunk_count = ?, page_count = ?, doc_kind = ?,
               short_label = ?, indexed_at = ?, error = NULL
         WHERE id = ?
        """,
        (chunk_count, page_count, doc_kind, short_label, _now(), document_id),
    )
    conn.commit()


def mark_failed(conn: sqlite3.Connection, document_id: int, error: str) -> None:
    conn.execute(
        "UPDATE documents SET status = 'failed', error = ? WHERE id = ?",
        (error[:1000], document_id),
    )
    conn.commit()


def reset_stale_indexing(conn: sqlite3.Connection) -> int:
    """Fail any document left mid-index by a crash or restart.

    Ingestion runs in a background task, so a process that dies partway through
    leaves the row in 'indexing' forever — the UI would poll it indefinitely and
    the user would have no way to retry. Called once at startup.
    """
    cur = conn.execute(
        """
        UPDATE documents
           SET status = 'failed',
               error = 'Indexing was interrupted before it completed. Delete and re-upload.'
         WHERE status IN ('pending', 'indexing')
        """
    )
    conn.commit()
    return cur.rowcount


def list_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM documents ORDER BY created_at DESC, id DESC"
    ).fetchall()


def get(conn: sqlite3.Connection, document_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()


def count_active(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])


def exists_with_filename(conn: sqlite3.Connection, filename: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM documents WHERE filename = ? LIMIT 1", (filename,)
    ).fetchone()
    return row is not None


def delete(conn: sqlite3.Connection, document_id: int) -> None:
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()
