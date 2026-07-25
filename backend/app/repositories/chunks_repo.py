"""Data access for the ``chunks`` table."""

import json
import sqlite3
from typing import Any, Iterable


def bulk_insert(conn: sqlite3.Connection, document_id: int, chunks: Iterable[dict[str, Any]]) -> int:
    rows = [
        (
            document_id,
            c["chunk_index"],
            c["content"],
            json.dumps(c["embedding"]) if c.get("embedding") is not None else None,
            c.get("page"),
            c.get("article"),
            c.get("article_num"),
            c.get("paragraph"),
            c.get("section_path"),
            c.get("heading"),
            c.get("chunk_kind", "body"),
        )
        for c in chunks
    ]
    conn.executemany(
        """
        INSERT INTO chunks
            (document_id, chunk_index, content, embedding, page, article, article_num,
             paragraph, section_path, heading, chunk_kind)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def fetch_for_retrieval(
    conn: sqlite3.Connection, document_ids: list[int] | None = None
) -> list[dict[str, Any]]:
    """Load every indexed chunk (with its embedding decoded) for scoring.

    The corpus here is capped at 20 documents, so loading it into memory is
    intentional and simpler than a vector index. See README "Scaling notes".
    """
    sql = """
        SELECT c.*, d.filename, d.short_label, d.doc_kind
          FROM chunks c
          JOIN documents d ON d.id = c.document_id
         WHERE d.status = 'ready'
    """
    params: list[Any] = []
    if document_ids:
        placeholders = ",".join("?" for _ in document_ids)
        sql += f" AND c.document_id IN ({placeholders})"
        params.extend(document_ids)

    result = []
    for row in conn.execute(sql, params):
        item = dict(row)
        item["embedding"] = json.loads(row["embedding"]) if row["embedding"] else None
        result.append(item)
    return result


def fetch_siblings(
    conn: sqlite3.Connection, document_id: int, article_num: int | None
) -> list[dict[str, Any]]:
    """Return the other chunks belonging to the same article.

    Used for context expansion: a hit on "Article 10(4)" is far more useful when
    the surrounding paragraphs of Article 10 come with it.
    """
    if article_num is None:
        return []
    rows = conn.execute(
        """
        SELECT c.*, d.filename, d.short_label, d.doc_kind
          FROM chunks c
          JOIN documents d ON d.id = c.document_id
         WHERE c.document_id = ? AND c.article_num = ?
         ORDER BY c.chunk_index
        """,
        (document_id, article_num),
    ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["embedding"] = None  # not needed for expansion
        out.append(item)
    return out


def count_for_document(conn: sqlite3.Connection, document_id: int) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,)
        ).fetchone()[0]
    )
