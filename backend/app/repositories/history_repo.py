"""Data access for ``query_history``."""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any


def record(
    conn: sqlite3.Connection,
    *,
    question: str,
    answer: str,
    mode: str,
    answer_path: str,
    confidence: float,
    sources: list[dict[str, Any]],
    elapsed_ms: int,
) -> int:
    # Only the citation metadata is kept — storing every excerpt body would
    # duplicate the chunks table for no benefit.
    slim = [
        {
            "citation": s.get("citation"),
            "score": s.get("score"),
            "chunk_id": s.get("chunk_id"),
            "short_label": s.get("short_label"),
        }
        for s in sources
    ]
    cur = conn.execute(
        """
        INSERT INTO query_history
            (question, answer, mode, answer_path, confidence, sources_json, elapsed_ms, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            question,
            answer,
            mode,
            answer_path,
            confidence,
            json.dumps(slim, ensure_ascii=False),
            elapsed_ms,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_recent(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM query_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["sources"] = json.loads(row["sources_json"])
        item.pop("sources_json", None)
        out.append(item)
    return out


def clear(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM query_history")
    conn.commit()
