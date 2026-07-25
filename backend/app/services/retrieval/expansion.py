"""Context expansion.

A retrieval hit is often a fragment of a larger legal unit: Article 10(4) makes
little sense without the sentence in 10(1) that says whom the article binds.
This module pulls in the sibling paragraphs of the top hits, staying inside a
character budget so the prompt does not blow past what a small local model can
actually attend to.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from app.repositories import chunks_repo
from app.services.retrieval.hybrid import RetrievedChunk

MAX_CONTEXT_CHARS = 7000
EXPAND_TOP_N = 2


def _as_context_entry(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "citation": chunk.citation,
        "content": chunk.content,
        "chunk_id": chunk.chunk_id,
        "expanded": False,
    }


def build_context(
    conn: sqlite3.Connection,
    results: list[RetrievedChunk],
    *,
    expand: bool = True,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> list[dict[str, Any]]:
    """Return ordered context entries, primary hits first then expansions."""
    entries: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    used = 0

    for chunk in results:
        if chunk.chunk_id in seen_ids:
            continue
        if used + len(chunk.content) > max_chars and entries:
            break
        entries.append(_as_context_entry(chunk))
        seen_ids.add(chunk.chunk_id)
        used += len(chunk.content)

    if not expand:
        return entries

    for chunk in results[:EXPAND_TOP_N]:
        if chunk.article_num is None:
            continue
        for sibling in chunks_repo.fetch_siblings(conn, chunk.document_id, chunk.article_num):
            if sibling["id"] in seen_ids:
                continue
            content = sibling["content"]
            if used + len(content) > max_chars:
                break
            label = sibling.get("short_label") or sibling.get("filename")
            article = sibling.get("article") or ""
            paragraph = sibling.get("paragraph")
            citation = f"{label}, {article}({paragraph})" if paragraph else f"{label}, {article}"
            entries.append(
                {
                    "citation": citation.strip(", "),
                    "content": content,
                    "chunk_id": sibling["id"],
                    "expanded": True,
                }
            )
            seen_ids.add(sibling["id"])
            used += len(content)

    return entries


def format_context(entries: list[dict[str, Any]]) -> str:
    """Render context entries into the block handed to the chat model."""
    return "\n\n---\n\n".join(
        f"[{entry['citation']}]\n{entry['content']}" for entry in entries
    )
