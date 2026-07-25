"""Ingestion pipeline: file on disk -> parsed -> chunked -> embedded -> SQLite."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from app.repositories import chunks_repo, documents_repo
from app.services.document_processing.chunker import build_chunks
from app.services.document_processing.legal_parser import parse_pages
from app.services.document_processing.readers import read_document

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 16


def ingest_document(conn: sqlite3.Connection, document_id: int, path: Path, embedder) -> int:
    """Parse, chunk, embed and store one document. Returns the chunk count.

    ``embedder`` is any object exposing ``embed_batch(list[str]) -> list[list[float]]``;
    injecting it keeps this function testable without loading a real model.
    """
    documents_repo.mark_indexing(conn, document_id)
    try:
        pages = read_document(path)
        if not pages:
            raise ValueError("No extractable text found in the document.")

        parsed = parse_pages(pages)
        chunks = build_chunks(parsed.blocks)
        if not chunks:
            raise ValueError("Document produced no indexable chunks.")

        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            vectors = embedder.embed_batch([c["content"] for c in batch])
            for chunk, vector in zip(batch, vectors):
                chunk["embedding"] = vector

        chunks_repo.bulk_insert(conn, document_id, chunks)
        documents_repo.mark_ready(
            conn,
            document_id,
            chunk_count=len(chunks),
            page_count=len(pages),
            doc_kind=parsed.detected_kind,
            short_label=parsed.short_label or Path(path).stem[:24],
        )
        logger.info(
            "Indexed document %s: %d chunks, %d pages, kind=%s",
            document_id,
            len(chunks),
            len(pages),
            parsed.detected_kind,
        )
        return len(chunks)
    except Exception as exc:  # noqa: BLE001 - failure is recorded on the document row
        logger.exception("Ingestion failed for document %s", document_id)
        documents_repo.mark_failed(conn, document_id, str(exc))
        raise
