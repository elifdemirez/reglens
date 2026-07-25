"""Document upload, listing and deletion."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File

from app.config import settings
from app.deps import get_db
from app.repositories import documents_repo
from app.repositories.db import get_connection
from app.services.document_processing.ingest import ingest_document
from app.services.foundry import client as foundry_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


def _serialize(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "filename": row["filename"],
        "file_type": row["file_type"],
        "size_bytes": row["size_bytes"],
        "status": row["status"],
        "doc_kind": row["doc_kind"],
        "short_label": row["short_label"],
        "chunk_count": row["chunk_count"],
        "page_count": row["page_count"],
        "error": row["error"],
        "created_at": row["created_at"],
        "indexed_at": row["indexed_at"],
    }


def _run_ingestion(document_id: int, path: Path) -> None:
    """Background ingestion. Opens its own connection: the request-scoped one
    is already closed by the time this runs."""
    conn = get_connection()
    try:
        models = foundry_client.get_models()
        ingest_document(conn, document_id, path, models)
    except Exception:  # noqa: BLE001 - the failure is already recorded on the row
        logger.exception("Background ingestion failed for document %s", document_id)
    finally:
        conn.close()


@router.get("")
def list_documents(conn: sqlite3.Connection = Depends(get_db)) -> list[dict]:
    return [_serialize(row) for row in documents_repo.list_all(conn)]


@router.post("", status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict:
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: "
            f"{', '.join(settings.allowed_extensions)}",
        )

    if documents_repo.count_active(conn) >= settings.max_document_count:
        raise HTTPException(
            status_code=409,
            detail=f"Document limit reached ({settings.max_document_count}). "
            "Delete a document before uploading another.",
        )

    if documents_repo.exists_with_filename(conn, filename):
        raise HTTPException(
            status_code=409, detail=f"A document named '{filename}' is already indexed."
        )

    payload = await file.read()
    size_mb = len(payload) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File is {size_mb:.1f} MB; the limit is {settings.max_file_size_mb} MB.",
        )
    if not payload:
        raise HTTPException(status_code=400, detail="File is empty.")

    stored_name = f"{uuid.uuid4().hex}{suffix}"
    destination = settings.upload_dir / stored_name
    destination.write_bytes(payload)

    document_id = documents_repo.create(
        conn,
        filename=filename,
        stored_name=stored_name,
        file_type=suffix.lstrip("."),
        size_bytes=len(payload),
    )

    # Indexing a 175-page regulation takes minutes on CPU, so it must not block
    # the upload response; the UI polls the document list for status.
    background_tasks.add_task(_run_ingestion, document_id, destination)

    row = documents_repo.get(conn, document_id)
    return _serialize(row)


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, conn: sqlite3.Connection = Depends(get_db)) -> None:
    row = documents_repo.get(conn, document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    stored = settings.upload_dir / row["stored_name"]
    stored.unlink(missing_ok=True)
    documents_repo.delete(conn, document_id)
