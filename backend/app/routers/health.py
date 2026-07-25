"""Health endpoint: is the backend up and can it reach Foundry Local?"""

import sqlite3

from fastapi import APIRouter, Depends

from app.config import settings
from app.deps import get_db
from app.repositories import documents_repo
from app.services.foundry import client as foundry_client

router = APIRouter(tags=["health"])


@router.get("/health")
def health(conn: sqlite3.Connection = Depends(get_db)) -> dict:
    documents = documents_repo.list_all(conn)
    ready = [d for d in documents if d["status"] == "ready"]
    return {
        "status": "ok",
        "foundry": foundry_client.probe(),
        "documents": {
            "total": len(documents),
            "ready": len(ready),
            "limit": settings.max_document_count,
        },
        "settings": {
            "chat_model": settings.chat_model_alias,
            "embedding_model": settings.embedding_model_alias,
            "top_k": settings.top_k,
            "max_file_size_mb": settings.max_file_size_mb,
        },
    }
