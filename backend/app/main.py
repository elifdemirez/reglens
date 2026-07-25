"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.repositories import documents_repo
from app.repositories.db import get_connection, init_db
from app.routers import chat, documents, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    logger = logging.getLogger(__name__)

    # Ingestion happens in a background task; a restart mid-index would otherwise
    # leave documents stuck on 'indexing' with the UI polling them forever.
    conn = get_connection()
    try:
        stale = documents_repo.reset_stale_indexing(conn)
        if stale:
            logger.warning("Marked %d interrupted document(s) as failed.", stale)
    finally:
        conn.close()

    logger.info("RegLens API ready.")
    yield


app = FastAPI(
    title="RegLens API",
    description="Local, document-grounded question answering over EU regulations.",
    version="1.0.0",
    lifespan=lifespan,
)

# The Vite dev server runs on 5173; everything stays on localhost by design.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
