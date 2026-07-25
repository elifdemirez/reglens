"""Question answering endpoints (buffered, streaming, history and export)."""

from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.deps import get_db
from app.repositories import history_repo
from app.repositories.db import get_connection
from app.services.foundry import client as foundry_client
from app.services.rag.answer import answer_question, answer_question_stream

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[int] | None = None
    stream: bool = False


@router.post("/chat")
def chat(payload: ChatRequest, conn: sqlite3.Connection = Depends(get_db)):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    models = foundry_client.get_models()

    if not payload.stream:
        result = answer_question(
            conn, question, models, document_ids=payload.document_ids
        )
        history_repo.record(
            conn,
            question=question,
            answer=result.answer,
            mode=result.mode,
            answer_path=result.answer_path,
            confidence=result.confidence,
            sources=result.sources,
            elapsed_ms=result.elapsed_ms,
        )
        return result.to_dict()

    def event_stream():
        # A dedicated connection: the dependency-scoped one closes as soon as
        # this handler returns, which is before the generator is consumed.
        stream_conn = get_connection()
        final: dict | None = None
        try:
            for event in answer_question_stream(
                stream_conn, question, models, document_ids=payload.document_ids
            ):
                if event.get("type") == "done":
                    final = event
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if final is not None:
                history_repo.record(
                    stream_conn,
                    question=question,
                    answer=final.get("answer", ""),
                    mode=final.get("mode", "ask"),
                    answer_path=final.get("answer_path", "synthesis"),
                    confidence=final.get("confidence", 0.0),
                    sources=final.get("sources", []),
                    elapsed_ms=final.get("elapsed_ms", 0),
                )
        except Exception as exc:  # noqa: BLE001 - surface errors inside the stream
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            stream_conn.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history")
def history(limit: int = 50, conn: sqlite3.Connection = Depends(get_db)) -> list[dict]:
    return history_repo.list_recent(conn, limit=limit)


@router.delete("/history", status_code=204)
def clear_history(conn: sqlite3.Connection = Depends(get_db)) -> None:
    history_repo.clear(conn)


@router.get("/history/export", response_class=PlainTextResponse)
def export_history(limit: int = 50, conn: sqlite3.Connection = Depends(get_db)) -> str:
    """Export the session as Markdown, with citations preserved."""
    rows = history_repo.list_recent(conn, limit=limit)
    lines = ["# RegLens session export", ""]
    for row in reversed(rows):
        lines.append(f"## {row['question']}")
        lines.append("")
        lines.append(row["answer"])
        lines.append("")
        lines.append(
            f"*Confidence: {row['confidence']:.0%} · path: {row['answer_path']} · "
            f"{row['elapsed_ms']} ms · {row['created_at']}*"
        )
        if row["sources"]:
            lines.append("")
            lines.append("**Sources**")
            for source in row["sources"]:
                lines.append(f"- {source.get('citation')} (score {source.get('score')})")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)
