"""API-level tests.

Foundry Local is stubbed out so these run without loading a model; what is
being tested here is routing, validation, limits and persistence, not
generation quality.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.repositories.db import SCHEMA
from tests.conftest import IVDR_SAMPLE, MDR_SAMPLE, FakeChatModel


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "db_path", tmp_path / "api.db")
    monkeypatch.setattr(config.settings, "upload_dir", tmp_path / "uploads")
    config.settings.upload_dir.mkdir(parents=True, exist_ok=True)

    fake = FakeChatModel()
    from app.services.foundry import client as foundry_client

    monkeypatch.setattr(foundry_client, "get_models", lambda: fake)
    monkeypatch.setattr(
        foundry_client, "probe", lambda: {"available": True, "models_loaded": True}
    )

    from app.repositories.db import get_connection

    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()

    from app.main import app

    with TestClient(app) as test_client:
        test_client.fake_models = fake  # type: ignore[attr-defined]
        yield test_client


def _upload(client, name: str, text: str):
    return client.post(
        "/api/documents",
        files={"file": (name, text.encode("utf-8"), "text/plain")},
    )


# --- health -------------------------------------------------------------------

def test_health_reports_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["foundry"]["available"] is True


# --- documents ----------------------------------------------------------------

def test_document_list_starts_empty(client):
    assert client.get("/api/documents").json() == []


def test_upload_indexes_document(client):
    response = _upload(client, "mdr.txt", MDR_SAMPLE)
    assert response.status_code == 201
    assert response.json()["filename"] == "mdr.txt"

    # Ingestion runs as a background task, which TestClient completes before
    # the context manager exits the request.
    documents = client.get("/api/documents").json()
    assert len(documents) == 1
    assert documents[0]["status"] == "ready"
    assert documents[0]["chunk_count"] > 0
    assert documents[0]["short_label"] == "MDR"


def test_rejects_unsupported_extension(client):
    response = client.post(
        "/api/documents", files={"file": ("notes.xlsx", b"binary", "application/octet-stream")}
    )
    assert response.status_code == 415


def test_rejects_empty_file(client):
    response = client.post("/api/documents", files={"file": ("empty.txt", b"", "text/plain")})
    assert response.status_code == 400


def test_rejects_oversized_file(client, monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "max_file_size_mb", 0.001)
    response = _upload(client, "big.txt", MDR_SAMPLE * 20)
    assert response.status_code == 413


def test_rejects_duplicate_filename(client):
    _upload(client, "mdr.txt", MDR_SAMPLE)
    response = _upload(client, "mdr.txt", MDR_SAMPLE)
    assert response.status_code == 409


def test_enforces_document_limit(client, monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "max_document_count", 1)
    _upload(client, "mdr.txt", MDR_SAMPLE)
    response = _upload(client, "ivdr.txt", IVDR_SAMPLE)
    assert response.status_code == 409


def test_delete_removes_document(client):
    document_id = _upload(client, "mdr.txt", MDR_SAMPLE).json()["id"]
    assert client.delete(f"/api/documents/{document_id}").status_code == 204
    assert client.get("/api/documents").json() == []


def test_delete_missing_document_is_404(client):
    assert client.delete("/api/documents/999").status_code == 404


# --- chat ---------------------------------------------------------------------

def test_chat_answers_from_uploaded_document(client):
    _upload(client, "mdr.txt", MDR_SAMPLE)
    response = client.post("/api/chat", json={"question": "What are the obligations of importers?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["sources"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["answer_path"] in {"direct", "synthesis", "refused"}


def test_chat_rejects_blank_question(client):
    assert client.post("/api/chat", json={"question": "   "}).status_code == 400


def test_chat_rejects_missing_question(client):
    assert client.post("/api/chat", json={}).status_code == 422


def test_chat_streaming_emits_sse_events(client):
    _upload(client, "mdr.txt", MDR_SAMPLE)
    with client.stream(
        "POST",
        "/api/chat",
        json={"question": "What are the obligations of importers?", "stream": True},
    ) as response:
        assert response.status_code == 200
        events = [
            json.loads(line[len("data: ") :])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    types = [e["type"] for e in events]
    assert "sources" in types
    assert types[-1] == "done"


def test_comparison_over_both_regulations(client):
    _upload(client, "mdr.txt", MDR_SAMPLE)
    _upload(client, "ivdr.txt", IVDR_SAMPLE)
    response = client.post(
        "/api/chat",
        json={"question": "What is the difference between MDR and IVDR manufacturer obligations?"},
    )
    body = response.json()
    assert body["mode"] == "compare"
    assert {s["short_label"] for s in body["sources"]} >= {"MDR", "IVDR"}


# --- history ------------------------------------------------------------------

def test_history_records_questions(client):
    _upload(client, "mdr.txt", MDR_SAMPLE)
    client.post("/api/chat", json={"question": "What are the obligations of importers?"})
    history = client.get("/api/history").json()
    assert len(history) == 1
    assert history[0]["question"] == "What are the obligations of importers?"


def test_history_export_is_markdown(client):
    _upload(client, "mdr.txt", MDR_SAMPLE)
    client.post("/api/chat", json={"question": "What are the obligations of importers?"})
    text = client.get("/api/history/export").text
    assert text.startswith("# RegLens session export")
    assert "**Sources**" in text


def test_history_can_be_cleared(client):
    _upload(client, "mdr.txt", MDR_SAMPLE)
    client.post("/api/chat", json={"question": "What are the obligations of importers?"})
    assert client.delete("/api/history").status_code == 204
    assert client.get("/api/history").json() == []
