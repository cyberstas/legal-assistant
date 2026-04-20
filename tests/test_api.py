"""Tests for FastAPI endpoints (uses TestClient with in-memory DB)."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.models.database as db_module
from src.models.database import reset_engine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with an in-memory SQLite DB and temp upload dir."""
    monkeypatch.setattr("src.config.settings.upload_dir", str(tmp_path))
    monkeypatch.setattr("src.config.settings.openai_api_key", "")  # disable LLM calls

    # Point the global engine at a fresh in-memory SQLite for each test
    reset_engine("sqlite:///:memory:")

    from src.main import app

    with TestClient(app) as c:
        yield c

    # Reset engine so subsequent tests get a fresh one
    reset_engine("sqlite:///:memory:")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------


def test_upload_text_document(client, tmp_path):
    content = b"This is a test legal document about custody arrangements."
    resp = client.post(
        "/documents/upload",
        files={"file": ("test_custody.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["original_filename"] == "test_custody.txt"
    assert data["doc_type"] == "text"
    assert data["status"] == "completed"


def test_upload_document_with_type_override(client):
    content = b"Transcript of custody hearing on January 15, 2024."
    resp = client.post(
        "/documents/upload",
        files={"file": ("hearing.txt", io.BytesIO(content), "text/plain")},
        data={"doc_type_override": "transcript"},
    )
    assert resp.status_code == 201
    assert resp.json()["doc_type"] == "transcript"


def test_list_documents_empty(client):
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_documents_after_upload(client):
    content = b"Some legal text."
    client.post(
        "/documents/upload",
        files={"file": ("doc1.txt", io.BytesIO(content), "text/plain")},
    )
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_get_document_not_found(client):
    resp = client.get("/documents/9999")
    assert resp.status_code == 404


def test_get_document_detail(client):
    content = b"Police report: incident on March 3, 2024."
    upload = client.post(
        "/documents/upload",
        files={"file": ("police_report.txt", io.BytesIO(content), "text/plain")},
    )
    doc_id = upload.json()["id"]

    resp = client.get(f"/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == doc_id
    assert "Police report" in (resp.json()["content_text"] or "")


def test_delete_document(client):
    content = b"Email communication."
    upload = client.post(
        "/documents/upload",
        files={"file": ("email.txt", io.BytesIO(content), "text/plain")},
    )
    doc_id = upload.json()["id"]

    resp = client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 204

    resp = client.get(f"/documents/{doc_id}")
    assert resp.status_code == 404


def test_get_document_timeline(client):
    content = b"Some text."
    upload = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(content), "text/plain")},
    )
    doc_id = upload.json()["id"]

    resp = client.get(f"/documents/{doc_id}/timeline")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_document_facts(client):
    content = b"Some text."
    upload = client.post(
        "/documents/upload",
        files={"file": ("notes2.txt", io.BytesIO(content), "text/plain")},
    )
    doc_id = upload.json()["id"]

    resp = client.get(f"/documents/{doc_id}/facts")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Timeline endpoint
# ---------------------------------------------------------------------------


def test_full_timeline_empty(client):
    resp = client.get("/timeline")
    assert resp.status_code == 200
    assert resp.json() == []


def test_post_timeline_filter(client):
    resp = client.post("/timeline", json={})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Analysis endpoints (no OpenAI key → 503)
# ---------------------------------------------------------------------------


def test_query_no_api_key(client):
    resp = client.post("/analysis/query", json={"query": "What happened?", "top_k": 3})
    assert resp.status_code == 503


def test_evidence_no_api_key(client):
    resp = client.post("/analysis/evidence", json={"topic": "custody violations", "top_k": 5})
    assert resp.status_code == 503


def test_cross_examination_no_api_key(client):
    resp = client.post(
        "/analysis/cross-examination",
        json={"witness_name": "Jane Doe", "topics": ["custody", "communication"]},
    )
    assert resp.status_code == 503


def test_get_all_facts_empty(client):
    resp = client.get("/analysis/facts")
    assert resp.status_code == 200
    assert resp.json() == []
