"""Tests for database models and document store (uses in-memory SQLite)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.models.schemas import Base, Document, DocumentType, ProcessingStatus, TimelineEvent, ExtractedFact
from src.storage.document_store import DocumentStore


@pytest.fixture()
def db_session():
    """In-memory SQLite session for each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestDocumentStore:
    def test_create_document(self, db_session):
        store = DocumentStore(db_session)
        doc = store.create_document("saved.pdf", "original.pdf", DocumentType.PDF)

        assert doc.id is not None
        assert doc.filename == "saved.pdf"
        assert doc.original_filename == "original.pdf"
        assert doc.doc_type == DocumentType.PDF
        assert doc.status == ProcessingStatus.PENDING

    def test_get_document(self, db_session):
        store = DocumentStore(db_session)
        doc = store.create_document("a.pdf", "a.pdf", DocumentType.PDF)
        fetched = store.get_document(doc.id)
        assert fetched.id == doc.id

    def test_get_document_not_found(self, db_session):
        store = DocumentStore(db_session)
        assert store.get_document(9999) is None

    def test_update_document_content(self, db_session):
        store = DocumentStore(db_session)
        doc = store.create_document("b.pdf", "b.pdf", DocumentType.TRANSCRIPT)
        store.update_document_content(doc.id, "Some text", {"page_count": 5})

        updated = store.get_document(doc.id)
        assert updated.content_text == "Some text"
        assert updated.metadata_["page_count"] == 5
        assert updated.status == ProcessingStatus.COMPLETED

    def test_list_documents(self, db_session):
        store = DocumentStore(db_session)
        store.create_document("c.pdf", "c.pdf", DocumentType.PDF)
        store.create_document("d.eml", "d.eml", DocumentType.EMAIL)
        store.create_document("e.txt", "e.txt", DocumentType.TEXT)

        all_docs = store.list_documents()
        assert len(all_docs) == 3

        pdf_docs = store.list_documents(doc_type=DocumentType.PDF)
        assert len(pdf_docs) == 1

    def test_delete_document(self, db_session):
        store = DocumentStore(db_session)
        doc = store.create_document("del.pdf", "del.pdf", DocumentType.PDF)
        doc_id = doc.id

        # delete_document tries to remove the file and vector chunks; both are
        # no-ops here (file doesn't exist, vector store not configured)
        result = store.delete_document(doc_id)
        assert result is True
        assert store.get_document(doc_id) is None

    def test_delete_nonexistent_document(self, db_session):
        store = DocumentStore(db_session)
        assert store.delete_document(9999) is False

    def test_save_and_get_timeline_events(self, db_session):
        store = DocumentStore(db_session)
        doc = store.create_document("ev.pdf", "ev.pdf", DocumentType.PDF)

        events = [
            {
                "document_id": doc.id,
                "description": "Custody hearing took place",
                "category": "hearing",
                "confidence": "high",
            }
        ]
        saved = store.save_timeline_events(events)
        assert len(saved) == 1

        fetched = store.get_timeline_events(document_ids=[doc.id])
        assert len(fetched) == 1
        assert fetched[0].description == "Custody hearing took place"

    def test_save_and_get_facts(self, db_session):
        store = DocumentStore(db_session)
        doc = store.create_document("facts.pdf", "facts.pdf", DocumentType.PDF)

        facts = [
            {
                "document_id": doc.id,
                "fact_text": "Defendant failed to pay child support in March 2023.",
                "category": "finance",
                "relevance": "high",
            }
        ]
        saved = store.save_facts(facts)
        assert len(saved) == 1

        fetched = store.get_facts(document_ids=[doc.id])
        assert len(fetched) == 1
        assert "child support" in fetched[0].fact_text

    def test_filter_facts_by_relevance(self, db_session):
        store = DocumentStore(db_session)
        doc = store.create_document("multi.pdf", "multi.pdf", DocumentType.PDF)

        facts = [
            {"document_id": doc.id, "fact_text": "High relevance fact.", "relevance": "high"},
            {"document_id": doc.id, "fact_text": "Low relevance fact.", "relevance": "low"},
        ]
        store.save_facts(facts)

        high = store.get_facts(relevance="high")
        assert len(high) == 1
        assert high[0].relevance == "high"
