"""Document store: coordinates DB persistence and vector indexing."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.config import settings
from src.models.schemas import (
    Document,
    DocumentType,
    ExtractedFact,
    ProcessingStatus,
    TimelineEvent,
)
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(
            persist_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    return _vector_store


class DocumentStore:
    """High-level service for persisting and retrieving legal documents."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    def create_document(
        self,
        filename: str,
        original_filename: str,
        doc_type: DocumentType,
    ) -> Document:
        doc = Document(
            filename=filename,
            original_filename=original_filename,
            doc_type=doc_type,
            status=ProcessingStatus.PENDING,
        )
        self._db.add(doc)
        self._db.commit()
        self._db.refresh(doc)
        return doc

    def update_document_content(
        self,
        document_id: int,
        text: str,
        metadata: Dict[str, Any],
        status: ProcessingStatus = ProcessingStatus.COMPLETED,
    ) -> None:
        doc = self._db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.content_text = text
            doc.metadata_ = metadata
            doc.status = status
            self._db.commit()

    def get_document(self, document_id: int) -> Optional[Document]:
        return self._db.query(Document).filter(Document.id == document_id).first()

    def list_documents(
        self,
        doc_type: Optional[DocumentType] = None,
        status: Optional[ProcessingStatus] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> List[Document]:
        q = self._db.query(Document)
        if doc_type:
            q = q.filter(Document.doc_type == doc_type)
        if status:
            q = q.filter(Document.status == status)
        return q.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()

    def delete_document(self, document_id: int) -> bool:
        doc = self._db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return False
        # Remove physical file
        file_path = Path(settings.upload_dir) / doc.filename
        if file_path.exists():
            file_path.unlink()
        # Remove vector store chunks (only when vector storage is enabled)
        if settings.openai_api_key:
            try:
                get_vector_store().delete_document_chunks(document_id)
            except Exception as exc:
                logger.warning("Could not delete vector chunks for doc %d: %s", document_id, exc)
        self._db.delete(doc)
        self._db.commit()
        return True

    # ------------------------------------------------------------------
    # Timeline events
    # ------------------------------------------------------------------

    def save_timeline_events(self, events: List[Dict[str, Any]]) -> List[TimelineEvent]:
        objs: List[TimelineEvent] = []
        for ev in events:
            obj = TimelineEvent(**ev)
            self._db.add(obj)
            objs.append(obj)
        self._db.commit()
        return objs

    def get_timeline_events(
        self,
        document_ids: Optional[List[int]] = None,
        start_date=None,
        end_date=None,
        category: Optional[str] = None,
    ) -> List[TimelineEvent]:
        q = self._db.query(TimelineEvent)
        if document_ids:
            q = q.filter(TimelineEvent.document_id.in_(document_ids))
        if start_date:
            q = q.filter(TimelineEvent.event_date >= start_date)
        if end_date:
            q = q.filter(TimelineEvent.event_date <= end_date)
        if category:
            q = q.filter(TimelineEvent.category == category)
        return q.order_by(TimelineEvent.event_date.asc()).all()

    # ------------------------------------------------------------------
    # Extracted facts
    # ------------------------------------------------------------------

    def save_facts(self, facts: List[Dict[str, Any]]) -> List[ExtractedFact]:
        objs: List[ExtractedFact] = []
        for f in facts:
            obj = ExtractedFact(**f)
            self._db.add(obj)
            objs.append(obj)
        self._db.commit()
        return objs

    def get_facts(
        self,
        document_ids: Optional[List[int]] = None,
        category: Optional[str] = None,
        relevance: Optional[str] = None,
    ) -> List[ExtractedFact]:
        q = self._db.query(ExtractedFact)
        if document_ids:
            q = q.filter(ExtractedFact.document_id.in_(document_ids))
        if category:
            q = q.filter(ExtractedFact.category == category)
        if relevance:
            q = q.filter(ExtractedFact.relevance == relevance)
        return q.order_by(ExtractedFact.created_at.desc()).all()
