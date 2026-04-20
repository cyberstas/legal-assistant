"""Document ingestion and management endpoints."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.analysis.fact_extractor import FactExtractor
from src.analysis.timeline import TimelineBuilder
from src.config import settings
from src.models.database import get_db
from src.models.schemas import (
    DocumentDetailSchema,
    DocumentSchema,
    DocumentType,
    ExtractedFactSchema,
    ProcessingStatus,
    TimelineEventSchema,
)
from src.processors.dispatcher import detect_document_type, process_document
from src.storage.document_store import DocumentStore, get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentSchema, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    doc_type_override: Optional[DocumentType] = Form(None),
    db: Session = Depends(get_db),
):
    """Upload and process a legal document.

    Supported formats: PDF, images (JPG/PNG/TIFF), EML emails, plain text.
    The document is processed synchronously: text is extracted, embedded, and
    timeline events + facts are extracted via LLM (requires OPENAI_API_KEY).
    """
    settings.ensure_dirs()

    original_filename = file.filename or "unknown"
    ext = Path(original_filename).suffix.lower()
    saved_filename = f"{uuid.uuid4().hex}{ext}"
    save_path = Path(settings.upload_dir) / saved_filename

    # Save uploaded file
    try:
        with save_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}") from exc

    doc_type = doc_type_override or detect_document_type(original_filename)
    store = DocumentStore(db)
    doc = store.create_document(
        filename=saved_filename,
        original_filename=original_filename,
        doc_type=doc_type,
    )

    # Process in the request (for simplicity – swap for a task queue in production)
    try:
        store.update_document_content(doc.id, "", {}, status=ProcessingStatus.PROCESSING)
        result = process_document(save_path, doc_type=doc_type)

        store.update_document_content(
            doc.id,
            result["text"],
            result.get("metadata", {}),
            status=ProcessingStatus.COMPLETED,
        )

        # Embed chunks into vector store (only if API key is set)
        if settings.openai_api_key:
            try:
                get_vector_store().add_document_chunks(
                    document_id=doc.id,
                    doc_type=doc_type.value,
                    filename=original_filename,
                    pages=result["pages"],
                )
            except Exception as exc:
                logger.warning("Vector embedding failed for doc %d: %s", doc.id, exc)

            # Extract timeline events
            try:
                builder = TimelineBuilder(settings.llm_model, settings.openai_api_key)
                events = builder.extract_events(
                    document_id=doc.id,
                    pages=result["pages"],
                    doc_type=doc_type.value,
                    source=original_filename,
                )
                if events:
                    store.save_timeline_events(events)
            except Exception as exc:
                logger.warning("Timeline extraction failed for doc %d: %s", doc.id, exc)

            # Extract facts
            try:
                extractor = FactExtractor(settings.llm_model, settings.openai_api_key)
                facts = extractor.extract_facts(
                    document_id=doc.id,
                    pages=result["pages"],
                    doc_type=doc_type.value,
                    source=original_filename,
                )
                if facts:
                    store.save_facts(facts)
            except Exception as exc:
                logger.warning("Fact extraction failed for doc %d: %s", doc.id, exc)

    except Exception as exc:
        logger.error("Processing failed for doc %d: %s", doc.id, exc)
        store.update_document_content(doc.id, "", {}, status=ProcessingStatus.FAILED)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {exc}") from exc

    db.refresh(doc)
    return doc


@router.get("", response_model=List[DocumentSchema])
def list_documents(
    doc_type: Optional[DocumentType] = None,
    status: Optional[ProcessingStatus] = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """List all ingested documents with optional filters."""
    store = DocumentStore(db)
    return store.list_documents(doc_type=doc_type, status=status, skip=skip, limit=limit)


@router.get("/{document_id}", response_model=DocumentDetailSchema)
def get_document(document_id: int, db: Session = Depends(get_db)):
    """Get full details including extracted text for a single document."""
    store = DocumentStore(db)
    doc = store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document and all its associated data."""
    store = DocumentStore(db)
    if not store.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")


@router.get("/{document_id}/timeline", response_model=List[TimelineEventSchema])
def get_document_timeline(document_id: int, db: Session = Depends(get_db)):
    """Get all timeline events extracted from a specific document."""
    store = DocumentStore(db)
    doc = store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return store.get_timeline_events(document_ids=[document_id])


@router.get("/{document_id}/facts", response_model=List[ExtractedFactSchema])
def get_document_facts(document_id: int, db: Session = Depends(get_db)):
    """Get all extracted facts from a specific document."""
    store = DocumentStore(db)
    doc = store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return store.get_facts(document_ids=[document_id])
