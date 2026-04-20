"""Pydantic schemas and SQLAlchemy ORM models."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DocumentType(str, enum.Enum):
    PDF = "pdf"
    IMAGE = "image"
    EMAIL = "email"
    TEXT = "text"
    TRANSCRIPT = "transcript"
    POLICE_REPORT = "police_report"
    IMESSAGE = "imessage"
    UNKNOWN = "unknown"


class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# SQLAlchemy ORM
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class Document(Base):
    """Persisted metadata for every ingested document."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(512), nullable=False)
    original_filename = Column(String(512), nullable=False)
    doc_type = Column(Enum(DocumentType), nullable=False, default=DocumentType.UNKNOWN)
    status = Column(Enum(ProcessingStatus), nullable=False, default=ProcessingStatus.PENDING)
    content_text = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    events = relationship("TimelineEvent", back_populates="document", cascade="all, delete-orphan")
    facts = relationship("ExtractedFact", back_populates="document", cascade="all, delete-orphan")


class TimelineEvent(Base):
    """A date-anchored event extracted from a document."""

    __tablename__ = "timeline_events"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    event_date = Column(DateTime, nullable=True)
    event_date_raw = Column(String(256), nullable=True)  # original unparsed date string
    description = Column(Text, nullable=False)
    participants = Column(JSON, nullable=True)   # list of names involved
    category = Column(String(128), nullable=True)  # e.g. "hearing", "communication", "incident"
    source_page = Column(Integer, nullable=True)
    confidence = Column(String(16), nullable=True)  # "high" | "medium" | "low"
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="events")


class ExtractedFact(Base):
    """A discrete fact extracted from a document, useful for motions / cross-examination."""

    __tablename__ = "extracted_facts"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    fact_text = Column(Text, nullable=False)
    category = Column(String(128), nullable=True)   # e.g. "custody", "communication", "finance"
    relevance = Column(String(16), nullable=True)   # "high" | "medium" | "low"
    source_page = Column(Integer, nullable=True)
    supporting_quote = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="facts")


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class DocumentSchema(BaseModel):
    id: int
    filename: str
    original_filename: str
    doc_type: DocumentType
    status: ProcessingStatus
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class DocumentDetailSchema(DocumentSchema):
    content_text: Optional[str] = None
    doc_metadata: Optional[Dict[str, Any]] = Field(None, alias="metadata_")

    model_config = {"from_attributes": True, "populate_by_name": True}


class TimelineEventSchema(BaseModel):
    id: int
    document_id: int
    event_date: Optional[datetime] = None
    event_date_raw: Optional[str] = None
    description: str
    participants: Optional[List[str]] = None
    category: Optional[str] = None
    source_page: Optional[int] = None
    confidence: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ExtractedFactSchema(BaseModel):
    id: int
    document_id: int
    fact_text: str
    category: Optional[str] = None
    relevance: Optional[str] = None
    source_page: Optional[int] = None
    supporting_quote: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural-language question to answer against the document corpus")
    top_k: int = Field(5, ge=1, le=20, description="Number of document chunks to retrieve")
    doc_type_filter: Optional[DocumentType] = Field(None, description="Restrict search to a specific document type")


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


class TimelineRequest(BaseModel):
    document_ids: Optional[List[int]] = Field(None, description="Restrict to specific document IDs; omit for all")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    category: Optional[str] = None


class EvidenceRequest(BaseModel):
    topic: str = Field(..., description="Topic or argument to gather evidence for")
    doc_type_filter: Optional[DocumentType] = None
    top_k: int = Field(10, ge=1, le=50)


class CrossExamRequest(BaseModel):
    witness_name: str = Field(..., description="Name of the witness to be cross-examined")
    topics: List[str] = Field(..., description="Topics / allegations to cover in cross-examination")
    context_query: Optional[str] = Field(None, description="Optional additional context query")
