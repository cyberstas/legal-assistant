"""Analysis endpoints: RAG query, evidence gathering, cross-examination planning."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.config import settings
from src.models.database import get_db
from src.models.schemas import (
    CrossExamRequest,
    EvidenceRequest,
    ExtractedFactSchema,
    QueryRequest,
    QueryResponse,
)
from src.analysis.evidence_analyzer import EvidenceAnalyzer
from src.storage.document_store import DocumentStore, get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis", tags=["analysis"])


def _require_api_key():
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured. Set it in .env to enable LLM features.",
        )


def _get_analyzer() -> EvidenceAnalyzer:
    _require_api_key()
    return EvidenceAnalyzer(
        llm_model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        vector_store=get_vector_store(),
    )


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Answer a free-form question using RAG over the entire document corpus.

    Example questions:
    - "When did the last custody exchange take place?"
    - "What did the ex-wife say about the children's school schedule?"
    - "List all instances of late child support payments."
    """
    analyzer = _get_analyzer()
    try:
        result = analyzer.query(
            query=request.query,
            top_k=request.top_k,
            doc_type_filter=request.doc_type_filter.value if request.doc_type_filter else None,
        )
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(**result)


@router.post("/evidence")
def gather_evidence(request: EvidenceRequest):
    """Gather and synthesise evidence for a specific legal argument or topic.

    The system retrieves the most relevant document passages and asks the LLM
    to produce a structured evidence brief, including strengths, weaknesses,
    and presentation strategy.
    """
    analyzer = _get_analyzer()
    try:
        result = analyzer.gather_evidence(
            topic=request.topic,
            top_k=request.top_k,
            doc_type_filter=request.doc_type_filter.value if request.doc_type_filter else None,
        )
    except Exception as exc:
        logger.error("Evidence gathering failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@router.post("/cross-examination")
def plan_cross_examination(request: CrossExamRequest):
    """Generate a detailed cross-examination plan for a witness.

    Provide the witness name and the topics you want to cover.
    The system retrieves relevant evidence and creates sequenced questions
    with expected answers and follow-up strategies.
    """
    analyzer = _get_analyzer()
    try:
        plan = analyzer.plan_cross_examination(
            witness_name=request.witness_name,
            topics=request.topics,
            context_query=request.context_query,
        )
    except Exception as exc:
        logger.error("Cross-examination planning failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"plan": plan}


@router.get("/facts", response_model=List[ExtractedFactSchema])
def get_all_facts(
    category: Optional[str] = None,
    relevance: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Retrieve extracted facts across all documents, with optional filters.

    Useful for quickly surfacing high-relevance evidence for motions.
    """
    store = DocumentStore(db)
    return store.get_facts(category=category, relevance=relevance)
