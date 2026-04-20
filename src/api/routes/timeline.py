"""Timeline endpoints."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.models.database import get_db
from src.models.schemas import TimelineEventSchema, TimelineRequest
from src.storage.document_store import DocumentStore

router = APIRouter(prefix="/timeline", tags=["timeline"])


@router.post("", response_model=List[TimelineEventSchema])
def get_timeline(request: TimelineRequest, db: Session = Depends(get_db)):
    """Return a filtered, chronologically sorted timeline of events.

    Optionally filter by document IDs, date range, and/or category.
    """
    store = DocumentStore(db)
    events = store.get_timeline_events(
        document_ids=request.document_ids,
        start_date=request.start_date,
        end_date=request.end_date,
        category=request.category,
    )
    return events


@router.get("", response_model=List[TimelineEventSchema])
def get_full_timeline(
    category: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Return the complete timeline across all documents."""
    store = DocumentStore(db)
    return store.get_timeline_events(category=category)
