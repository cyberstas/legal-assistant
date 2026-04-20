"""SQLAlchemy database session factory and helpers."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from src.config import settings
from src.models.schemas import Base

_engine: Optional[Engine] = None
_SessionLocal = None


def _make_engine(url: str) -> Engine:
    connect_args: dict = {}
    kwargs: dict = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # Use StaticPool for in-memory SQLite so all sessions share the same DB
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
    return create_engine(url, connect_args=connect_args, **kwargs)


def get_engine() -> Engine:
    """Return (and lazily create) the global SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = _make_engine(settings.database_url)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def init_db() -> None:
    """Create all tables using the current engine."""
    Base.metadata.create_all(bind=get_engine())


def get_db():
    """FastAPI dependency that yields a database session."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def reset_engine(url: Optional[str] = None) -> None:
    """Replace the global engine (used in tests to inject an in-memory DB)."""
    global _engine, _SessionLocal
    _engine = _make_engine(url or settings.database_url)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
