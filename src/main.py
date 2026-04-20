"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.models.database import init_db
from src.api.routes import documents, timeline, analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising database…")
    settings.ensure_dirs()
    init_db()
    logger.info("Legal Assistant API ready.")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Legal Assistant API",
        description=(
            "Backend processing engine for family court legal documents. "
            "Supports ingestion of PDFs, images, emails, transcripts, and more. "
            "Provides RAG-powered querying, timeline extraction, fact extraction, "
            "evidence analysis, and cross-examination planning."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(documents.router)
    app.include_router(timeline.router)
    app.include_router(analysis.router)

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok", "version": "0.1.0"}

    return app


app = create_app()
