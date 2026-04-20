"""Vector store wrapper around PostgreSQL + pgvector + LangChain embeddings.

Stores document chunks with metadata so they can be retrieved via
semantic similarity search (RAG).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around a PGVector collection backed by PostgreSQL."""

    COLLECTION_NAME = "legal_documents"

    def __init__(self, connection_string: str, embedding_model: str, openai_api_key: str) -> None:
        self._connection_string = connection_string
        self._embedding_model = embedding_model
        self._openai_api_key = openai_api_key
        self._collection = None
        self._embeddings = None

    # ------------------------------------------------------------------
    # Lazy initialisation – avoids importing heavy libs at module load
    # ------------------------------------------------------------------

    def _get_embeddings(self):
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                model=self._embedding_model,
                openai_api_key=self._openai_api_key,
            )
        return self._embeddings

    def _get_collection(self):
        if self._collection is None:
            from langchain_postgres.vectorstores import PGVector

            self._collection = PGVector(
                embeddings=self._get_embeddings(),
                collection_name=self.COLLECTION_NAME,
                connection=self._connection_string,
                use_jsonb=True,
            )
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document_chunks(
        self,
        document_id: int,
        doc_type: str,
        filename: str,
        pages: List[Dict[str, Any]],
    ) -> List[str]:
        """Embed and store each page chunk.  Returns list of inserted IDs."""
        from langchain_core.documents import Document as LCDocument

        docs: List[LCDocument] = []
        for page in pages:
            text = page.get("text", "").strip()
            if not text:
                continue
            chunk_id = str(uuid.uuid4())
            docs.append(
                LCDocument(
                    page_content=text,
                    metadata={
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "doc_type": doc_type,
                        "filename": filename,
                        "page_number": page.get("page_number", 1),
                    },
                )
            )

        if not docs:
            return []

        collection = self._get_collection()
        ids = [d.metadata["chunk_id"] for d in docs]
        collection.add_documents(docs, ids=ids)
        return ids

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over stored chunks.

        Returns list of {text, metadata, score} dicts.
        """
        collection = self._get_collection()
        kwargs: Dict[str, Any] = {"k": top_k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata

        results = collection.similarity_search_with_relevance_scores(query, **kwargs)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for doc, score in results
        ]

    def delete_document_chunks(self, document_id: int) -> None:
        """Remove all chunks belonging to a given document."""
        from sqlalchemy import text

        collection = self._get_collection()
        # _make_sync_session is langchain_postgres' internal session factory.
        # There is no public API for metadata-filtered deletion in PGVector, so
        # we use a direct SQL DELETE against the underlying tables.
        with collection._make_sync_session() as session:
            session.execute(
                text(
                    "DELETE FROM langchain_pg_embedding e "
                    "USING langchain_pg_collection c "
                    "WHERE e.collection_id = c.uuid "
                    "AND c.name = :name "
                    "AND (e.cmetadata->>'document_id')::int = :doc_id"
                ),
                {"name": self.COLLECTION_NAME, "doc_id": document_id},
            )
            session.commit()
