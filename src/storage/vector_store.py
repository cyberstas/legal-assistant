"""Vector store wrapper around ChromaDB + LangChain embeddings.

Stores document chunks with metadata so they can be retrieved via
semantic similarity search (RAG).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    COLLECTION_NAME = "legal_documents"

    def __init__(self, persist_dir: str, embedding_model: str, openai_api_key: str) -> None:
        self._persist_dir = persist_dir
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
            from langchain_chroma import Chroma

            self._collection = Chroma(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self._get_embeddings(),
                persist_directory=self._persist_dir,
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

        Returns list of {page_content, metadata, score} dicts.
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
        collection = self._get_collection()
        collection.delete(where={"document_id": document_id})
