"""Evidence analyser: RAG-powered querying, evidence summarisation, and cross-examination planning."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EVIDENCE_SYSTEM_PROMPT = """You are an expert legal analyst and strategist specialising in family court litigation.
You have access to a corpus of legal documents including court transcripts, police reports,
email communications, iMessage records, and other evidence.

When asked to analyse evidence for a particular topic or argument:
1. Synthesise the relevant retrieved passages into a clear, logical summary.
2. Identify the strongest pieces of evidence and explain why they are persuasive.
3. Note any gaps, inconsistencies, or weaknesses in the evidence.
4. Suggest how this evidence could be presented in motions or at a hearing.

Be factual, precise, and objective. Cite the source document and page number for every claim."""

CROSS_EXAM_SYSTEM_PROMPT = """You are an experienced trial attorney specialising in family court.
You are preparing cross-examination questions for a witness.

Using the provided evidence excerpts, create:
1. A list of key points to establish (admissions to elicit or facts to confirm).
2. A sequence of specific cross-examination questions for each point.
3. Expected answers and follow-up strategies.
4. Documents or exhibits that should be introduced.

Format your output clearly with numbered points and sub-questions.
Keep questions short, leading, and factually grounded in the provided evidence."""


class EvidenceAnalyzer:
    """Provides RAG-backed evidence analysis and cross-examination planning."""

    def __init__(self, llm_model: str, openai_api_key: str, vector_store) -> None:
        self._model = llm_model
        self._api_key = openai_api_key
        self._vector_store = vector_store

    def _get_llm(self):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=self._model, openai_api_key=self._api_key, temperature=0)

    # ------------------------------------------------------------------
    # General RAG query
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        top_k: int = 5,
        doc_type_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Answer a free-form question using RAG over the document corpus."""
        from langchain_core.messages import HumanMessage, SystemMessage

        filter_meta = {"doc_type": doc_type_filter} if doc_type_filter else None
        hits = self._vector_store.similarity_search(query, top_k=top_k, filter_metadata=filter_meta)

        context = self._format_context(hits)
        prompt = (
            f"Using the following excerpts from legal documents, answer the question:\n\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            f"Provide a detailed answer with citations (source file + page number)."
        )
        llm = self._get_llm()
        response = llm.invoke(
            [SystemMessage(content=EVIDENCE_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        return {
            "answer": response.content,
            "sources": [
                {
                    "text": h["text"][:300],
                    "filename": h["metadata"].get("filename"),
                    "page_number": h["metadata"].get("page_number"),
                    "doc_type": h["metadata"].get("doc_type"),
                    "score": h["score"],
                }
                for h in hits
            ],
        }

    # ------------------------------------------------------------------
    # Evidence summary for a motion / hearing argument
    # ------------------------------------------------------------------

    def gather_evidence(
        self,
        topic: str,
        top_k: int = 10,
        doc_type_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve and synthesise evidence relevant to a specific legal argument."""
        from langchain_core.messages import HumanMessage, SystemMessage

        filter_meta = {"doc_type": doc_type_filter} if doc_type_filter else None
        hits = self._vector_store.similarity_search(topic, top_k=top_k, filter_metadata=filter_meta)
        context = self._format_context(hits)

        prompt = (
            f"I need to build a legal argument on the following topic for a family court hearing:\n\n"
            f"TOPIC: {topic}\n\n"
            f"Here are the relevant excerpts retrieved from the document corpus:\n\n"
            f"{context}\n\n"
            f"Please provide:\n"
            f"1. A concise summary of the evidence supporting this argument.\n"
            f"2. The strongest supporting facts with document citations.\n"
            f"3. Any counter-arguments or weaknesses to anticipate.\n"
            f"4. Recommended strategy for presenting this at a hearing or in a motion."
        )
        llm = self._get_llm()
        response = llm.invoke(
            [SystemMessage(content=EVIDENCE_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        return {
            "analysis": response.content,
            "sources": [
                {
                    "text": h["text"][:300],
                    "filename": h["metadata"].get("filename"),
                    "page_number": h["metadata"].get("page_number"),
                    "doc_type": h["metadata"].get("doc_type"),
                    "score": h["score"],
                }
                for h in hits
            ],
        }

    # ------------------------------------------------------------------
    # Cross-examination planning
    # ------------------------------------------------------------------

    def plan_cross_examination(
        self,
        witness_name: str,
        topics: List[str],
        context_query: Optional[str] = None,
    ) -> str:
        """Generate a cross-examination plan for a specific witness."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Retrieve evidence for each topic
        query = context_query or f"evidence about {witness_name} " + " ".join(topics)
        hits = self._vector_store.similarity_search(query, top_k=15)
        context = self._format_context(hits)

        topics_str = "\n".join(f"- {t}" for t in topics)
        prompt = (
            f"Prepare a detailed cross-examination plan for the following witness:\n\n"
            f"WITNESS: {witness_name}\n\n"
            f"TOPICS TO COVER:\n{topics_str}\n\n"
            f"EVIDENCE EXCERPTS:\n{context}\n\n"
            f"Create a structured cross-examination plan including specific questions, "
            f"expected responses, and follow-up strategies. "
            f"Reference specific documents and page numbers where applicable."
        )
        llm = self._get_llm()
        response = llm.invoke(
            [SystemMessage(content=CROSS_EXAM_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        return response.content

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(hits: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for i, h in enumerate(hits, start=1):
            meta = h["metadata"]
            parts.append(
                f"[{i}] Source: {meta.get('filename', 'unknown')} "
                f"(page {meta.get('page_number', '?')}, type: {meta.get('doc_type', '?')})\n"
                f"{h['text']}"
            )
        return "\n\n".join(parts)
