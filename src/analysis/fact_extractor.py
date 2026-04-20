"""Fact extractor: pulls discrete legal facts from document text using an LLM."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

FACT_SYSTEM_PROMPT = """You are an expert legal analyst specialising in family court cases.
Your task is to extract key facts that would be useful as evidence in a custody / family court case.

For each fact output a JSON object with these fields:
- "fact_text": a concise, standalone factual statement (1-2 sentences)
- "category": one of ["custody", "communication", "parental_behaviour", "child_welfare",
  "finance", "court_order_violation", "substance_abuse", "mental_health", "domestic_violence",
  "other"]
- "relevance": "high" | "medium" | "low"  (high = directly relevant to case outcome)
- "supporting_quote": the verbatim excerpt from the text supporting this fact (or null)
- "source_page": page number (integer or null)

Return ONLY a valid JSON array of fact objects, no prose, no markdown fences."""

FACT_USER_TEMPLATE = """Document type: {doc_type}
Source file: {source}
Page: {page}

--- BEGIN TEXT ---
{text}
--- END TEXT ---

Extract all relevant facts and return the JSON array."""


class FactExtractor:
    """Uses an LLM to extract discrete legal facts from document pages."""

    def __init__(self, llm_model: str, openai_api_key: str) -> None:
        self._model = llm_model
        self._api_key = openai_api_key

    def _get_llm(self):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=self._model, openai_api_key=self._api_key, temperature=0)

    def extract_facts(
        self,
        document_id: int,
        pages: List[Dict[str, Any]],
        doc_type: str,
        source: str,
    ) -> List[Dict[str, Any]]:
        """Extract facts from all pages of a document.

        Returns list of dicts ready to insert into ExtractedFact table.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = self._get_llm()
        facts: List[Dict[str, Any]] = []

        for page in pages:
            text = page.get("text", "").strip()
            if not text:
                continue
            page_num = page.get("page_number", 1)
            try:
                prompt = FACT_USER_TEMPLATE.format(
                    doc_type=doc_type,
                    source=source,
                    page=page_num,
                    text=text[:4000],
                )
                response = llm.invoke(
                    [SystemMessage(content=FACT_SYSTEM_PROMPT), HumanMessage(content=prompt)]
                )
                raw = response.content.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                page_facts = json.loads(raw)
                if not isinstance(page_facts, list):
                    page_facts = []
                for f in page_facts:
                    f["document_id"] = document_id
                    if "source_page" not in f or f["source_page"] is None:
                        f["source_page"] = page_num
                    facts.append(f)
            except Exception as exc:
                logger.warning("Fact extraction failed on page %d of doc %d: %s", page_num, document_id, exc)

        return facts
