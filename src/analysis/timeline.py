"""Timeline builder: extracts chronological events from document text using an LLM."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__)

TIMELINE_SYSTEM_PROMPT = """You are a legal analyst assistant specialising in family court cases.
Your task is to extract all chronological events from the provided document text.

For each event output a JSON object with these fields:
- "event_date_raw": the date/time string as it appears in the text (or null)
- "description": concise description of the event (1-2 sentences)
- "participants": list of person names involved (empty list if none)
- "category": one of ["hearing", "communication", "custody", "incident", "legal_filing",
  "medical", "financial", "police", "other"]
- "confidence": "high" | "medium" | "low"
- "source_page": page number where this event was found (integer or null)

Return ONLY a valid JSON array of event objects, no prose, no markdown fences."""

TIMELINE_USER_TEMPLATE = """Document type: {doc_type}
Source file: {source}
Page number(s): {pages}

--- BEGIN TEXT ---
{text}
--- END TEXT ---

Extract all events and return the JSON array."""


class TimelineBuilder:
    """Uses an LLM to extract dated events from document chunks."""

    def __init__(self, llm_model: str, openai_api_key: str) -> None:
        self._model = llm_model
        self._api_key = openai_api_key

    def _get_llm(self):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=self._model, openai_api_key=self._api_key, temperature=0)

    def extract_events(
        self,
        document_id: int,
        pages: List[Dict[str, Any]],
        doc_type: str,
        source: str,
    ) -> List[Dict[str, Any]]:
        """Extract timeline events from all pages of a document.

        Returns list of dicts ready to insert into TimelineEvent table.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = self._get_llm()
        events: List[Dict[str, Any]] = []

        for page in pages:
            text = page.get("text", "").strip()
            if not text:
                continue
            page_num = page.get("page_number", 1)
            try:
                prompt = TIMELINE_USER_TEMPLATE.format(
                    doc_type=doc_type,
                    source=source,
                    pages=page_num,
                    text=text[:4000],  # keep within token budget per page
                )
                response = llm.invoke(
                    [SystemMessage(content=TIMELINE_SYSTEM_PROMPT), HumanMessage(content=prompt)]
                )
                raw = response.content.strip()
                # Strip markdown fences if present
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                page_events = json.loads(raw)
                if not isinstance(page_events, list):
                    page_events = []
                for ev in page_events:
                    ev["document_id"] = document_id
                    ev["event_date"] = self._parse_date(ev.get("event_date_raw"))
                    if "source_page" not in ev or ev["source_page"] is None:
                        ev["source_page"] = page_num
                    events.append(ev)
            except Exception as exc:
                logger.warning("Timeline extraction failed on page %d of doc %d: %s", page_num, document_id, exc)

        return events

    @staticmethod
    def _parse_date(raw: Optional[str]) -> Optional[datetime]:
        if not raw:
            return None
        try:
            return dateutil_parser.parse(raw, fuzzy=True)
        except Exception:
            return None
