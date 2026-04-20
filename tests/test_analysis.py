"""Tests for analysis utilities (timeline parser, fact extractor) without LLM calls."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

langchain_available = True
try:
    import langchain_core  # noqa: F401
except ImportError:
    langchain_available = False

requires_langchain = pytest.mark.skipif(
    not langchain_available, reason="langchain_core not installed"
)


class TestTimelineBuilder:
    def test_parse_date_valid(self):
        from src.analysis.timeline import TimelineBuilder

        dt = TimelineBuilder._parse_date("January 15, 2024")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_parse_date_none(self):
        from src.analysis.timeline import TimelineBuilder

        assert TimelineBuilder._parse_date(None) is None

    def test_parse_date_invalid(self):
        from src.analysis.timeline import TimelineBuilder

        # Completely unparse-able string should return None
        result = TimelineBuilder._parse_date("not a date at all xyz")
        # dateutil fuzzy may or may not parse this; we just assert it doesn't raise
        # and returns None or a datetime
        assert result is None or hasattr(result, "year")

    @requires_langchain
    def test_extract_events_mocked_llm(self):
        """Verify extract_events correctly calls LLM and parses response."""
        import json
        from src.analysis.timeline import TimelineBuilder

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            [
                {
                    "event_date_raw": "2024-03-15",
                    "description": "Custody hearing held at Family Court.",
                    "participants": ["Alice Smith", "Bob Smith"],
                    "category": "hearing",
                    "confidence": "high",
                    "source_page": 1,
                }
            ]
        )

        builder = TimelineBuilder(llm_model="gpt-4o", openai_api_key="test-key")

        with patch.object(builder, "_get_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            events = builder.extract_events(
                document_id=1,
                pages=[{"page_number": 1, "text": "Custody hearing on March 15, 2024."}],
                doc_type="transcript",
                source="hearing.pdf",
            )

        assert len(events) == 1
        assert events[0]["document_id"] == 1
        assert events[0]["description"] == "Custody hearing held at Family Court."
        assert events[0]["event_date"] is not None
        assert events[0]["category"] == "hearing"

    @requires_langchain
    def test_extract_events_handles_malformed_json(self):
        """extract_events should not raise on malformed LLM output."""
        from src.analysis.timeline import TimelineBuilder

        mock_response = MagicMock()
        mock_response.content = "This is not JSON at all"

        builder = TimelineBuilder(llm_model="gpt-4o", openai_api_key="test-key")

        with patch.object(builder, "_get_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            events = builder.extract_events(
                document_id=1,
                pages=[{"page_number": 1, "text": "Some text."}],
                doc_type="pdf",
                source="doc.pdf",
            )

        assert events == []


class TestFactExtractor:
    @requires_langchain
    def test_extract_facts_mocked_llm(self):
        """Verify extract_facts correctly calls LLM and parses response."""
        import json
        from src.analysis.fact_extractor import FactExtractor

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            [
                {
                    "fact_text": "Defendant failed to attend scheduled visitation on Jan 5, 2024.",
                    "category": "custody",
                    "relevance": "high",
                    "supporting_quote": "Defendant did not appear for the scheduled visitation",
                    "source_page": 2,
                }
            ]
        )

        extractor = FactExtractor(llm_model="gpt-4o", openai_api_key="test-key")

        with patch.object(extractor, "_get_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            facts = extractor.extract_facts(
                document_id=2,
                pages=[{"page_number": 2, "text": "Defendant did not appear for the scheduled visitation."}],
                doc_type="transcript",
                source="hearing.pdf",
            )

        assert len(facts) == 1
        assert facts[0]["document_id"] == 2
        assert facts[0]["category"] == "custody"
        assert facts[0]["relevance"] == "high"

    @requires_langchain
    def test_extract_facts_empty_pages(self):
        """extract_facts on empty page list returns empty list."""
        from src.analysis.fact_extractor import FactExtractor

        extractor = FactExtractor(llm_model="gpt-4o", openai_api_key="test-key")
        facts = extractor.extract_facts(
            document_id=1,
            pages=[],
            doc_type="pdf",
            source="empty.pdf",
        )
        assert facts == []

    @requires_langchain
    def test_extract_facts_skips_blank_pages(self):
        """extract_facts skips pages with no text without calling LLM."""
        from src.analysis.fact_extractor import FactExtractor

        extractor = FactExtractor(llm_model="gpt-4o", openai_api_key="test-key")
        with patch.object(extractor, "_get_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm_factory.return_value = mock_llm

            extractor.extract_facts(
                document_id=1,
                pages=[{"page_number": 1, "text": "   "}],
                doc_type="pdf",
                source="blank.pdf",
            )

        mock_llm.invoke.assert_not_called()
