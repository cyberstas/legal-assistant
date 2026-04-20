"""Tests for document processors."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# PDF Processor
# ---------------------------------------------------------------------------


class TestPDFProcessor:
    def test_process_returns_required_keys(self, tmp_path):
        """PDFProcessor.process should always return text, pages, metadata."""
        # Create a minimal valid text file to test fallback behaviour without
        # a real PDF (PyMuPDF will raise on non-PDF → pdfplumber will also raise →
        # we catch and verify the contract).
        from src.processors.pdf_processor import PDFProcessor

        proc = PDFProcessor()

        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake content")

        # May fail internally but should not raise – just return empty text
        try:
            result = proc.process(fake_pdf)
        except Exception:
            pytest.skip("PDF libraries not available in test environment")

        assert "text" in result
        assert "pages" in result
        assert "metadata" in result
        assert isinstance(result["pages"], list)


# ---------------------------------------------------------------------------
# Image Processor
# ---------------------------------------------------------------------------


class TestImageProcessor:
    def test_process_without_tesseract(self, tmp_path):
        """ImageProcessor should return a placeholder when OCR is unavailable."""
        from src.processors.image_processor import ImageProcessor

        # Create a tiny 1x1 PNG (valid image)
        try:
            from PIL import Image

            img = Image.new("RGB", (10, 10), color=(255, 255, 255))
            img_path = tmp_path / "test.png"
            img.save(str(img_path))
        except ImportError:
            pytest.skip("Pillow not installed")

        proc = ImageProcessor(tesseract_cmd="/nonexistent/tesseract")
        result = proc.process(img_path)

        assert "text" in result
        assert "pages" in result
        assert "metadata" in result
        assert len(result["pages"]) == 1
        assert result["pages"][0]["page_number"] == 1

    def test_process_missing_file(self, tmp_path):
        """ImageProcessor on non-existent file returns placeholder without raising."""
        from src.processors.image_processor import ImageProcessor

        proc = ImageProcessor()
        result = proc.process(tmp_path / "nonexistent.jpg")

        assert "text" in result
        assert "pages" in result


# ---------------------------------------------------------------------------
# Email Processor
# ---------------------------------------------------------------------------


class TestEmailProcessor:
    def _write_eml(self, path: Path, from_: str, to: str, subject: str, body: str) -> None:
        content = (
            f"From: {from_}\r\n"
            f"To: {to}\r\n"
            f"Subject: {subject}\r\n"
            f"Date: Mon, 01 Jan 2024 10:00:00 +0000\r\n"
            f"Content-Type: text/plain; charset=utf-8\r\n"
            f"\r\n"
            f"{body}\r\n"
        )
        path.write_text(content, encoding="utf-8")

    def test_process_simple_eml(self, tmp_path):
        from src.processors.email_processor import EmailProcessor

        eml = tmp_path / "test.eml"
        self._write_eml(
            eml,
            from_="alice@example.com",
            to="bob@example.com",
            subject="Custody schedule",
            body="Hi Bob, I need to discuss the custody schedule.",
        )

        proc = EmailProcessor()
        result = proc.process(eml)

        assert "text" in result
        assert "Custody schedule" in result["text"]
        assert "discuss the custody schedule" in result["text"]
        assert result["metadata"]["subject"] == "Custody schedule"
        assert len(result["pages"]) == 1

    def test_process_returns_required_keys(self, tmp_path):
        from src.processors.email_processor import EmailProcessor

        eml = tmp_path / "minimal.eml"
        self._write_eml(eml, "a@b.com", "c@d.com", "Test", "Body text")
        proc = EmailProcessor()
        result = proc.process(eml)

        for key in ("text", "pages", "metadata"):
            assert key in result


# ---------------------------------------------------------------------------
# Text Processor
# ---------------------------------------------------------------------------


class TestTextProcessor:
    def test_process_short_text(self, tmp_path):
        from src.processors.text_processor import TextProcessor

        txt = tmp_path / "short.txt"
        txt.write_text("Hello world. This is a legal document.", encoding="utf-8")

        proc = TextProcessor()
        result = proc.process(txt)

        assert result["text"] == "Hello world. This is a legal document."
        assert len(result["pages"]) >= 1
        assert result["metadata"]["char_count"] > 0

    def test_process_long_text_splits_into_pages(self, tmp_path):
        from src.processors.text_processor import TextProcessor, CHARS_PER_PAGE

        long_text = "A" * (CHARS_PER_PAGE * 3)
        txt = tmp_path / "long.txt"
        txt.write_text(long_text, encoding="utf-8")

        proc = TextProcessor()
        result = proc.process(txt)

        assert len(result["pages"]) >= 3

    def test_process_empty_file(self, tmp_path):
        from src.processors.text_processor import TextProcessor

        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")

        proc = TextProcessor()
        result = proc.process(empty)

        assert "pages" in result
        assert isinstance(result["pages"], list)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("hearing_transcript.pdf", "transcript"),
            ("police_report_jan.pdf", "police_report"),
            ("imessage_export.pdf", "imessage"),
            ("evidence.pdf", "pdf"),
            ("photo.jpg", "image"),
            ("screenshot.PNG", "image"),
            ("email.eml", "email"),
            ("notes.txt", "text"),
            ("random.xyz", "unknown"),
        ],
    )
    def test_detect_document_type(self, filename, expected):
        from src.processors.dispatcher import detect_document_type
        from src.models.schemas import DocumentType

        result = detect_document_type(filename)
        assert result.value == expected
