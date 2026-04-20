"""PDF document processor.

Extracts text from PDF files using PyMuPDF (fitz) with a pdfplumber fallback.
Each page is returned as a separate chunk to preserve source-page metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Extract text and metadata from PDF documents."""

    def process(self, file_path: str | Path) -> Dict[str, Any]:
        """Process a PDF file and return extracted content.

        Returns:
            {
                "text": full concatenated text,
                "pages": [{"page_number": int, "text": str}, ...],
                "metadata": {title, author, page_count, ...},
            }
        """
        file_path = Path(file_path)
        pages: List[Dict[str, Any]] = []
        doc_metadata: Dict[str, Any] = {}

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(file_path))
            doc_metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "page_count": len(doc),
                "source": file_path.name,
            }
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                pages.append({"page_number": page_num, "text": text.strip()})
            doc.close()
        except Exception as exc:
            logger.warning("PyMuPDF failed (%s), trying pdfplumber", exc)
            try:
                pages = self._pdfplumber_extract(file_path)
            except Exception as fallback_exc:
                logger.warning("pdfplumber also failed for %s: %s", file_path.name, fallback_exc)
                pages = []
            doc_metadata["source"] = file_path.name

        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        return {
            "text": full_text,
            "pages": pages,
            "metadata": doc_metadata,
        }

    @staticmethod
    def _pdfplumber_extract(file_path: Path) -> List[Dict[str, Any]]:
        import pdfplumber

        pages: List[Dict[str, Any]] = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append({"page_number": page_num, "text": text.strip()})
        return pages
