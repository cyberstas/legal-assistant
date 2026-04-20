"""Plain-text and transcript processor.

Handles .txt, .csv, and similar text-based formats.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Approx characters per "page" when we split a long text into logical chunks
CHARS_PER_PAGE = 3000


class TextProcessor:
    """Extract and chunk plain-text documents."""

    def process(self, file_path: str | Path) -> Dict[str, Any]:
        """Read a text file and split into pages.

        Returns:
            {
                "text": full text,
                "pages": [{"page_number": int, "text": str}, ...],
                "metadata": {"source": filename, "char_count": int},
            }
        """
        file_path = Path(file_path)
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.error("Failed to read %s: %s", file_path, exc)
            raw = ""

        pages = self._split_into_pages(raw)
        return {
            "text": raw,
            "pages": pages,
            "metadata": {
                "source": file_path.name,
                "char_count": len(raw),
            },
        }

    @staticmethod
    def _split_into_pages(text: str) -> List[Dict[str, Any]]:
        """Split text into page-sized chunks."""
        chunks: List[Dict[str, Any]] = []
        for i in range(0, max(1, len(text)), CHARS_PER_PAGE):
            chunk = text[i : i + CHARS_PER_PAGE].strip()
            if chunk:
                chunks.append({"page_number": len(chunks) + 1, "text": chunk})
        return chunks or [{"page_number": 1, "text": ""}]
