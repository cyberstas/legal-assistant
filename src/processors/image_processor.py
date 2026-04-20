"""Image document processor.

Uses Pillow + pytesseract for OCR.  Falls back to a descriptive placeholder
when tesseract is unavailable so the rest of the pipeline still works.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class ImageProcessor:
    """Extract text from images via OCR."""

    def __init__(self, tesseract_cmd: str | None = None) -> None:
        if tesseract_cmd:
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            except ImportError:
                logger.warning("pytesseract not installed; OCR will be unavailable")

    def process(self, file_path: str | Path) -> Dict[str, Any]:
        """Run OCR on an image file.

        Returns:
            {
                "text": extracted text,
                "pages": [{"page_number": 1, "text": str}],
                "metadata": {"source": filename, "ocr_used": bool},
            }
        """
        file_path = Path(file_path)
        text = ""
        ocr_used = False

        try:
            from PIL import Image
            import pytesseract

            image = Image.open(str(file_path))
            text = pytesseract.image_to_string(image)
            ocr_used = True
        except Exception as exc:
            logger.warning("OCR failed for %s: %s", file_path.name, exc)
            text = f"[Image file: {file_path.name} – OCR unavailable]"

        text = text.strip()
        return {
            "text": text,
            "pages": [{"page_number": 1, "text": text}],
            "metadata": {
                "source": file_path.name,
                "ocr_used": ocr_used,
            },
        }
