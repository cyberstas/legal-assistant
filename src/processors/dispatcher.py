"""Document processor dispatcher.

Routes an uploaded file to the correct processor based on its MIME type /
file extension and returns a normalised result dict.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from src.models.schemas import DocumentType
from src.processors.pdf_processor import PDFProcessor
from src.processors.image_processor import ImageProcessor, SUPPORTED_EXTENSIONS as IMAGE_EXTS
from src.processors.email_processor import EmailProcessor
from src.processors.text_processor import TextProcessor

logger = logging.getLogger(__name__)

_pdf_proc = PDFProcessor()
_email_proc = EmailProcessor()
_text_proc = TextProcessor()


def detect_document_type(filename: str) -> DocumentType:
    """Infer DocumentType from filename."""
    ext = Path(filename).suffix.lower()
    name_lower = filename.lower()

    if ext == ".pdf":
        if any(kw in name_lower for kw in ("transcript", "hearing")):
            return DocumentType.TRANSCRIPT
        if any(kw in name_lower for kw in ("police", "report")):
            return DocumentType.POLICE_REPORT
        if "imessage" in name_lower or "message" in name_lower:
            return DocumentType.IMESSAGE
        return DocumentType.PDF

    if ext in IMAGE_EXTS:
        return DocumentType.IMAGE

    if ext in (".eml", ".msg"):
        return DocumentType.EMAIL

    if ext in (".txt", ".csv", ".rtf", ".md"):
        return DocumentType.TEXT

    return DocumentType.UNKNOWN


def process_document(file_path: str | Path, doc_type: DocumentType | None = None) -> Dict[str, Any]:
    """Process a document file and return extracted content.

    Args:
        file_path: Path to the file on disk.
        doc_type: Optional override; auto-detected from filename if not given.

    Returns:
        {
            "text": str,
            "pages": list[dict],
            "metadata": dict,
            "doc_type": DocumentType,
        }
    """
    file_path = Path(file_path)
    if doc_type is None:
        doc_type = detect_document_type(file_path.name)

    ext = file_path.suffix.lower()

    # Choose processor based on actual file extension first, then semantic doc_type
    if ext == ".pdf":
        result = _pdf_proc.process(file_path)
    elif ext in IMAGE_EXTS:
        result = ImageProcessor().process(file_path)
    elif ext in (".eml", ".msg"):
        result = _email_proc.process(file_path)
    else:
        # Plain text, CSV, transcripts stored as TXT, etc.
        result = _text_proc.process(file_path)

    result["doc_type"] = doc_type
    return result
