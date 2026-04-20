"""Email document processor.

Parses .eml files (RFC 2822) and extracts structured fields plus body text.
Falls back to raw text reading when mail-parser is unavailable.
"""

from __future__ import annotations

import email
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class EmailProcessor:
    """Extract text and metadata from email files (.eml)."""

    def process(self, file_path: str | Path) -> Dict[str, Any]:
        """Parse an email file.

        Returns:
            {
                "text": combined headers + body text,
                "pages": [{"page_number": 1, "text": str}],
                "metadata": {from, to, cc, subject, date, attachments},
            }
        """
        file_path = Path(file_path)

        # Try mail-parser first (richer API)
        try:
            return self._mailparser_parse(file_path)
        except Exception as exc:
            logger.warning("mail-parser failed (%s), falling back to stdlib", exc)

        # Stdlib email fallback
        return self._stdlib_parse(file_path)

    @staticmethod
    def _mailparser_parse(file_path: Path) -> Dict[str, Any]:
        import mailparser

        mail = mailparser.parse_from_file(str(file_path))
        body = (mail.body or "").strip()
        metadata: Dict[str, Any] = {
            "from": mail.from_,
            "to": mail.to,
            "cc": mail.cc,
            "subject": mail.subject,
            "date": str(mail.date) if mail.date else None,
            "attachments": [a.get("filename") for a in (mail.attachments or [])],
            "source": file_path.name,
        }
        header_text = (
            f"From: {metadata['from']}\n"
            f"To: {metadata['to']}\n"
            f"CC: {metadata['cc']}\n"
            f"Subject: {metadata['subject']}\n"
            f"Date: {metadata['date']}\n"
        )
        full_text = f"{header_text}\n{body}"
        return {
            "text": full_text,
            "pages": [{"page_number": 1, "text": full_text}],
            "metadata": metadata,
        }

    @staticmethod
    def _stdlib_parse(file_path: Path) -> Dict[str, Any]:
        raw = file_path.read_bytes()
        msg = email.message_from_bytes(raw)

        # Extract body
        body_parts: List[str] = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    charset = part.get_content_charset() or "utf-8"
                    body_parts.append(part.get_payload(decode=True).decode(charset, errors="replace"))
        else:
            charset = msg.get_content_charset() or "utf-8"
            payload = msg.get_payload(decode=True)
            if payload:
                body_parts.append(payload.decode(charset, errors="replace"))

        body = "\n".join(body_parts).strip()
        metadata: Dict[str, Any] = {
            "from": msg.get("From", ""),
            "to": msg.get("To", ""),
            "cc": msg.get("Cc", ""),
            "subject": msg.get("Subject", ""),
            "date": msg.get("Date", ""),
            "attachments": [],
            "source": file_path.name,
        }
        header_text = (
            f"From: {metadata['from']}\n"
            f"To: {metadata['to']}\n"
            f"CC: {metadata['cc']}\n"
            f"Subject: {metadata['subject']}\n"
            f"Date: {metadata['date']}\n"
        )
        full_text = f"{header_text}\n{body}"
        return {
            "text": full_text,
            "pages": [{"page_number": 1, "text": full_text}],
            "metadata": metadata,
        }
