"""
text_utils.py — Shared text cleaning helpers used by parser.py and rules.py.
"""

from __future__ import annotations
import re
from datetime import datetime
from typing import Optional


def clean_text(raw: str) -> str:
    """
    Normalize raw OCR output:
      - Strip non-printable characters
      - Collapse multiple spaces / newlines
      - Fix common OCR ligature errors
    """
    # Remove non-printable except newline
    text = re.sub(r"[^\x20-\x7E\n]", " ", raw)

    # Common OCR character confusions
    text = text.replace("l\n", "I\n")   # lowercase L at end of word
    text = text.replace(" l ", " I ")   # standalone lowercase L

    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalise_date(date_str: str) -> Optional[str]:
    """
    Parse a date string in any common format and return DD/MM/YYYY.
    Returns None if unparseable.
    """
    date_str = date_str.strip()
    formats = [
        "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d",
        "%d-%m-%Y", "%d.%m.%Y",
        "%d %B %Y", "%d %b %Y",
        "%B %d, %Y", "%b %d, %Y",
        "%d/%m/%y", "%m/%d/%y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%d/%m/%Y")
        except ValueError:
            continue
    return None


def extract_numbers(text: str) -> list[str]:
    """Return every contiguous digit sequence found in text."""
    return re.findall(r"\d+", text)
