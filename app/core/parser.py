"""
parser.py — Step 2 of the pipeline.
Extracts structured fields from raw OCR text using regex patterns.
RapidFuzz handles noisy OCR output where exact patterns miss.
"""

from __future__ import annotations
import re
from typing import Any, Dict, Optional
from app.utils.text_utils import clean_text, normalise_date


# Each field has a prioritised list of regex patterns.
# The first match wins.
FIELD_PATTERNS: Dict[str, list[str]] = {
    "doctor_name": [
        r"Dr\.?\s+([A-Za-z][A-Za-z\s\.]{2,40})",
        r"Doctor\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]{2,40})",
        r"Physician\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]{2,40})",
        r"Consultant\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]{2,40})",
    ],
    "registration_number": [
        r"Reg(?:istration)?\.?\s*No\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]{3,20})",
        r"MCI\s*[-\/]?\s*(\d{4}[-\/]\d{4,6})",
        r"Reg\s*[:\-]?\s*([A-Z]{2,4}[-\/]\d{4}[-\/]\d{4,6})",
        r"([A-Z]{2,4}[-\/]\d{4}[-\/]\d{4,6})",
        r"License\s*No\.?\s*[:\-]?\s*([A-Z0-9\-\/]{5,20})",
    ],
    "hospital_name": [
        r"([A-Za-z][A-Za-z\s]{2,40}(?:Hospital|Clinic|Medical|Centre|Center|Health|Care))",
        r"(?:Hospital|Clinic)\s*[:\-]?\s*([A-Za-z][A-Za-z\s]{2,40})",
    ],
    "patient_name": [
        r"Patient\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]{2,40})",
        r"Patient\s*Name\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]{2,40})",
        r"Name\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]{2,40})",
        r"(?:Mr|Mrs|Ms|Miss)\.?\s+([A-Za-z][A-Za-z\s\.]{1,40})",
    ],
    "issue_date": [
        r"Date\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"Dated\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})",
        r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})",
    ],
    "leave_days": [
        r"(\d+)\s*days?\s*(?:rest|leave|off|bed\s*rest|complete\s*rest)",
        r"rest\s*for\s*(\d+)\s*days?",
        r"leave\s*of\s*(\d+)\s*days?",
        r"advised\s*(\d+)\s*days?",
        r"(\d+)\s*days?\s*medical\s*leave",
    ],
    "diagnosis": [
        r"Diagnosis\s*[:\-]?\s*([^\n]{3,80})",
        r"Diagnosed\s*[:\-]?\s*([^\n]{3,80})",
        r"Complaint\s*[:\-]?\s*([^\n]{3,80})",
        r"Condition\s*[:\-]?\s*([^\n]{3,80})",
        r"Suffering\s*from\s*[:\-]?\s*([^\n]{3,80})",
    ],
}


def _try_patterns(text: str, patterns: list[str]) -> Optional[str]:
    """Try each pattern in order; return first match group 1, or None."""
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def parse_fields(raw_text: str) -> Dict[str, Any]:
    """
    Args:
        raw_text: cleaned OCR output string.

    Returns:
        Dictionary with keys matching FIELD_PATTERNS.
        Missing fields are None.
    """
    text = clean_text(raw_text)
    fields: Dict[str, Any] = {}

    for field, patterns in FIELD_PATTERNS.items():
        value = _try_patterns(text, patterns)

        # Post-process specific fields
        if value is not None:
            if field == "issue_date":
                value = normalise_date(value) or value
            elif field == "leave_days":
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    value = None
            elif field in ("doctor_name", "patient_name", "hospital_name"):
                # Strip trailing punctuation / extra words
                value = re.sub(r"[,\.\-:]+$", "", value).strip()
                # Reject implausibly short values
                if len(value) < 3:
                    value = None

        fields[field] = value

    return fields
