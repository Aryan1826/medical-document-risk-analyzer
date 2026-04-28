"""
rules.py — Step 3 of the pipeline.
Validates extracted fields against hard business rules.
Each rule is a standalone function returning None (pass) or a string (fail reason).
Add new rules without touching anything else.
"""

from __future__ import annotations
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional

# Registration numbers must follow: 2-4 uppercase letters, dash, 4-digit year, dash, 4-6 digits
# Examples: MCI-2019-48231  |  DMC-2015-0042  |  KMC/2020/123456
REG_NUMBER_RE = re.compile(r"^[A-Z]{2,4}[-\/]\d{4}[-\/]\d{4,6}$")

MANDATORY_FIELDS = [
    "doctor_name",
    "registration_number",
    "patient_name",
    "issue_date",
]

DATE_FORMATS = [
    "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d",
    "%d-%m-%Y", "%d.%m.%Y",
    "%d %B %Y", "%d %b %Y",
]


def _parse_date(date_str: str) -> Optional[date]:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


# ── Individual rule functions ─────────────────────────────────────────────────

def _rule_mandatory_fields(fields: Dict[str, Any]) -> List[str]:
    violations = []
    for f in MANDATORY_FIELDS:
        if not fields.get(f):
            label = f.replace("_", " ").title()
            violations.append(f"Missing mandatory field: {label}")
    return violations


def _rule_registration_format(fields: Dict[str, Any]) -> Optional[str]:
    reg = fields.get("registration_number")
    if reg and not REG_NUMBER_RE.match(str(reg).upper().strip()):
        return f"Registration number has unexpected format: '{reg}'"
    return None


def _rule_date_not_future(fields: Dict[str, Any]) -> Optional[str]:
    raw = fields.get("issue_date")
    if not raw:
        return None
    parsed = _parse_date(str(raw))
    if parsed and parsed > date.today():
        return f"Issue date is in the future: {raw}"
    return None


def _rule_date_not_too_old(fields: Dict[str, Any]) -> Optional[str]:
    raw = fields.get("issue_date")
    if not raw:
        return None
    parsed = _parse_date(str(raw))
    if parsed and (date.today() - parsed).days > 365:
        return f"Document is older than 1 year: {raw}"
    return None


def _rule_leave_duration(fields: Dict[str, Any]) -> Optional[str]:
    leave = fields.get("leave_days")
    if leave is None:
        return None
    if leave < 1:
        return "Leave duration is less than 1 day"
    if leave > 30:
        return f"Suspicious leave duration: {leave} days (threshold is 30)"
    return None


def _rule_doctor_name_length(fields: Dict[str, Any]) -> Optional[str]:
    name = fields.get("doctor_name") or ""
    # A real doctor's name should have at least two words
    if name and len(name.split()) < 2:
        return f"Doctor name appears incomplete: '{name}'"
    return None


# ── Main validation function ──────────────────────────────────────────────────

def validate_rules(fields: Dict[str, Any]) -> List[str]:
    """
    Run all rules against the parsed fields dict.

    Returns:
        List of violation strings. Empty list means all rules passed.
    """
    violations: List[str] = []

    violations.extend(_rule_mandatory_fields(fields))

    for rule_fn in [
        _rule_registration_format,
        _rule_date_not_future,
        _rule_date_not_too_old,
        _rule_leave_duration,
        _rule_doctor_name_length,
    ]:
        result = rule_fn(fields)
        if result:
            violations.append(result)

    return violations
