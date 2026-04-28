"""
doctor_verify.py — Step 4 of the pipeline.
Looks up extracted doctor name + registration number in doctors.json.
RapidFuzz fuzzy matching handles OCR noise and spelling variations.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

# Path relative to project root (works whether run from root or inside app/)
_DB_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "doctors.json"),
    os.path.join("data", "doctors.json"),
]


def _load_doctors() -> List[Dict[str, Any]]:
    for path in _DB_CANDIDATES:
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                print(f"[DoctorVerify] Failed to load DB at {abs_path}: {exc}")
    print("[DoctorVerify] Warning: doctors.json not found. All doctors will be UNVERIFIED.")
    return []


def verify_doctor(
    doctor_name: Optional[str],
    registration_number: Optional[str],
) -> Dict[str, Any]:
    """
    Args:
        doctor_name:          Extracted from parser (may be None / noisy).
        registration_number:  Extracted from parser (may be None / noisy).

    Returns:
        {
            status:          VERIFIED | SUSPICIOUS | UNVERIFIED
            confidence:      0.0 – 1.0
            message:         Human-readable explanation
            matched_doctor:  Full doctor record or None
        }
    """
    doctors = _load_doctors()

    if not doctor_name and not registration_number:
        return {
            "status": "UNVERIFIED",
            "confidence": 0.0,
            "message": "No doctor name or registration number found in document",
            "matched_doctor": None,
        }

    best_doctor: Optional[Dict[str, Any]] = None
    best_name_score: float = 0.0
    best_reg_match: bool = False

    for doctor in doctors:
        if not doctor.get("active", True):
            continue

        # ── Name similarity (fuzzy) ───────────────────────────────────
        name_score: float = 0.0
        if doctor_name:
            name_score = fuzz.token_sort_ratio(
                doctor_name.lower().strip(),
                doctor["name"].lower().strip(),
            )

        # ── Registration number (exact, case-insensitive) ─────────────
        reg_match = False
        if registration_number and doctor.get("registration_number"):
            reg_match = (
                registration_number.upper().strip()
                == doctor["registration_number"].upper().strip()
            )

        # ── Combined ranking ──────────────────────────────────────────
        # Exact reg match is a strong signal → boost score by 40
        combined = name_score + (40 if reg_match else 0)

        if combined > best_name_score + (40 if best_reg_match else 0):
            best_name_score = name_score
            best_doctor = doctor
            best_reg_match = reg_match

    # ── Decision ─────────────────────────────────────────────────────────────
    if best_doctor is None or best_name_score < 50:
        return {
            "status": "UNVERIFIED",
            "confidence": 0.0,
            "message": "Doctor not found in verified database",
            "matched_doctor": None,
        }

    if best_name_score >= 85 and best_reg_match:
        return {
            "status": "VERIFIED",
            "confidence": round(best_name_score / 100, 2),
            "message": (
                f"Verified: {best_doctor['name']} "
                f"({best_doctor['registration_number']}) — "
                f"{best_doctor.get('hospital', 'Unknown hospital')}"
            ),
            "matched_doctor": best_doctor,
        }

    if best_name_score >= 70:
        reason = (
            "registration number mismatch"
            if not best_reg_match
            else "name is a close but not exact match"
        )
        return {
            "status": "SUSPICIOUS",
            "confidence": round(best_name_score / 100, 2),
            "message": (
                f"Partial match: {best_doctor['name']} — {reason}"
            ),
            "matched_doctor": best_doctor,
        }

    return {
        "status": "UNVERIFIED",
        "confidence": round(best_name_score / 100, 2),
        "message": "Doctor not found in verified database",
        "matched_doctor": None,
    }
