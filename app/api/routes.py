"""
routes.py — API endpoint definitions.
Defines POST /api/analyze — the single entry point for document analysis.
Orchestrates the 6-step pipeline and returns the structured response.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.ocr import extract_text
from app.core.parser import parse_fields
from app.core.rules import validate_rules
from app.core.doctor_verify import verify_doctor
from app.core.tampering import analyze_tampering
from app.core.risk_scorer import compute_risk_score
from app.models.schemas import AnalyzeResponse

router = APIRouter()

ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "image/webp",
    "image/heic",
    "image/heif",
}

MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a medical document for risk",
    description=(
        "Upload a medical prescription image (JPEG / PNG). "
        "The system extracts text via OCR, validates fields, "
        "verifies the doctor, detects image tampering, "
        "and returns a 0–100 risk score with reasons."
    ),
)
async def analyze_document(
    file: UploadFile = File(..., description="Medical document image (JPEG or PNG)"),
) -> AnalyzeResponse:
    # ── Input validation ──────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: {file.content_type}. "
                f"Accepted: {', '.join(ALLOWED_TYPES)}"
            ),
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(image_bytes) // 1024} KB). Max 10 MB.",
        )

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    raw_text = extract_text(image_bytes)
    fields = parse_fields(raw_text)
    rule_violations = validate_rules(fields)
    doctor_status = verify_doctor(
        fields.get("doctor_name"),
        fields.get("registration_number"),
    )
    tampering_result = analyze_tampering(image_bytes)
    result = compute_risk_score(rule_violations, doctor_status, tampering_result, fields)

    # ── Response ──────────────────────────────────────────────────────────────
    return AnalyzeResponse(
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        reasons=result["reasons"],
        parsed_fields=fields,
        doctor_verification=doctor_status,
        tampering=tampering_result,
    )


@router.get(
    "/analyze/fields",
    summary="List all fields the parser can extract",
)
def list_fields() -> dict:
    """Returns the list of fields the OCR parser attempts to extract."""
    return {
        "fields": [
            "doctor_name",
            "registration_number",
            "hospital_name",
            "patient_name",
            "issue_date",
            "leave_days",
            "diagnosis",
        ]
    }
