"""
routes.py — API endpoint definitions.
Defines POST /api/analyze — the single entry point for document analysis.
Orchestrates the 6-step pipeline and returns the structured response.

Supported formats: JPEG, PNG, TIFF, WebP, HEIC/HEIF, PDF
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
from app.utils.image_utils import is_pdf, pdf_to_images

router = APIRouter()

ALLOWED_TYPES = {
    # Images
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "image/webp",
    "image/heic",
    "image/heif",
    # PDF
    "application/pdf",
    # Some browsers/OS send PDF with this MIME type
    "application/octet-stream",
}

MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB (PDFs can be larger)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a medical document for risk",
    description=(
        "Upload a medical prescription or document "
        "(JPEG, PNG, TIFF, WebP, HEIC, or PDF). "
        "The system extracts text via OCR, validates fields, "
        "verifies the doctor, detects image tampering, "
        "and returns a 0–100 risk score with reasons. "
        "For multi-page PDFs, OCR runs on all pages and "
        "tampering analysis runs on the first page."
    ),
)
async def analyze_document(
    file: UploadFile = File(..., description="Medical document (image or PDF)"),
) -> AnalyzeResponse:
    # ── Input validation ──────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{file.content_type}'. "
                "Accepted formats: JPEG, PNG, TIFF, WebP, HEIC, PDF."
            ),
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes) // 1024} KB). Max 20 MB.",
        )

    # ── PDF handling ──────────────────────────────────────────────────────────
    # Detect PDF by magic bytes (%PDF) — catches even misreported MIME types
    if is_pdf(file_bytes):
        try:
            page_images = pdf_to_images(file_bytes)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        if not page_images:
            raise HTTPException(
                status_code=422,
                detail="PDF appears to have no pages or could not be rendered.",
            )

        # OCR all pages and join their text — important for multi-page reports
        all_text_parts: list[str] = []
        for page_bytes in page_images:
            page_text = extract_text(page_bytes)
            if page_text.strip():
                all_text_parts.append(page_text)
        raw_text = "\n".join(all_text_parts)

        # Tampering check on page 1 only (where doctor signature/header lives)
        tampering_result = analyze_tampering(page_images[0])

    # ── Image handling ────────────────────────────────────────────────────────
    else:
        raw_text = extract_text(file_bytes)
        tampering_result = analyze_tampering(file_bytes)

    # ── Shared pipeline (same for both PDF and image) ─────────────────────────
    fields = parse_fields(raw_text)
    rule_violations = validate_rules(fields)
    doctor_status = verify_doctor(
        fields.get("doctor_name"),
        fields.get("registration_number"),
    )
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
