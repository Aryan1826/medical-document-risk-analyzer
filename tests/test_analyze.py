"""
tests/test_analyze.py — Integration tests for the /api/analyze endpoint.
Run with:  pytest tests/ -v
"""

from __future__ import annotations
import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from app.main import app

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_dummy_image(text: str = "Dr. Test\nReg: MCI-2020-12345") -> bytes:
    """Create a simple white image with text rendered on it for testing."""
    img = Image.new("RGB", (800, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_analyze_returns_valid_structure():
    img_bytes = _make_dummy_image()
    response = client.post(
        "/api/analyze",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    assert "risk_score" in data
    assert "risk_level" in data
    assert "reasons" in data
    assert "parsed_fields" in data
    assert "doctor_verification" in data
    assert "tampering" in data

    assert isinstance(data["risk_score"], int)
    assert 0 <= data["risk_score"] <= 100
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(data["reasons"], list)


def test_analyze_rejects_wrong_mime():
    response = client.post(
        "/api/analyze",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_analyze_rejects_empty_file():
    response = client.post(
        "/api/analyze",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400


def test_list_fields():
    response = client.get("/api/analyze/fields")
    assert response.status_code == 200
    data = response.json()
    assert "fields" in data
    assert "doctor_name" in data["fields"]
    assert "registration_number" in data["fields"]


# ── Unit tests for individual modules ────────────────────────────────────────

def test_parser_extracts_doctor_name():
    from app.core.parser import parse_fields
    text = "Dr. Rajesh Mehta\nReg No: MCI-2019-48231\nDate: 01/01/2024"
    fields = parse_fields(text)
    assert fields["doctor_name"] is not None
    assert "Rajesh" in fields["doctor_name"] or "Mehta" in fields["doctor_name"]


def test_rules_catches_future_date():
    from app.core.rules import validate_rules
    fields = {
        "doctor_name": "Dr. Rajesh Mehta",
        "registration_number": "MCI-2019-48231",
        "patient_name": "Aryan Patel",
        "issue_date": "01/01/2099",
        "leave_days": 3,
    }
    violations = validate_rules(fields)
    assert any("future" in v.lower() for v in violations)


def test_rules_catches_missing_fields():
    from app.core.rules import validate_rules
    fields = {"doctor_name": None, "registration_number": None,
              "patient_name": None, "issue_date": None}
    violations = validate_rules(fields)
    assert len(violations) >= 4


def test_doctor_verify_known_doctor():
    from app.core.doctor_verify import verify_doctor
    result = verify_doctor("Dr. Rajesh Mehta", "MCI-2019-48231")
    assert result["status"] == "VERIFIED"
    assert result["confidence"] >= 0.85


def test_doctor_verify_unknown():
    from app.core.doctor_verify import verify_doctor
    result = verify_doctor("Dr. Unknown Person", "ZZZ-9999-00000")
    assert result["status"] == "UNVERIFIED"


def test_risk_scorer_high_for_bad_inputs():
    from app.core.risk_scorer import compute_risk_score
    violations = ["Missing: Doctor Name", "Missing: Registration Number",
                  "Issue date in future"]
    doctor = {"status": "UNVERIFIED", "message": "Not found", "confidence": 0.0}
    tampering = {"score": 0.8, "tampered": True,
                 "findings": ["High ELA variance"], "raw_scores": {}}
    fields = {}
    result = compute_risk_score(violations, doctor, tampering, fields)
    assert result["risk_level"] == "HIGH"
    assert result["risk_score"] > 60


def test_risk_scorer_low_for_clean_inputs():
    from app.core.risk_scorer import compute_risk_score
    result = compute_risk_score(
        rule_violations=[],
        doctor_status={"status": "VERIFIED", "message": "OK", "confidence": 0.95},
        tampering_result={"score": 0.0, "tampered": False, "findings": [], "raw_scores": {}},
        parsed_fields={},
    )
    assert result["risk_level"] == "LOW"
    assert result["risk_score"] <= 30
