"""
schemas.py — Pydantic models for request/response validation.
All data flowing in and out of the API is typed here.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DoctorVerification(BaseModel):
    status: str                        # VERIFIED | SUSPICIOUS | UNVERIFIED
    confidence: float                  # 0.0 – 1.0
    message: str
    matched_doctor: Optional[Dict[str, Any]] = None


class TamperingResult(BaseModel):
    score: float                       # 0.0 (clean) – 1.0 (heavily tampered)
    tampered: bool
    findings: List[str]
    raw_scores: Dict[str, float]


class AnalyzeResponse(BaseModel):
    risk_score: int                    # 0 – 100
    risk_level: RiskLevel
    reasons: List[str]
    parsed_fields: Dict[str, Any]
    doctor_verification: Dict[str, Any]
    tampering: Dict[str, Any]
