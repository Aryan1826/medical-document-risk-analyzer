"""
risk_scorer.py — Step 6 of the pipeline (final step).
Collects all findings from rules, doctor_verify, and tampering,
applies a weighted formula, and outputs the final risk score + level.

Score breakdown (max 100):
  Tampering detected        → up to 40 pts
  Doctor UNVERIFIED         → 25 pts
  Doctor SUSPICIOUS         → 10 pts
  Each rule violation       →  5 pts  (max 6 violations counted)
  Missing mandatory field   →  5 pts  (already included in rule violations)
"""

from __future__ import annotations
from typing import Any, Dict, List


def compute_risk_score(
    rule_violations: List[str],
    doctor_status: Dict[str, Any],
    tampering_result: Dict[str, Any],
    parsed_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Args:
        rule_violations:  List of violation strings from rules.validate_rules()
        doctor_status:    Dict from doctor_verify.verify_doctor()
        tampering_result: Dict from tampering.analyze_tampering()
        parsed_fields:    Dict from parser.parse_fields() (used for context)

    Returns:
        {
            risk_score: int (0–100),
            risk_level: str (LOW | MEDIUM | HIGH),
            reasons:    list of explanation strings
        }
    """
    score = 0
    reasons: List[str] = []

    # ── 1. Tampering (max 40 pts) ─────────────────────────────────────────────
    tampering_score = tampering_result.get("score", 0.0)
    tampering_pts = int(tampering_score * 40)
    if tampering_pts > 0:
        score += tampering_pts
        for finding in tampering_result.get("findings", []):
            reasons.append(f"[TAMPERING] {finding}")

    # ── 2. Doctor verification (max 25 pts) ───────────────────────────────────
    d_status = doctor_status.get("status", "UNVERIFIED")
    d_message = doctor_status.get("message", "")

    if d_status == "UNVERIFIED":
        score += 25
        reasons.append(f"[DOCTOR] {d_message}")
    elif d_status == "SUSPICIOUS":
        score += 10
        reasons.append(f"[DOCTOR] {d_message}")
    # VERIFIED → 0 pts added

    # ── 3. Rule violations (5 pts each, max 6 counted = 30 pts) ──────────────
    for violation in rule_violations[:6]:
        score += 5
        reasons.append(f"[RULE] {violation}")

    # ── Cap and classify ──────────────────────────────────────────────────────
    score = min(score, 100)

    if score <= 30:
        risk_level = "LOW"
    elif score <= 60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    if not reasons:
        reasons.append("No significant risk factors detected — document appears legitimate")

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "reasons": reasons,
    }
