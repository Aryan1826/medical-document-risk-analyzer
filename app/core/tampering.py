"""
tampering.py — Step 5 of the pipeline.
Detects image manipulation using three OpenCV-based checks:
  1. ELA  — Error Level Analysis (copy-paste / region edits)
  2. Blur — Laplacian variance (artificially softened edits)
  3. Noise uniformity — tile-wise std deviation (composite images)

No ML required — all classical computer vision.
"""

from __future__ import annotations
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.utils.image_utils import load_image


# ── Check 1: Error Level Analysis ────────────────────────────────────────────

def _ela_check(image: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    Re-compress the image at low quality and diff it against the original.
    Edited regions (copy-paste, text replacement) survive compression differently
    and show up as bright spots in the diff.

    Returns: (ela_mean, finding_string or None)
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)

    try:
        cv2.imwrite(tmp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        recompressed = cv2.imread(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if recompressed is None:
        return 0.0, None

    # Match dimensions (safety)
    if image.shape != recompressed.shape:
        recompressed = cv2.resize(
            recompressed,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    diff = cv2.absdiff(image, recompressed).astype(np.float32)
    ela_mean = float(np.mean(diff))

    finding = None
    if ela_mean > 12.0:
        finding = (
            f"High ELA variance ({ela_mean:.1f}) — "
            "one or more regions may have been digitally edited"
        )
    return ela_mean, finding


# ── Check 2: Blur / sharpness ─────────────────────────────────────────────────

def _blur_check(image: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    Laplacian variance measures overall sharpness.
    Legitimate scanned prescriptions have variance > 50.
    Artificially blurred images (hiding edited text) are much lower.

    Returns: (laplacian_variance, finding_string or None)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    finding = None
    if lap_var < 40.0:
        finding = (
            f"Very low image sharpness ({lap_var:.1f}) — "
            "document may have been blurred to hide alterations"
        )
    return lap_var, finding


# ── Check 3: Noise uniformity ─────────────────────────────────────────────────

def _noise_check(image: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    Real scans have roughly uniform noise across the page.
    Composite images (different source regions pasted together) show
    wildly different noise levels in different tiles.

    Returns: (noise_std, finding_string or None)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    th, tw = max(h // 4, 1), max(w // 4, 1)

    tile_stds: List[float] = []
    for i in range(4):
        for j in range(4):
            tile = gray[i * th: (i + 1) * th, j * tw: (j + 1) * tw]
            if tile.size > 0:
                tile_stds.append(float(np.std(tile)))

    if not tile_stds:
        return 0.0, None

    noise_std = float(np.std(tile_stds))

    finding = None
    if noise_std > 28.0:
        finding = (
            f"Inconsistent noise distribution across image ({noise_std:.1f}) — "
            "possible composite document (regions from different sources)"
        )
    return noise_std, finding


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze_tampering(image_bytes: bytes) -> Dict[str, Any]:
    """
    Run all three tampering checks and aggregate into a single score.

    Returns:
        {
            score:      float 0.0–1.0  (higher = more suspicious)
            tampered:   bool
            findings:   list of human-readable flag strings
            raw_scores: dict of raw metric values per check
        }
    """
    image = load_image(image_bytes)

    ela_val, ela_finding = _ela_check(image)
    blur_val, blur_finding = _blur_check(image)
    noise_val, noise_finding = _noise_check(image)

    findings: List[str] = []
    if ela_finding:
        findings.append(ela_finding)
    if blur_finding:
        findings.append(blur_finding)
    if noise_finding:
        findings.append(noise_finding)

    # Weighted tampering score (0.0 → 1.0)
    score = 0.0
    if ela_val > 12.0:
        score += 0.40
    if blur_val < 40.0:
        score += 0.35
    if noise_val > 28.0:
        score += 0.25

    score = round(min(score, 1.0), 2)

    return {
        "score": score,
        "tampered": score >= 0.40,
        "findings": findings,
        "raw_scores": {
            "ela_mean": round(ela_val, 2),
            "blur_laplacian_variance": round(blur_val, 2),
            "noise_tile_std": round(noise_val, 2),
        },
    }
