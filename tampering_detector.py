#!/usr/bin/env python3
"""
tampering_detector.py — Multi-Feature Medical Document Tampering Detector
==========================================================================

5 detection methods (no deep learning, pure OpenCV + NumPy):

  Method 1 ─ ELA         : Re-compress at low quality, diff reveals edited pixels
  Method 2 ─ Blur        : Block-wise sharpness — tampered blocks stand out
  Method 3 ─ Edge        : Block-wise edge density — copy-paste breaks uniformity
  Method 4 ─ Noise       : Median-filter residual — composite images expose themselves
  Method 5 ─ Text        : Text blob size/shape outliers — different font = different stats

Supports: JPEG, PNG, TIFF, WebP, HEIC, PDF

Usage:
    # Analyze one file
    python tampering_detector.py prescription.jpg
    python tampering_detector.py report.pdf

    # Batch test on your real + fake dataset
    python tampering_detector.py --batch dataset/real/ dataset/fake/

Output:
    → Prints structured JSON to console
    → Saves  <filename>_analysis.png  (6-panel visual report)
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Use Agg backend so it works even without a screen / display server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ═══════════════════════════════════════════════════════════════════════════════
#  TUNING CONSTANTS
#  ─ Adjust these if you want more / less sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

GRID          = 6      # Image divided into GRID × GRID blocks (6×6 = 36 blocks)
ELA_QUALITY   = 75     # JPEG quality used for ELA re-compression
MAD_FACTOR    = 2.0    # A block is "suspicious" if it is MAD_FACTOR × MAD from median
TEXT_Z_THRESH = 2.5    # A text blob is "outlier" if its z-score exceeds this

# How much each method contributes to the final score (must sum to 1.0)
WEIGHTS = {
    "ela":   0.30,   # Most reliable signal for digital edits
    "blur":  0.20,   # Good for detecting region-level edits
    "edge":  0.20,   # Good for detecting copy-paste
    "noise": 0.20,   # Good for detecting composite images
    "text":  0.10,   # Catches font-change tampering
}

# Final risk bands  [low, high)
RISK_BANDS = [
    (0.00, 0.30, "LOW",    "Document appears genuine — no strong tampering signals"),
    (0.30, 0.55, "MEDIUM", "Some anomalies detected — manual review recommended"),
    (0.55, 1.01, "HIGH",   "Strong tampering signals — document is likely forged"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 0 ─ FILE LOADING
#  Handles JPEG/PNG/TIFF/WebP/HEIC and PDF (first page)
# ═══════════════════════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    """
    Load any supported file and return an OpenCV BGR image array.

    Supported formats:
      - JPEG, PNG, TIFF, WebP  → loaded directly with cv2.imread
      - HEIC / HEIF            → loaded via pillow-heif + PIL
      - PDF                    → first page rendered at 144 DPI via PyMuPDF
    """
    path = str(path)
    ext  = Path(path).suffix.lower()

    # ── PDF ──────────────────────────────────────────────────────────────────
    if ext == ".pdf" or _magic_bytes_is_pdf(path):
        return _load_pdf_first_page(path)

    # ── HEIC / HEIF ──────────────────────────────────────────────────────────
    if ext in (".heic", ".heif"):
        return _load_heic(path)

    # ── Standard image (JPEG / PNG / TIFF / WebP) ────────────────────────────
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"[ERROR] Cannot load file: {path}")
    return img


def _magic_bytes_is_pdf(path: str) -> bool:
    """Check first 4 bytes — PDFs always start with %PDF."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except OSError:
        return False


def _load_pdf_first_page(path: str) -> np.ndarray:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        sys.exit("[ERROR] PDF support requires pymupdf — run:  pip install pymupdf")

    doc  = fitz.open(path)
    page = doc[0]
    # 2× zoom → ~144 DPI, good balance of speed and OCR clarity
    pix  = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
    doc.close()

    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _load_heic(path: str) -> np.ndarray:
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        sys.exit("[ERROR] HEIC support requires pillow-heif — run:  pip install pillow-heif")

    from PIL import Image
    pil = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 ─ ELA  (Error Level Analysis)
#
#  CONCEPT:
#    JPEG compression is lossy — it throws away fine detail to save space.
#    When you edit part of a JPEG and save it again, that edited part gets
#    compressed a SECOND time.  The rest of the image only compressed once.
#    Re-compressing the whole image at low quality and subtracting from the
#    original makes doubly-compressed (edited) regions show up as bright spots.
# ═══════════════════════════════════════════════════════════════════════════════

def ela_check(image: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Returns:
        score     : 0.0–1.0  (higher = more suspicious)
        ela_vis   : amplified difference image  (for visualisation)
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)

    try:
        cv2.imwrite(tmp_path, image, [cv2.IMWRITE_JPEG_QUALITY, ELA_QUALITY])
        recomp = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
    finally:
        os.path.exists(tmp_path) and os.unlink(tmp_path)

    if recomp is None:
        return 0.0, np.zeros_like(image)

    if image.shape != recomp.shape:
        recomp = cv2.resize(recomp, (image.shape[1], image.shape[0]))

    diff     = cv2.absdiff(image, recomp).astype(np.float32)
    ela_mean = float(np.mean(diff))

    # Amplify ×10 so small differences are visible in the output image
    ela_vis  = np.clip(diff * 10, 0, 255).astype(np.uint8)

    # Normalise: ELA_QUALITY=75 gives ~6.0 on clean images, ~20+ on edited
    score    = min(ela_mean / 24.0, 1.0)

    return round(score, 4), ela_vis


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 ─ BLUR CONSISTENCY  (block-wise Laplacian variance)
#
#  CONCEPT:
#    The Laplacian filter measures how much the brightness changes between
#    neighbouring pixels — sharp edges give high values, blurry areas give low.
#    A genuine scanned document has roughly uniform sharpness everywhere.
#    A tampered document often has one region noticeably sharper or blurrier
#    than the rest (e.g. a pasted-in text block photographed at different focus).
# ═══════════════════════════════════════════════════════════════════════════════

def blur_consistency_check(image: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Returns:
        score        : 0.0–1.0
        block_flags  : GRID×GRID float array  (1.0 = suspicious block, 0.0 = clean)
    """
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    bh, bw = max(h // GRID, 1), max(w // GRID, 1)

    # Compute Laplacian variance for every block
    lap_vars = np.zeros((GRID, GRID), dtype=np.float32)
    for i in range(GRID):
        for j in range(GRID):
            block = gray[i*bh : (i+1)*bh, j*bw : (j+1)*bw]
            if block.size > 0:
                lap_vars[i, j] = float(cv2.Laplacian(block, cv2.CV_64F).var())

    # MAD = Median Absolute Deviation — more robust than std for outlier detection
    median  = np.median(lap_vars)
    mad     = np.median(np.abs(lap_vars - median)) + 1e-6
    z_score = np.abs(lap_vars - median) / mad

    block_flags = (z_score > MAD_FACTOR).astype(np.float32)
    score       = float(np.mean(block_flags))   # fraction of suspicious blocks

    return round(score, 4), block_flags


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 ─ EDGE CONSISTENCY  (block-wise Canny edge density)
#
#  CONCEPT:
#    The Canny algorithm finds the outlines of objects in an image.
#    Text on paper produces a predictable density of edges.
#    When content from a different source is pasted in, the edge density
#    in that block is often very different from the surrounding blocks.
# ═══════════════════════════════════════════════════════════════════════════════

def edge_consistency_check(image: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Returns:
        score        : 0.0–1.0
        block_flags  : GRID×GRID float array
    """
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    h, w   = edges.shape
    bh, bw = max(h // GRID, 1), max(w // GRID, 1)

    densities = np.zeros((GRID, GRID), dtype=np.float32)
    for i in range(GRID):
        for j in range(GRID):
            block = edges[i*bh : (i+1)*bh, j*bw : (j+1)*bw]
            if block.size > 0:
                densities[i, j] = float(np.mean(block > 0))

    median  = np.median(densities)
    mad     = np.median(np.abs(densities - median)) + 1e-6
    z_score = np.abs(densities - median) / mad

    block_flags = (z_score > MAD_FACTOR).astype(np.float32)
    score       = float(np.mean(block_flags))

    return round(score, 4), block_flags


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 ─ NOISE MAP ANALYSIS  (median filter residual)
#
#  CONCEPT:
#    Every camera sensor introduces a tiny random grain called "noise".
#    A median blur (size 5) removes the noise but keeps the real content.
#    Subtracting: noise_map = |original – median_blurred|  isolates the noise.
#    On a real photo, noise is uniform everywhere.
#    On a composite image (paste from different photos), each region has
#    its own noise signature — the std of block-level noise reveals this.
# ═══════════════════════════════════════════════════════════════════════════════

def noise_check(image: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Returns:
        score        : 0.0–1.0
        noise_vis    : per-pixel noise magnitude image  (for visualisation)
        block_flags  : GRID×GRID float array
    """
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smoothed = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
    noise    = np.abs(gray - smoothed)   # residual = pure noise

    h, w   = noise.shape
    bh, bw = max(h // GRID, 1), max(w // GRID, 1)

    block_noise = np.zeros((GRID, GRID), dtype=np.float32)
    for i in range(GRID):
        for j in range(GRID):
            block = noise[i*bh : (i+1)*bh, j*bw : (j+1)*bw]
            if block.size > 0:
                block_noise[i, j] = float(np.mean(block))

    median  = np.median(block_noise)
    mad     = np.median(np.abs(block_noise - median)) + 1e-6
    z_score = np.abs(block_noise - median) / mad

    block_flags = (z_score > MAD_FACTOR).astype(np.float32)
    score       = float(np.mean(block_flags))

    # Amplify ×3 for visibility in the output image
    noise_vis = np.clip(noise * 3, 0, 255).astype(np.uint8)

    return round(score, 4), noise_vis, block_flags


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 ─ TEXT BLOB IRREGULARITY  (morphological + contour analysis)
#
#  CONCEPT:
#    Real prescriptions use one font / one size throughout.
#    Forgers often paste in text from a different source — different font size,
#    weight, or pixel density.  We find all "text-shaped blobs" using
#    morphological operations, then flag any blob whose area or shape is a
#    statistical outlier compared to the rest of the document.
# ═══════════════════════════════════════════════════════════════════════════════

def text_irregularity_check(
    image: np.ndarray,
) -> tuple[float, list[tuple[int, int, int, int]]]:
    """
    Returns:
        score         : 0.0–1.0
        outlier_boxes : list of (x, y, w, h) bounding boxes of suspicious text blobs
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu threshold: automatically finds the best dark/light split
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Connect nearby characters into word-level blobs
    # kernel width=15 merges characters in a word; height=3 keeps lines separate
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    text_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0, []

    # Filter: keep only blobs that are plausibly text-sized
    h_img, w_img  = image.shape[:2]
    min_area = max((h_img * w_img) * 0.0003, 50)   # at least 0.03% of image
    max_area = (h_img * w_img) * 0.08              # at most  8%  of image

    valid_boxes = []
    valid_areas = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if min_area <= area <= max_area:
            valid_boxes.append((x, y, bw, bh))
            valid_areas.append(float(area))

    if len(valid_areas) < 4:
        # Not enough text blobs to do meaningful statistics
        return 0.0, []

    areas  = np.array(valid_areas, dtype=np.float32)
    median = np.median(areas)
    mad    = np.median(np.abs(areas - median)) + 1e-6
    z      = np.abs(areas - median) / mad

    outlier_boxes = [b for b, zi in zip(valid_boxes, z) if zi > TEXT_Z_THRESH]
    score         = len(outlier_boxes) / max(len(valid_boxes), 1)

    return round(min(score, 1.0), 4), outlier_boxes


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6 ─ COMBINE ALL SCORES INTO FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_final_score(
    ela:   float,
    blur:  float,
    edge:  float,
    noise: float,
    text:  float,
) -> tuple[float, str, list[str]]:
    """
    Weighted sum → 0.0–1.0 score → LOW / MEDIUM / HIGH risk level.

    Returns:
        final_score : float 0.0–1.0
        risk_level  : "LOW" | "MEDIUM" | "HIGH"
        reasons     : human-readable list of triggered signals
    """
    final_score = (
        ela   * WEIGHTS["ela"]   +
        blur  * WEIGHTS["blur"]  +
        edge  * WEIGHTS["edge"]  +
        noise * WEIGHTS["noise"] +
        text  * WEIGHTS["text"]
    )
    final_score = round(min(final_score, 1.0), 4)

    # Build reason list — only mention signals that crossed their threshold
    reasons: list[str] = []
    if ela   > 0.25: reasons.append(f"ELA anomaly detected (score={ela:.2f}) — possible digital re-editing")
    if blur  > 0.20: reasons.append(f"Blur inconsistency across {int(blur*GRID*GRID)} blocks (score={blur:.2f})")
    if edge  > 0.20: reasons.append(f"Edge density irregularity in {int(edge*GRID*GRID)} blocks (score={edge:.2f})")
    if noise > 0.20: reasons.append(f"Noise pattern inconsistency in {int(noise*GRID*GRID)} blocks (score={noise:.2f})")
    if text  > 0.15: reasons.append(f"Text blob size/shape outliers detected (score={text:.2f})")
    if not reasons:
        reasons.append("No significant tampering signals detected")

    risk_level = "LOW"
    for lo, hi, level, _ in RISK_BANDS:
        if lo <= final_score < hi:
            risk_level = level
            break

    return final_score, risk_level, reasons


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 7 ─ VISUALISATION
#  Saves a 2×3 panel figure showing every analysis step
# ═══════════════════════════════════════════════════════════════════════════════

def create_visualization(
    image:         np.ndarray,
    ela_vis:       np.ndarray,
    noise_vis:     np.ndarray,
    blur_blocks:   np.ndarray,
    edge_blocks:   np.ndarray,
    noise_blocks:  np.ndarray,
    outlier_boxes: list,
    result:        dict,
    save_path:     str,
) -> None:
    """
    Panel layout:
      ┌──────────────┬──────────────┬──────────────┐
      │  Original    │  ELA Map     │  Noise Map   │
      ├──────────────┼──────────────┼──────────────┤
      │  Blur Blocks │  Combined    │  Overlay     │
      │  (heatmap)   │  Heatmap     │  (boxes)     │
      └──────────────┴──────────────┴──────────────┘
    """
    # Convert BGR → RGB for matplotlib
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Combined block suspicion = average of the three block-level checks
    combined = (blur_blocks + edge_blocks + noise_blocks) / 3.0

    # Build overlay image — draw red boxes on suspicious blocks, yellow on text outliers
    overlay = img_rgb.copy()
    h_img, w_img = image.shape[:2]
    bh, bw = max(h_img // GRID, 1), max(w_img // GRID, 1)

    for i in range(GRID):
        for j in range(GRID):
            if combined[i, j] > 0.3:
                x1, y1 = j * bw, i * bh
                x2, y2 = min(x1 + bw, w_img), min(y1 + bh, h_img)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (220, 50, 50), 3)

    for (bx, by, bw_, bh_) in outlier_boxes:
        cv2.rectangle(overlay, (bx, by), (bx + bw_, by + bh_), (255, 210, 0), 2)

    # Custom dark-theme red heatmap
    risk_cmap = LinearSegmentedColormap.from_list(
        "risk", ["#0d0d1a", "#8b0000", "#ff4444", "#ffaa00"]
    )

    level_color = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
    color = level_color.get(result["risk_level"], "white")

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.patch.set_facecolor("#12121f")

    header = (
        f"Medical Document Tampering Analysis  ─  "
        f"Risk Level: {result['risk_level']}   "
        f"Final Score: {result['risk_score']:.1%}"
    )
    fig.suptitle(header, fontsize=15, fontweight="bold", color=color, y=0.98)

    def _show(ax, img, title, cmap=None, vmin=None, vmax=None):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, color="#dddddd", fontsize=10, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#12121f")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    _show(axes[0, 0], img_rgb, "① Original Image")
    _show(axes[0, 1],
          cv2.cvtColor(ela_vis, cv2.COLOR_BGR2RGB),
          "② ELA Map  (bright = re-edited pixels)")
    _show(axes[0, 2], noise_vis,
          "③ Noise Map  (bright = noisy, inconsistency = suspicious)",
          cmap="inferno")
    _show(axes[1, 0], blur_blocks,
          "④ Blur Inconsistency  (white blocks = outliers)",
          cmap=risk_cmap, vmin=0, vmax=1)
    _show(axes[1, 1], combined,
          "⑤ Combined Suspicion Heatmap  (avg of blur+edge+noise)",
          cmap=risk_cmap, vmin=0, vmax=1)
    _show(axes[1, 2], overlay,
          "⑥ Suspicious Regions  (red=block anomaly, yellow=text outlier)")

    # Score summary bar at the bottom
    score_line = (
        f"ELA {result['ela_score']:.2f}   "
        f"Blur {result['blur_score']:.2f}   "
        f"Edge {result['edge_score']:.2f}   "
        f"Noise {result['noise_score']:.2f}   "
        f"Text {result['text_score']:.2f}   "
        f"│   Tampered Blocks: {result['tampered_regions']}/{GRID*GRID}   "
        f"│   FINAL: {result['risk_score']:.2f}"
    )
    fig.text(
        0.5, 0.005, score_line,
        ha="center", fontsize=10, color="#aaaacc",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e1e33", edgecolor="#444466"),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    [saved] {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(file_path: str, save_visual: bool = True) -> dict:
    """
    Run the full 5-method pipeline on one file.
    Returns a structured result dictionary.
    """
    file_path = str(file_path)
    print(f"\n{'─'*62}")
    print(f"  File : {Path(file_path).name}")
    print(f"{'─'*62}")

    # ── Load ──────────────────────────────────────────────────────────────────
    image = load_image(file_path)
    h, w  = image.shape[:2]
    print(f"  Size : {w} × {h} px")

    # ── Run all 5 checks ──────────────────────────────────────────────────────
    print("  [1/5] ELA check ...")
    ela_score,   ela_vis                          = ela_check(image)

    print("  [2/5] Blur consistency ...")
    blur_score,  blur_blocks                      = blur_consistency_check(image)

    print("  [3/5] Edge consistency ...")
    edge_score,  edge_blocks                      = edge_consistency_check(image)

    print("  [4/5] Noise map ...")
    noise_score, noise_vis,    noise_blocks       = noise_check(image)

    print("  [5/5] Text irregularity ...")
    text_score,  outlier_boxes                    = text_irregularity_check(image)

    # ── Combine ───────────────────────────────────────────────────────────────
    final_score, risk_level, reasons = compute_final_score(
        ela_score, blur_score, edge_score, noise_score, text_score
    )

    combined_blocks  = (blur_blocks + edge_blocks + noise_blocks) / 3.0
    tampered_regions = int(np.sum(combined_blocks > 0.3))

    result = {
        "file":             Path(file_path).name,
        "ela_score":        ela_score,
        "blur_score":       blur_score,
        "edge_score":       edge_score,
        "noise_score":      noise_score,
        "text_score":       text_score,
        "tampered_regions": tampered_regions,
        "total_blocks":     GRID * GRID,
        "risk_score":       final_score,
        "risk_level":       risk_level,
        "reasons":          reasons,
    }

    # ── Visualise ─────────────────────────────────────────────────────────────
    if save_visual:
        out_path = str(Path(file_path).with_suffix("")) + "_analysis.png"
        create_visualization(
            image, ela_vis, noise_vis,
            blur_blocks, edge_blocks, noise_blocks,
            outlier_boxes, result, out_path,
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH MODE — run on real/ and fake/ folders to measure accuracy
# ═══════════════════════════════════════════════════════════════════════════════

def batch_test(real_dir: str, fake_dir: str) -> None:
    """
    Test every image in both folders, print a summary table,
    and report how many the system correctly classifies.

    A real document is "correct" if it scores LOW.
    A fake document is "correct" if it scores MEDIUM or HIGH.
    """
    SUPPORTED = {".jpg", ".jpeg", ".png", ".tiff", ".tif",
                 ".webp", ".heic", ".heif", ".pdf"}

    def run_folder(folder: str, true_label: str) -> list[dict]:
        results = []
        files   = sorted(
            f for f in Path(folder).iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED
        )
        print(f"\n{'═'*66}")
        print(f"  {true_label.upper()} folder — {len(files)} files")
        print(f"{'═'*66}")
        print(f"  {'File':<42} {'Score':>6}  {'Level':<8}  {'Correct?'}")
        print(f"  {'─'*42} {'─'*6}  {'─'*8}  {'─'*8}")

        for f in files:
            r     = analyze(str(f), save_visual=False)
            pred  = r["risk_level"]
            ok    = (true_label == "REAL"   and pred == "LOW") or \
                    (true_label == "FAKE"   and pred in ("MEDIUM", "HIGH"))
            tick  = "✓" if ok else "✗"
            print(f"  {f.name:<42} {r['risk_score']:>6.2f}  {pred:<8}  {tick}")
            results.append({"pred": pred, "true": true_label, "correct": ok})

        return results

    all_res = run_folder(real_dir, "REAL") + run_folder(fake_dir, "FAKE")

    total   = len(all_res)
    correct = sum(1 for r in all_res if r["correct"])
    reals   = [r for r in all_res if r["true"] == "REAL"]
    fakes   = [r for r in all_res if r["true"] == "FAKE"]

    print(f"\n{'═'*66}")
    print(f"  BATCH SUMMARY")
    print(f"{'─'*66}")
    print(f"  Total files tested : {total}")
    print(f"  Real  — correct    : {sum(r['correct'] for r in reals)}/{len(reals)}")
    print(f"  Fake  — correct    : {sum(r['correct'] for r in fakes)}/{len(fakes)}")
    print(f"  Overall accuracy   : {correct}/{total}  ({100*correct/max(total,1):.1f}%)")
    print(f"{'═'*66}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-feature medical document tampering detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tampering_detector.py prescription.jpg
  python tampering_detector.py report.pdf
  python tampering_detector.py --batch dataset/real/ dataset/fake/
        """,
    )
    parser.add_argument(
        "input", nargs="?",
        help="Path to image or PDF to analyse",
    )
    parser.add_argument(
        "--batch", nargs=2, metavar=("REAL_DIR", "FAKE_DIR"),
        help="Run accuracy test on two folders",
    )
    args = parser.parse_args()

    if args.batch:
        batch_test(args.batch[0], args.batch[1])

    elif args.input:
        if not Path(args.input).exists():
            sys.exit(f"[ERROR] File not found: {args.input}")

        result = analyze(args.input)

        print(f"\n{'═'*62}")
        print("  FINAL RESULT")
        print(f"{'─'*62}")
        print(json.dumps(result, indent=2))
        print(f"{'═'*62}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
