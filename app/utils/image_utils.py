"""
image_utils.py — Shared OpenCV helpers used by ocr.py and tampering.py.
Centralising image operations here avoids duplication across core modules.
"""

from __future__ import annotations
import io
import cv2
import numpy as np
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC support unavailable; install pillow-heif to enable it


def is_pdf(file_bytes: bytes) -> bool:
    """Detect PDF by checking its magic bytes header (%PDF-)."""
    return file_bytes[:4] == b"%PDF"


def pdf_to_images(pdf_bytes: bytes) -> list[bytes]:
    """
    Convert every page of a PDF into a PNG image (bytes).
    Uses PyMuPDF (fitz) — no external tools like poppler required.

    Renders at 2× zoom (~144 DPI) which gives Tesseract enough resolution
    to read text accurately.

    Returns a list of PNG byte strings, one per page.
    Raises RuntimeError if pymupdf is not installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "pymupdf is required for PDF support. "
            "Run: pip install pymupdf"
        )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[bytes] = []

    for page in doc:
        # 2× zoom → ~144 DPI — good balance of quality vs. speed
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pages.append(pix.tobytes("png"))

    doc.close()
    return pages


def load_image(image_bytes: bytes) -> np.ndarray:
    """
    Load raw image bytes into an OpenCV BGR numpy array.
    Falls back to PIL for formats OpenCV cannot decode directly
    (TIFF, WebP, HEIC, etc.).

    NOTE: Do NOT pass raw PDF bytes here — convert with pdf_to_images() first.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        # PIL fallback — handles HEIC (via pillow-heif), TIFF, WebP, etc.
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return image


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality before passing to Tesseract:
      1. Upscale small images (Tesseract needs at least ~1000px wide)
      2. Convert to grayscale
      3. Denoise
      4. Adaptive threshold → high-contrast black/white text
    """
    h, w = image.shape[:2]

    # Upscale if too small
    if w < 1000:
        scale = 1000 / w
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold handles uneven lighting across document
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return thresh


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    """Scale image to a given width while preserving aspect ratio."""
    h, w = image.shape[:2]
    if w == width:
        return image
    scale = width / w
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
