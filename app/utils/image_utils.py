"""
image_utils.py — Shared OpenCV helpers used by ocr.py and tampering.py.
Centralising image operations here avoids duplication across core modules.
"""

from __future__ import annotations
import io
import cv2
import numpy as np
from PIL import Image


def load_image(image_bytes: bytes) -> np.ndarray:
    """
    Load raw image bytes into an OpenCV BGR numpy array.
    Falls back to PIL for formats OpenCV cannot decode directly (TIFF, WebP, etc.).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        # PIL fallback
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
