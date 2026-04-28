"""
ocr.py — Step 1 of the pipeline.
Pre-processes the input image with OpenCV and extracts raw text using Tesseract.
"""

from __future__ import annotations
import pytesseract
from app.utils.image_utils import load_image, preprocess_for_ocr


def extract_text(image_bytes: bytes) -> str:
    """
    Args:
        image_bytes: raw bytes of the uploaded image file.

    Returns:
        Raw text string extracted by Tesseract.
        Returns an empty string if extraction fails.
    """
    try:
        image = load_image(image_bytes)
        processed = preprocess_for_ocr(image)

        # --oem 3  → LSTM neural net engine (most accurate)
        # --psm 6  → assume a uniform block of text (good for prescriptions)
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(processed, config=config, lang="eng")
        return text.strip()

    except Exception as exc:
        # Do not crash the pipeline — return empty and let downstream handle it
        print(f"[OCR] Warning: extraction failed — {exc}")
        return ""
