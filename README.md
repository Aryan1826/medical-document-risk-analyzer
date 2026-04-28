# Medical Document Risk Analyzer

> An AI-powered FastAPI backend that analyzes medical prescriptions for authenticity risk.
> No classifier. No dataset required to run. Fully rule-based + OCR + image analysis.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What it does

Upload a medical prescription image → get back a **risk score (0–100)** with plain-English explanations.

```json
{
  "risk_score": 72,
  "risk_level": "HIGH",
  "reasons": [
    "[TAMPERING] High ELA variance (18.4) — region may have been edited",
    "[DOCTOR] Doctor not found in verified database",
    "[RULE] Missing mandatory field: Registration Number"
  ],
  "parsed_fields": {
    "doctor_name": "Dr. R. Mehta",
    "registration_number": null,
    "patient_name": "Aryan Patel",
    "issue_date": "22/04/2026",
    "leave_days": 7,
    "diagnosis": "Viral fever"
  },
  "doctor_verification": {
    "status": "UNVERIFIED",
    "confidence": 0.0,
    "message": "Doctor not found in verified database",
    "matched_doctor": null
  },
  "tampering": {
    "score": 0.40,
    "tampered": true,
    "findings": ["High ELA variance (18.4) — region may have been edited"],
    "raw_scores": {
      "ela_mean": 18.4,
      "blur_laplacian_variance": 210.5,
      "noise_tile_std": 12.3
    }
  }
}
```

---

## How the pipeline works

```
Image Upload
     │
     ▼
  Step 1 ─ OCR         (OpenCV pre-process → Tesseract → raw text)
     │
     ▼
  Step 2 ─ Parser      (regex + RapidFuzz → structured fields)
     │
     ▼
  Step 3 ─ Rules       (format checks, date validation, required fields)
     │
     ▼
  Step 4 ─ Doctor DB   (fuzzy lookup in doctors.json → VERIFIED / SUSPICIOUS / UNVERIFIED)
     │
     ▼
  Step 5 ─ Tampering   (ELA + blur + noise analysis → tamper score)
     │
     ▼
  Step 6 ─ Risk Score  (weighted sum → 0–100 → LOW / MEDIUM / HIGH)
```

| Score | Level  | Meaning                          |
|-------|--------|----------------------------------|
| 0–30  | LOW    | Likely legitimate                |
| 31–60 | MEDIUM | Needs human review               |
| 61–100| HIGH   | Likely forged or tampered        |

---

## Project Structure

```
medical-document-risk-analyzer/
│
├── app/
│   ├── main.py                  ← FastAPI app entry point
│   ├── api/
│   │   └── routes.py            ← POST /api/analyze endpoint
│   ├── core/
│   │   ├── ocr.py               ← Step 1: image → text
│   │   ├── parser.py            ← Step 2: text → fields
│   │   ├── rules.py             ← Step 3: field validation
│   │   ├── doctor_verify.py     ← Step 4: DB lookup
│   │   ├── tampering.py         ← Step 5: image analysis
│   │   └── risk_scorer.py       ← Step 6: final score
│   ├── models/
│   │   └── schemas.py           ← Pydantic response models
│   └── utils/
│       ├── image_utils.py       ← shared OpenCV helpers
│       └── text_utils.py        ← shared text cleaning
│
├── data/
│   └── doctors.json             ← trusted doctor database
│
├── samples/
│   ├── fake/                    ← test with fake prescriptions
│   └── real/                    ← test with real prescriptions
│
├── tests/
│   └── test_analyze.py          ← pytest test suite
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | 3.10+   |
| Tesseract   | 4.x or 5.x |
| pip         | latest  |

### 1 — Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install -y tesseract-ocr
```

**Windows:**
Download installer from https://github.com/UB-Mannheim/tesseract/wiki
Then add to PATH and set in code:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Verify:
```bash
tesseract --version
```

---

### 2 — Clone the repo

```bash
git clone https://github.com/Aryan1826/medical-document-risk-analyzer.git
cd medical-document-risk-analyzer
```

---

### 3 — Create virtual environment

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

### 4 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5 — Copy environment config

```bash
cp .env.example .env
```

No changes needed for local development.

---

### 6 — Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server is now running at:
- API: http://localhost:8000
- Swagger UI: **http://localhost:8000/docs**
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

---

## Dataset Download (Google Drive)

The `samples/` folder is where you put your test prescription images.

### Method A — gdown (recommended)

```bash
pip install gdown

# Replace YOUR_FILE_ID with the ID from your Google Drive share link
# Share link format: https://drive.google.com/file/d/YOUR_FILE_ID/view
gdown "https://drive.google.com/uc?id=YOUR_FILE_ID" -O samples.zip
unzip samples.zip -d samples/
```

### Method B — wget (if the file is publicly shared)

```bash
# Replace YOUR_FILE_ID with your actual ID
wget -O samples.zip "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
unzip samples.zip -d samples/
```

### Folder structure after extraction

```
samples/
├── fake/
│   ├── fake_001.jpg
│   ├── fake_002.jpg
│   └── ...
└── real/
    ├── real_001.jpg
    └── ...
```

The system doesn't train on these images — they are used for **manual testing** only.
Point Swagger UI to any file in `samples/fake/` or `samples/real/` to test.

---

## API Usage

### Endpoint

```
POST /api/analyze
Content-Type: multipart/form-data
```

### cURL example

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@samples/fake/fake_001.jpg"
```

### Python example

```python
import requests

with open("samples/fake/fake_001.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/analyze",
        files={"file": ("fake_001.jpg", f, "image/jpeg")},
    )

print(response.json())
```

### Example Response

```json
{
  "risk_score": 55,
  "risk_level": "MEDIUM",
  "reasons": [
    "[DOCTOR] Doctor not found in verified database",
    "[RULE] Registration number has unexpected format: 'REG12345'",
    "[RULE] Missing mandatory field: Patient Name"
  ],
  "parsed_fields": {
    "doctor_name": "Dr. S. Kumar",
    "registration_number": "REG12345",
    "hospital_name": "City Hospital",
    "patient_name": null,
    "issue_date": "15/04/2026",
    "leave_days": 5,
    "diagnosis": "Acute fever"
  },
  "doctor_verification": {
    "status": "UNVERIFIED",
    "confidence": 0.0,
    "message": "Doctor not found in verified database",
    "matched_doctor": null
  },
  "tampering": {
    "score": 0.0,
    "tampered": false,
    "findings": [],
    "raw_scores": {
      "ela_mean": 5.2,
      "blur_laplacian_variance": 320.1,
      "noise_tile_std": 11.4
    }
  }
}
```

---

## Testing

### Run all tests

```bash
pytest tests/ -v
```

### Test via Swagger UI (step-by-step)

1. Open http://localhost:8000/docs
2. Click **POST /api/analyze**
3. Click **Try it out**
4. Click **Choose File** → select any image from `samples/fake/`
5. Click **Execute**
6. View the response body below

### Test via health check

```bash
curl http://localhost:8000/health
# → {"status":"ok","service":"Medical Document Risk Analyzer","version":"1.0.0"}
```

---

## Adding More Doctors

Edit `data/doctors.json` and add entries in this format:

```json
{
  "name": "Dr. Full Name",
  "registration_number": "MCI-2020-12345",
  "hospital": "Hospital Name",
  "city": "City",
  "specialisation": "General Physician",
  "active": true
}
```

No restart needed if you use the file-based lookup — the file is re-read on each request.

---

## Common Errors & Fixes

### `TesseractNotFoundError`
**Cause:** Tesseract binary not installed or not in PATH.
```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt install tesseract-ocr

# Verify
tesseract --version
```

### `ModuleNotFoundError: cv2`
```bash
pip install opencv-python==4.9.0.80
```

### `error: (-215:Assertion failed) !_src.empty()`
**Cause:** OpenCV received an empty or corrupt image.
- Check the uploaded file is a valid JPEG/PNG
- Check it is not 0 bytes

### `OSError: [Errno 2] No such file or directory: 'data/doctors.json'`
**Cause:** Server started from wrong directory.
```bash
# Always run from the project root
cd medical-document-risk-analyzer
uvicorn app.main:app --reload
```

### Low OCR accuracy
- Use high-res scans (300 DPI minimum)
- Ensure document is well-lit with no shadows
- Try JPEG → PNG conversion before uploading

---

## Why no ML classifier?

This project deliberately avoids a binary real/fake classifier because:

1. **No balanced dataset** — fake samples only, or very few real ones
2. **Classifiers overfit** — with < 100 images, a CNN learns image artifacts, not forgery patterns
3. **Rule-based is more explainable** — each risk reason is traceable to a specific check
4. **Faster to build** — no training pipeline, no GPU needed, runs on CPU

The **dataset** (your fake prescriptions) is used only to:
- Test that the OCR pipeline extracts text correctly
- Test that the tampering detector catches real edited images
- Demonstrate the system in Swagger UI during evaluation

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Web framework | FastAPI |
| ASGI server | Uvicorn |
| OCR engine | pytesseract + Tesseract |
| Image processing | OpenCV |
| Fuzzy matching | RapidFuzz |
| Image loading | Pillow |
| Data validation | Pydantic v2 |
| Testing | pytest + httpx |
