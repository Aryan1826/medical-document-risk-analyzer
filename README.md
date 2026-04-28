# Medical Document Risk Analyzer

> AI-powered system to detect forged or tampered medical certificates using Transfer Learning (MobileNetV2) + OCR analysis.

---

## Project Structure

```
medical-document-risk-analyzer/
├── dataset/
│   ├── real/          ← place your real certificate images here
│   └── fake/          ← place your forged certificate images here
├── models/            ← saved weights & thresholds (git-ignored)
├── notebook/
│   └── medical_document_risk_analyzer.ipynb   ← main Colab notebook
├── src/               ← standalone Python modules
└── requirements.txt
```

## Quick Start (Google Colab)

1. Upload the notebook to [Google Colab](https://colab.research.google.com)
2. Mount Google Drive and point `DATASET_PATH` to your image folder
3. Run all cells top-to-bottom

## Dataset Format

```
dataset/
├── real/
│   ├── cert_001.jpg
│   └── ...
└── fake/
    ├── fake_001.jpg
    └── ...
```

Minimum recommended: **50 images per class**.

## Tech Stack

- **Model:** MobileNetV2 (ImageNet pre-trained, Transfer Learning)
- **Framework:** TensorFlow / Keras
- **OCR:** pytesseract
- **UI:** Streamlit / Flask
- **Platform:** Google Colab (GPU)
