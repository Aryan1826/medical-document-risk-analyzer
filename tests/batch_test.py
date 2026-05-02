"""
batch_test.py — Run the full pipeline on every image in dataset/fake/ and dataset/real/
and print a summary table showing risk_score, risk_level, and tampering per file.

Usage (from project root, with server running on port 8000):
    python tests/batch_test.py

Or test without a live server (direct pipeline call):
    python tests/batch_test.py --direct
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DATASET_ROOT = Path(__file__).parent.parent / "dataset"


def _mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".webp": "image/webp",
        ".heic": "image/heic",
        ".heif": "image/heif",
    }.get(ext, "application/octet-stream")


# ── HTTP mode (server must be running) ───────────────────────────────────────

def run_http(url: str, folder: Path, label: str) -> list[dict]:
    import httpx

    results = []
    files = sorted(folder.iterdir())
    print(f"\n{'='*70}")
    print(f"  {label.upper()} — {len(files)} files  [{folder}]")
    print(f"{'='*70}")
    print(f"{'File':<45} {'Score':>5}  {'Level':<8}  {'Tampered'}")
    print(f"{'-'*45} {'-'*5}  {'-'*8}  {'-'*8}")

    for f in files:
        if not f.is_file():
            continue
        mime = _mime(f)
        try:
            with open(f, "rb") as fh:
                resp = httpx.post(
                    url,
                    files={"file": (f.name, fh, mime)},
                    timeout=30,
                )
            if resp.status_code == 200:
                data = resp.json()
                score = data["risk_score"]
                level = data["risk_level"]
                tampered = data["tampering"]["tampered"]
                print(f"{f.name:<45} {score:>5}  {level:<8}  {tampered}")
                results.append({"file": f.name, "label": label, "score": score, "level": level})
            else:
                print(f"{f.name:<45}  ERROR {resp.status_code}: {resp.text[:60]}")
        except Exception as e:
            print(f"{f.name:<45}  EXCEPTION: {e}")

    return results


# ── Direct mode (no server needed) ───────────────────────────────────────────

def run_direct(folder: Path, label: str) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from app.core.ocr import extract_text
    from app.core.parser import parse_fields
    from app.core.rules import validate_rules
    from app.core.doctor_verify import verify_doctor
    from app.core.tampering import analyze_tampering
    from app.core.risk_scorer import compute_risk_score

    results = []
    files = sorted(folder.iterdir())
    print(f"\n{'='*70}")
    print(f"  {label.upper()} — {len(files)} files  [{folder}]")
    print(f"{'='*70}")
    print(f"{'File':<45} {'Score':>5}  {'Level':<8}  {'Tampered'}")
    print(f"{'-'*45} {'-'*5}  {'-'*8}  {'-'*8}")

    for f in files:
        if not f.is_file():
            continue
        try:
            image_bytes = f.read_bytes()
            text = extract_text(image_bytes)
            fields = parse_fields(text)
            violations = validate_rules(fields)
            doctor = verify_doctor(fields.get("doctor_name"), fields.get("registration_number"))
            tampering = analyze_tampering(image_bytes)
            result = compute_risk_score(violations, doctor, tampering, fields)
            score = result["risk_score"]
            level = result["risk_level"]
            tampered = tampering["tampered"]
            print(f"{f.name:<45} {score:>5}  {level:<8}  {tampered}")
            results.append({"file": f.name, "label": label, "score": score, "level": level})
        except Exception as e:
            print(f"{f.name:<45}  ERROR: {e}")

    return results


def summarise(results: list[dict]) -> None:
    if not results:
        return
    from collections import Counter
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for label in ("real", "fake"):
        subset = [r for r in results if r["label"] == label]
        if not subset:
            continue
        scores = [r["score"] for r in subset]
        levels = Counter(r["level"] for r in subset)
        avg = sum(scores) / len(scores)
        print(f"\n  {label.upper()} ({len(subset)} files)")
        print(f"    Avg risk score : {avg:.1f}")
        print(f"    LOW  : {levels.get('LOW', 0)}")
        print(f"    MEDIUM: {levels.get('MEDIUM', 0)}")
        print(f"    HIGH : {levels.get('HIGH', 0)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", action="store_true", help="Run pipeline directly (no server)")
    parser.add_argument("--url", default="http://localhost:8000/api/analyze", help="Server URL")
    parser.add_argument("--folder", default=None, help="Test only this folder (fake or real)")
    args = parser.parse_args()

    folders = []
    if args.folder:
        p = DATASET_ROOT / args.folder
        folders = [(p, args.folder)]
    else:
        folders = [
            (DATASET_ROOT / "real", "real"),
            (DATASET_ROOT / "fake", "fake"),
        ]

    all_results: list[dict] = []
    for folder, label in folders:
        if not folder.exists():
            print(f"[WARN] Folder not found: {folder}")
            continue
        if args.direct:
            all_results += run_direct(folder, label)
        else:
            all_results += run_http(args.url, folder, label)

    summarise(all_results)


if __name__ == "__main__":
    main()
