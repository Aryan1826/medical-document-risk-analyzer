"""
main.py — FastAPI application entry point.
Registers middleware, routers, and health-check endpoint.
Nothing business-logic lives here — it stays intentionally thin.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="Medical Document Risk Analyzer",
    description=(
        "Analyzes medical prescriptions for authenticity risk using "
        "OCR extraction, rule-based validation, doctor verification, "
        "and image tampering detection. Returns a 0–100 risk score."
    ),
    version="1.0.0",
    docs_url="/docs",      # Swagger UI  → http://localhost:8000/docs
    redoc_url="/redoc",    # ReDoc UI    → http://localhost:8000/redoc
)

# Allow all origins during development — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes under /api prefix
app.include_router(router, prefix="/api", tags=["Analysis"])


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """Quick liveness probe — returns 200 if the server is up."""
    return {
        "status": "ok",
        "service": "Medical Document Risk Analyzer",
        "version": "1.0.0",
    }
