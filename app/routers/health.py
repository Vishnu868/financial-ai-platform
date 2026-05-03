"""Health and status endpoints."""

from fastapi import APIRouter
from app.config import settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
def root():
    """Root endpoint — project info and documentation links."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "invoice_extract": "POST /api/v2/invoice/extract",
            "bank_extract": "POST /api/v2/bank/extract",
            "classify": "POST /api/v2/classify/document",
            "batch_process": "POST /api/v2/batch/process",
            "batch_status": "GET /api/v2/batch/status/{batch_id}",
            "export_csv": "POST /api/v2/export/csv",
        },
    }


@router.get("/health")
def health():
    """Health check — returns healthy if the server is running."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "llm_fallback_enabled": settings.use_llm_fallback,
        "gpu_enabled": settings.ocr_use_gpu,
    }
