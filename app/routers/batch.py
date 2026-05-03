"""
Batch processing endpoint — upload a ZIP of invoices, process all at once.
Processing runs in background; poll /status/{batch_id} for results.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from app.models.schemas import (
    BatchStatus, ExtractionResponse, OCRMetadata,
    InvoiceData, DocumentType, ExtractionStatus,
)
from app.services.ocr_service import OCRService
from app.services.extraction_service import InvoiceExtractionService
from app.services.validator import ExtractionValidator
from app.config import settings
import zipfile
import io
import uuid
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory batch status store
batch_store: dict = {}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".webp"}
CONTENT_TYPE_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".pdf": "application/pdf",
    ".webp": "image/webp",
}


def _process_batch(batch_id: str, files: list):
    """Background task: process all files in a batch."""
    batch = batch_store[batch_id]
    ocr = OCRService()
    extractor = InvoiceExtractionService()
    validator = ExtractionValidator()
    results = []

    for filename, file_bytes in files:
        try:
            ext = "." + filename.rsplit(".", 1)[-1].lower()
            content_type = CONTENT_TYPE_MAP.get(ext, "image/jpeg")

            start = time.time()
            raw_text, conf, meta = ocr.extract_from_file(file_bytes, content_type)
            platform = extractor.detect_platform(raw_text)
            data = extractor.extract_all_fields(raw_text, platform)
            valid, warns = validator.validate(data)
            confidence = extractor.calculate_confidence(data, conf)

            results.append(ExtractionResponse(
                status=ExtractionStatus.SUCCESS if valid else ExtractionStatus.PARTIAL,
                message=f"{filename}: {data.fields_extracted} fields extracted",
                document_type=DocumentType.INVOICE,
                platform=platform,
                confidence_score=confidence,
                extracted_data=data,
                ocr_metadata=OCRMetadata(
                    engine_used="ensemble",
                    confidence=conf,
                    merged_regions=meta.get("merged_regions", 0),
                ),
                processing_time_seconds=round(time.time() - start, 3),
                validation_passed=valid,
                warnings=warns,
                extraction_id=str(uuid.uuid4())[:8],
            ))
            batch["completed"] += 1
            logger.info(f"[batch:{batch_id}] Processed {filename}")

        except Exception as e:
            logger.error(f"[batch:{batch_id}] Failed {filename}: {e}")
            batch["failed"] += 1

    batch["results"] = [r.model_dump() for r in results]
    batch["status"] = "completed"
    logger.info(
        f"[batch:{batch_id}] Complete: "
        f"{batch['completed']} ok, {batch['failed']} failed"
    )


@router.post("/process")
async def start_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="ZIP file containing invoices"),
):
    """
    Upload a ZIP archive of invoice images/PDFs for batch processing.

    Returns a batch_id immediately. Poll `/batch/status/{batch_id}` for results.
    Maximum files per batch: configurable via MAX_BATCH_SIZE (default: 20).
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file")

    contents = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(contents))
    except zipfile.BadZipFile:
        raise HTTPException(400, "Invalid or corrupted ZIP file")

    # Extract valid files
    files_to_process = []
    for name in zf.namelist():
        # Skip macOS metadata
        if name.startswith("__MACOSX") or name.startswith("."):
            continue
        ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext in ALLOWED_EXTENSIONS:
            file_bytes = zf.read(name)
            files_to_process.append((name, file_bytes))

    if not files_to_process:
        raise HTTPException(
            400,
            "No valid invoice files found in ZIP. "
            f"Supported: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    if len(files_to_process) > settings.max_batch_size:
        raise HTTPException(
            400,
            f"Too many files: {len(files_to_process)}. "
            f"Maximum: {settings.max_batch_size}",
        )

    batch_id = str(uuid.uuid4())[:12]
    batch_store[batch_id] = {
        "batch_id": batch_id,
        "total_files": len(files_to_process),
        "completed": 0,
        "failed": 0,
        "status": "processing",
        "results": None,
        "created_at": datetime.now().isoformat(),
    }

    background_tasks.add_task(_process_batch, batch_id, files_to_process)

    return batch_store[batch_id]


@router.get("/status/{batch_id}")
async def batch_status(batch_id: str):
    """Check the status and results of a batch processing job."""
    if batch_id not in batch_store:
        raise HTTPException(404, f"Batch '{batch_id}' not found")
    return batch_store[batch_id]
