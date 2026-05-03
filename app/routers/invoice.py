"""
Invoice extraction API endpoint.

Full pipeline:
1. File upload (image/PDF) → validated
2. Ensemble OCR (PaddleOCR + EasyOCR merged by IoU)
3. Platform detection (9 platforms)
4. Regex field extraction
5. LLM fallback if < min_fields extracted
6. Post-extraction validation (GSTIN, tax math, totals)
7. Structured JSON response with confidence + warnings
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from app.models.schemas import (
    ExtractionResponse, OCRMetadata, InvoiceData,
    DocumentType, ExtractionStatus, DocumentPlatform,
)
from app.services.ocr_service import OCRService
from app.services.extraction_service import InvoiceExtractionService
from app.services.validator import ExtractionValidator
from app.services.llm_extractor import LLMExtractor
from app.services.finetuned_extractor import FinetunedLLMExtractor
from app.config import settings
import time
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Singletons — created once at first request, reused thereafter
_ocr = None
_extractor = InvoiceExtractionService()
_validator = ExtractionValidator()
_llm = LLMExtractor()
_finetuned_llm = FinetunedLLMExtractor()


def get_ocr():
    """Lazy-load OCR service (avoids slow startup if not needed)."""
    global _ocr
    if _ocr is None:
        _ocr = OCRService()
    return _ocr


ALLOWED_TYPES = [
    "image/jpeg", "image/png", "image/jpg",
    "image/webp", "application/pdf",
]


@router.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    """
    Upload an invoice (image or PDF) and get structured data back.

    **Supports:** Amazon, Flipkart, Meesho, Myntra, Swiggy, Zomato,
    BigBasket, Blinkit, JioMart

    **Features:**
    - Ensemble OCR (PaddleOCR + EasyOCR merged by bounding box IoU)
    - LLM fallback for messy/unusual invoices
    - GSTIN checksum validation
    - Tax math cross-checking (CGST + SGST = Total Tax)
    - Total amount verification (Subtotal + Tax - Discount = Total)

    **Accepted formats:** JPEG, PNG, WebP, PDF
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file.content_type}. "
                f"Accepted: {', '.join(ALLOWED_TYPES)}"
            ),
        )

    start = time.time()
    extraction_id = str(uuid.uuid4())[:8]
    warnings = []
    fallback_used = False
    contents = await file.read()

    try:
        ocr = get_ocr()

        # Step 1: Ensemble OCR
        raw_text, ocr_confidence, ocr_meta = ocr.extract_from_file(
            contents, file.content_type
        )

        if not raw_text.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "OCR could not read any text from this file. "
                    "Try a clearer image or higher-resolution PDF."
                ),
            )

        # Step 2: Detect platform
        platform = _extractor.detect_platform(raw_text)
        logger.info(f"[{extraction_id}] Platform detected: {platform.value}")

        # Step 3: Regex extraction
        invoice_data = _extractor.extract_all_fields(raw_text, platform)
        logger.info(
            f"[{extraction_id}] Regex extracted {invoice_data.fields_extracted} fields"
        )

        # Step 4: LLM fallback if extraction is sparse
        if (
            invoice_data.fields_extracted < settings.min_fields_before_llm
            and settings.use_llm_fallback
        ):
            logger.info(
                f"[{extraction_id}] Triggering LLM fallback "
                f"({invoice_data.fields_extracted} < {settings.min_fields_before_llm})"
            )

            llm_result = None

            # Try fine-tuned model first (if enabled)
            if settings.use_finetuned_model:
                logger.info(f"[{extraction_id}] Trying fine-tuned LoRA model...")
                llm_result = _finetuned_llm.extract(raw_text, platform)
                if llm_result:
                    fallback_used = True
                    warnings.append(
                        "Fine-tuned LoRA model used for extraction"
                    )

            # Fall back to Ollama if fine-tuned model failed or disabled
            if llm_result is None:
                logger.info(f"[{extraction_id}] Trying Ollama ({settings.ollama_model})...")
                llm_result = _llm.extract(raw_text, platform)
                if llm_result:
                    fallback_used = True
                    warnings.append(
                        f"Ollama ({settings.ollama_model}) fallback used"
                    )

            if llm_result:
                invoice_data = _merge_extractions(invoice_data, llm_result)
                invoice_data.fields_extracted = _extractor._count_fields(
                    invoice_data
                )
                logger.info(
                    f"[{extraction_id}] After LLM merge: "
                    f"{invoice_data.fields_extracted} fields"
                )

        # Step 5: Validate
        valid, validation_warnings = _validator.validate(invoice_data)
        warnings.extend(validation_warnings)
        invoice_data.validation_warnings = validation_warnings

        # Step 6: Calculate confidence
        confidence = _extractor.calculate_confidence(
            invoice_data, ocr_confidence
        )

        # Determine status
        n = invoice_data.fields_extracted
        if n >= 5 and valid:
            status = ExtractionStatus.SUCCESS
            message = (
                f"Successfully extracted {n} fields "
                f"from {platform.value} invoice"
            )
        elif n >= 3:
            status = ExtractionStatus.PARTIAL
            message = (
                f"Partially extracted {n} fields — "
                f"some data may be missing or inaccurate"
            )
        elif ocr_confidence < settings.confidence_threshold:
            status = ExtractionStatus.LOW_CONFIDENCE
            message = "OCR confidence too low — try a clearer image"
        else:
            status = ExtractionStatus.PARTIAL
            message = f"Extracted {n} fields — document format may be unusual"

        return ExtractionResponse(
            status=status,
            message=message,
            document_type=DocumentType.INVOICE,
            platform=platform,
            confidence_score=confidence,
            extracted_data=invoice_data,
            ocr_metadata=OCRMetadata(
                engine_used=ocr_meta.get("engine_used", "ensemble"),
                confidence=ocr_meta.get("confidence", ocr_confidence),
                paddle_regions=ocr_meta.get("paddle_regions", 0),
                easy_regions=ocr_meta.get("easy_regions", 0),
                merged_regions=ocr_meta.get("merged_regions", 0),
                processing_time_ms=ocr_meta.get("processing_time_ms", 0),
                fallback_used=fallback_used,
                pages_processed=ocr_meta.get("pages_processed", 1),
            ),
            raw_text=raw_text,
            processing_time_seconds=round(time.time() - start, 3),
            validation_passed=valid,
            warnings=warnings,
            extraction_id=extraction_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{extraction_id}] Extraction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}",
        )


@router.get("/platforms")
def list_platforms():
    """List all supported invoice platforms and features."""
    return {
        "supported_platforms": [p.value for p in DocumentPlatform if p != DocumentPlatform.UNKNOWN],
        "supported_formats": ["PDF", "JPG", "JPEG", "PNG", "WebP"],
        "features": [
            "Ensemble OCR (PaddleOCR + EasyOCR merged by bounding box IoU)",
            "LLM fallback extraction (Mistral via Ollama — local, free)",
            "GSTIN checksum validation",
            "Tax math cross-checking (CGST + SGST = Total Tax)",
            "Total amount verification",
            "Platform auto-detection",
        ],
    }


def _merge_extractions(
    regex_data: InvoiceData, llm_data: InvoiceData
) -> InvoiceData:
    """
    Merge regex and LLM extraction results.
    Strategy: regex takes priority (higher precision), LLM fills gaps.
    Now works with flat schema — all fields at top level.
    """
    merged = regex_data.model_copy(deep=True)

    # All mergeable fields (skip platform, metadata, lists)
    merge_fields = [
        "billing_address", "shipping_address", "invoice_type",
        "order_number", "invoice_number", "order_date",
        "invoice_details", "invoice_date", "seller_info",
        "seller_pan", "seller_gst", "fssai_license",
        "billing_state_code", "shipping_state_code",
        "place_of_supply", "place_of_delivery", "reverse_charge",
        "amount_in_words", "seller_name", "seller_address",
        "total_tax", "total_amount",
        "buyer_name", "buyer_phone", "subtotal",
        "cgst_amount", "sgst_amount", "igst_amount",
        "cgst_rate", "sgst_rate", "igst_rate",
        "discount", "delivery_charge", "packaging_charge",
        "payment_method",
    ]

    for field in merge_fields:
        regex_val = getattr(merged, field, None)
        llm_val = getattr(llm_data, field, None)
        if (regex_val is None or regex_val == "") and llm_val is not None:
            setattr(merged, field, llm_val)

    return merged
