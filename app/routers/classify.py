"""Document classification endpoint — auto-detect invoice vs bank statement."""

from fastapi import APIRouter, File, UploadFile, HTTPException
from app.models.schemas import (
    ClassificationResult, DocumentType, DocumentPlatform,
)
from app.services.ocr_service import OCRService
from app.services.extraction_service import InvoiceExtractionService

router = APIRouter()
_ocr = None


def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = OCRService()
    return _ocr


INVOICE_WORDS = [
    "invoice", "bill", "receipt", "tax invoice", "gstin",
    "gst number", "invoice number", "order id", "sold by",
    "total amount", "subtotal", "cgst", "sgst", "igst",
    "seller", "hsn", "sac", "place of supply",
]

BANK_WORDS = [
    "bank statement", "account statement", "account number",
    "ifsc", "opening balance", "closing balance", "transaction",
    "debit", "credit", "account holder", "balance b/f",
    "statement of account", "passbook", "branch",
]


@router.post("/document", response_model=ClassificationResult)
async def classify_document(file: UploadFile = File(...)):
    """
    Auto-detect document type: invoice or bank statement.
    Also identifies the platform (Amazon, Flipkart, etc.) for invoices.
    """
    contents = await file.read()

    ocr = get_ocr()

    content_type = file.content_type or "application/octet-stream"
    raw_text, _, _ = ocr.extract_from_file(contents, content_type)
    if not raw_text:
        raise HTTPException(400, "OCR failed to extract text")
    text_lower = raw_text.lower()

    invoice_score = sum(1 for w in INVOICE_WORDS if w in text_lower)
    bank_score = sum(1 for w in BANK_WORDS if w in text_lower)
    total = invoice_score + bank_score

    if total == 0:
        return ClassificationResult(
            document_type=DocumentType.UNKNOWN,
            platform=DocumentPlatform.UNKNOWN,
            confidence=0.0,
            method="keyword",
            all_scores={},
        )

    if invoice_score >= bank_score:
        doc_type = DocumentType.INVOICE
        confidence = invoice_score / total
        extractor = InvoiceExtractionService()
        platform = extractor.detect_platform(raw_text)
    else:
        doc_type = DocumentType.BANK_STATEMENT
        confidence = bank_score / total
        platform = DocumentPlatform.UNKNOWN

    return ClassificationResult(
        document_type=doc_type,
        platform=platform,
        confidence=round(min(confidence, 1.0), 3),
        method="keyword",
        all_scores={
            "invoice": round(invoice_score / max(total, 1), 3),
            "bank_statement": round(bank_score / max(total, 1), 3),
        },
    )
