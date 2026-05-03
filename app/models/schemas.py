"""
Pydantic schemas matching the EXACT output template from the mentor.

The mentor's template has 22 fields (rows 2-23 in the Excel):
  billing_address, shipping_address, invoice_type, order_number,
  invoice_number, order_date, invoice_details, invoice_date,
  seller_info, seller_pan, seller_gst, fssai_license,
  billing_state_code, shipping_state_code, place_of_supply,
  place_of_delivery, reverse_charge, amount_in_words,
  seller_name, seller_address, total_tax, total_amount

Plus: platform (auto-detected)
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENUMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DocumentPlatform(str, Enum):
    AMAZON = "amazon"
    FLIPKART = "flipkart"
    MEESHO = "meesho"
    MYNTRA = "myntra"
    SWIGGY = "swiggy"
    ZOMATO = "zomato"
    BIGBASKET = "bigbasket"
    BLINKIT = "blinkit"
    JIOMART = "jiomart"
    UNKNOWN = "unknown"


class DocumentType(str, Enum):
    INVOICE = "invoice"
    BANK_STATEMENT = "bank_statement"
    UNKNOWN = "unknown"


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    LOW_CONFIDENCE = "low_confidence"
    FAILED = "failed"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INVOICE DATA — matches mentor's Output Template exactly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InvoiceItem(BaseModel):
    """Individual line item from the invoice."""
    sr_no: Optional[int] = None
    description: Optional[str] = None
    hsn_code: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    discount: Optional[float] = None
    tax_rate: Optional[float] = None
    total_price: Optional[float] = None


class InvoiceData(BaseModel):
    """
    Matches the mentor's Output Template Excel exactly.
    22 required fields + platform + extras.
    """
    # ── Row 2-6: Core identifiers ────────────────────────────────────────
    platform: DocumentPlatform = DocumentPlatform.UNKNOWN
    billing_address: Optional[str] = Field(None, description="Customer billing address")
    shipping_address: Optional[str] = Field(None, description="Customer shipping/delivery address")
    invoice_type: Optional[str] = Field(None, description="Tax Invoice / Bill of Supply / Credit Note etc.")
    order_number: Optional[str] = Field(None, description="Platform order number/ID")
    invoice_number: Optional[str] = Field(None, description="Invoice or bill number")

    # ── Row 7-9: Dates ───────────────────────────────────────────────────
    order_date: Optional[str] = Field(None, description="Date when order was placed")
    invoice_details: Optional[str] = Field(None, description="Invoice description or reference details")
    invoice_date: Optional[str] = Field(None, description="Date on the invoice")

    # ── Row 10-13: Seller identification ─────────────────────────────────
    seller_info: Optional[str] = Field(None, description="Full seller identification text block")
    seller_pan: Optional[str] = Field(None, description="Seller PAN number (10 chars)")
    seller_gst: Optional[str] = Field(None, description="Seller GSTIN (15 chars)")
    fssai_license: Optional[str] = Field(None, description="FSSAI license number (for food platforms)")

    # ── Row 14-18: State codes & supply info ─────────────────────────────
    billing_state_code: Optional[str] = Field(None, description="State code for billing address")
    shipping_state_code: Optional[str] = Field(None, description="State code for shipping address")
    place_of_supply: Optional[str] = Field(None, description="Place of supply for GST")
    place_of_delivery: Optional[str] = Field(None, description="Place of delivery")
    reverse_charge: Optional[str] = Field(None, description="Whether reverse charge applies (Yes/No)")

    # ── Row 19-23: Summary ───────────────────────────────────────────────
    amount_in_words: Optional[str] = Field(None, description="Total amount in words")
    seller_name: Optional[str] = Field(None, description="Seller/merchant name")
    seller_address: Optional[str] = Field(None, description="Seller's registered address")
    total_tax: Optional[float] = Field(None, description="Total tax amount (CGST+SGST or IGST)")
    total_amount: Optional[float] = Field(None, description="Grand total / final amount payable")

    # ── Extra fields (beyond mentor's 22 — bonus) ───────────────────────
    buyer_name: Optional[str] = Field(None, description="Customer/buyer name")
    buyer_phone: Optional[str] = Field(None, description="Customer phone number")
    subtotal: Optional[float] = Field(None, description="Subtotal before tax")
    cgst_rate: Optional[float] = None
    cgst_amount: Optional[float] = None
    sgst_rate: Optional[float] = None
    sgst_amount: Optional[float] = None
    igst_rate: Optional[float] = None
    igst_amount: Optional[float] = None
    discount: Optional[float] = None
    delivery_charge: Optional[float] = None
    packaging_charge: Optional[float] = None
    payment_method: Optional[str] = None
    items: Optional[List[InvoiceItem]] = None

    # ── Metadata ─────────────────────────────────────────────────────────
    tax_validated: bool = False
    validation_warnings: List[str] = []
    fields_extracted: int = 0
    fields_total: int = 22  # mentor's 22 required fields


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OCR METADATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OCRMetadata(BaseModel):
    engine_used: str = "ensemble"
    confidence: float = 0.0
    paddle_regions: int = 0
    easy_regions: int = 0
    merged_regions: int = 0
    processing_time_ms: float = 0.0
    fallback_used: bool = False
    image_preprocessed: bool = True
    pages_processed: int = 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXTRACTION RESPONSE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ExtractionResponse(BaseModel):
    status: ExtractionStatus
    message: str
    document_type: DocumentType
    platform: DocumentPlatform
    confidence_score: float
    extracted_data: InvoiceData
    ocr_metadata: OCRMetadata
    raw_text: Optional[str] = None
    processing_time_seconds: float
    validation_passed: bool = False
    warnings: List[str] = []
    extraction_id: Optional[str] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BANK STATEMENT MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Transaction(BaseModel):
    date: Optional[str] = None
    value_date: Optional[str] = None
    reference_no: Optional[str] = None
    description: Optional[str] = None
    debit: Optional[float] = None
    credit: Optional[float] = None
    balance: Optional[float] = None


class BankStatementData(BaseModel):
    bank_name: Optional[str] = None
    branch: Optional[str] = None
    account_holder: Optional[str] = None
    account_number: Optional[str] = None
    account_type: Optional[str] = None
    ifsc_code: Optional[str] = None
    statement_period_from: Optional[str] = None
    statement_period_to: Optional[str] = None
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    total_debits: Optional[float] = None
    total_credits: Optional[float] = None
    transaction_count: int = 0
    transactions: List[Transaction] = Field(default_factory=list)
    largest_debit: Optional[float] = None
    largest_credit: Optional[float] = None


class BankStatementResponse(BaseModel):
    status: ExtractionStatus
    message: str
    confidence_score: float
    extracted_data: BankStatementData
    ocr_metadata: OCRMetadata
    processing_time_seconds: float
    warnings: List[str] = []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATCH PROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BatchStatus(BaseModel):
    batch_id: str
    total_files: int
    completed: int
    failed: int
    status: str
    results: Optional[List[ExtractionResponse]] = None
    created_at: Optional[str] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLASSIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ClassificationResult(BaseModel):
    document_type: DocumentType
    platform: DocumentPlatform
    confidence: float
    method: str
    all_scores: Dict[str, float]
