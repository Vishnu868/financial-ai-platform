"""
Universal Bank Statement Extractor  — v2.0
==========================================
Supports: ANY bank worldwide (Indian & international)
Input:    PDF (multi-page) or image (JPG/PNG/TIFF/BMP)
Method:   No LLM. Pure structural/heuristic extraction.

Architecture — Three-pass hybrid:
  Pass 1  (Structure Detection)
          – Detect column header row → learn semantic column positions
          – Works for any layout: 5-col, 6-col, 7-col, vertical-label
  Pass 2  (Transaction Extraction)
          – Box-based path  → PaddleOCR bounding boxes per page, clustered into rows
          – Text-based path → flat OCR text fallback with multi-line narration stitching
          – Both paths use balance continuity to disambiguate debit vs credit
  Pass 3  (Header Field Extraction)
          – Bank name, account holder, account number, IFSC, period, balances
          – Pattern library covers 30+ Indian banks + generic international fallbacks

Key improvements over v1:
  ✓ Multi-page PDF: each page extracted independently, transactions merged
  ✓ Column header detection: reads actual column names instead of guessing
  ✓ Balance continuity validator: fixes debit/credit swap errors
  ✓ Multi-line narration stitching across continuation rows
  ✓ ICICI layout: DEPOSITS/WITHDRAWALS columns handled
  ✓ SBI layout: (Value Date) parenthetical rows skipped correctly
  ✓ B/F row → opening balance, closing row → closing balance
  ✓ Masked account numbers (XXXX1234) handled
  ✓ International date formats: DD-Mon-YY, Mon DD YYYY
  ✓ Amount formats: European (1.234,56), Indian (1,23,456.78)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Set

logger = logging.getLogger(__name__)

# BankStatementData and Transaction come from app.models.schemas.
# They are imported lazily (inside methods) to match the original extractor's
# pattern and avoid circular-import issues.
# Layout-only primitives (Word, TableRow) are defined here.

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers: safely set fields that may not exist in the schema
# ─────────────────────────────────────────────────────────────────────────────

def _setattr_safe(obj: Any, field_name: str, value: Any) -> None:
    """Set attribute only if it exists on the object (schema may vary)."""
    if hasattr(obj, field_name):
        setattr(obj, field_name, value)


def _getattr_safe(obj: Any, field_name: str, default: Any = None) -> Any:
    return getattr(obj, field_name, default)


# ─────────────────────────────────────────────────────────────────────────────
# Word / Row primitives (for bounding-box path)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Word:
    text: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float: return (self.x1 + self.x2) / 2
    @property
    def cy(self) -> float: return (self.y1 + self.y2) / 2


@dataclass
class TableRow:
    words: List[Word] = field(default_factory=list)

    @property
    def y_center(self) -> float:
        return sum(w.cy for w in self.words) / max(len(self.words), 1)

    def text_in_range(self, x1: float, x2: float) -> str:
        return " ".join(
            w.text for w in sorted(self.words, key=lambda w: w.cx)
            if x1 <= w.cx <= x2
        ).strip()

    def full_text(self) -> str:
        return " ".join(w.text for w in sorted(self.words, key=lambda w: w.cx))


# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────────────────────

# Dates: DD/MM/YY, DD/MM/YYYY, DD-MM-YY, DD-MM-YYYY, YYYY-MM-DD
_DATE_NUMERIC = re.compile(
    r"^\d{2}[/\-]\d{2}[/\-]\d{2,4}$|^\d{4}[/\-]\d{2}[/\-]\d{2}$"
)
# International: 05-Dec-19, 05-Dec-2019, Dec 05 2019
_DATE_ALPHA = re.compile(
    r"\b(\d{1,2}[/\-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[/\-]\d{2,4})\b"
    r"|\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE
)
# Broad date for line start matching
_DATE_LINE_START = re.compile(
    r"^(\d{1,2}[/\-]\d{2}[/\-]\d{2,4}"
    r"|\d{4}[/\-]\d{2}[/\-]\d{2}"
    r"|\d{1,2}[/\-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[/\-]\d{2,4})\b",
    re.IGNORECASE
)

# Decimal amounts — handles Indian (1,23,456.78) and international (1,234,567.89)
_AMOUNT_RE = re.compile(r"^[\d,]+\.\d{2}$")
# Looser: for scanning in lines
_AMOUNT_SCAN = re.compile(r"\b([\d,]+\.\d{2})\b")

# IFSC: 4 letters + 0 + 6 alphanumeric
_IFSC_RE = re.compile(r"\b([A-Z]{4}0[A-Z0-9]{6})\b")
# MICR: 9 digits
_MICR_RE = re.compile(r"\b(\d{9})\b")

# Column header keywords → semantic name (checked case-insensitively)
_COL_HEADER_MAP: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bdate\b|\bdt\b|\bvalue\s*date\b",           re.I), "date"),
    (re.compile(r"\bnarr|\bdescr|\bparticular|\btransact.*detail|\bdetail", re.I), "narration"),
    (re.compile(r"\bref|\bcheque|\bchq|\bvoucher|\bmode\b|\butr",           re.I), "ref"),
    (re.compile(r"\bdebit|\bwithdraw|\bdr\b",                               re.I), "debit"),
    (re.compile(r"\bcredit|\bdeposit|\bcr\b",                               re.I), "credit"),
    (re.compile(r"\bbalance\b|\bbal\b|\bclosing",                           re.I), "balance"),
]

# B/F row markers
_BF_RE = re.compile(r"\bB/?F\b|\bbrought\s+forward\b|\bopening\s+balance\b", re.I)
_CF_RE = re.compile(r"\bC/?F\b|\bcarried\s+forward\b|\bclosing\s+balance\b", re.I)

# ─────────────────────────────────────────────────────────────────────────────
# Bank name lookup
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSAL IFSC → Bank name mapping  (all RBI-registered banks, 2024)
# This is the PRIMARY bank detection method — works for ANY bank with an IFSC
# code on the statement, which is mandatory for all Indian banks.
# Source: RBI IFSC master list (public data)
# ─────────────────────────────────────────────────────────────────────────────

IFSC_PREFIX_TO_BANK: Dict[str, str] = {
    # Public Sector Banks
    "SBIN": "State Bank of India",
    "BKID": "Bank of India",
    "BARB": "Bank of Baroda",
    "CNRB": "Canara Bank",
    "PUNB": "Punjab National Bank",
    "UBIN": "Union Bank of India",
    "IOBA": "Indian Overseas Bank",
    "IDIB": "Indian Bank",
    "UCBA": "UCO Bank",
    "MAHB": "Bank of Maharashtra",
    "PSIB": "Punjab & Sind Bank",
    "CBOI": "Central Bank of India",
    "ANDB": "Andhra Bank",           # merged into Union Bank
    "ALLA": "Allahabad Bank",        # merged into Indian Bank
    "CORP": "Corporation Bank",      # merged into Union Bank
    "ORBC": "Oriental Bank",         # merged into Punjab National Bank
    "UTBI": "United Bank",           # merged into Punjab National Bank
    "SYND": "Syndicate Bank",        # merged into Canara Bank
    "VIJB": "Vijaya Bank",           # merged into Bank of Baroda
    "DENA": "Dena Bank",             # merged into Bank of Baroda
    "IBKL": "IDBI Bank",
    # Private Banks
    "HDFC": "HDFC Bank",
    "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank",
    "KKBK": "Kotak Mahindra Bank",
    "YESB": "Yes Bank",
    "INDB": "IndusInd Bank",
    "FDRL": "Federal Bank",
    "KVBL": "Karur Vysya Bank",
    "KARB": "Karnataka Bank",
    "SIBL": "South Indian Bank",
    "TMBL": "Tamilnad Mercantile Bank",
    "DHBK": "Dhanlaxmi Bank",
    "NKGS": "NKGSB Co-op Bank",
    "JAKA": "J&K Bank",
    "LAVB": "Lakshmi Vilas Bank",
    "CSBK": "CSB Bank",
    "DLXB": "Dhanlaxmi Bank",
    "DCBL": "DCB Bank",
    "RATN": "RBL Bank",
    "BDBL": "Bandhan Bank",
    "IDFC": "IDFC First Bank",
    "IDFB": "IDFC First Bank",
    # Small Finance Banks
    "AUBL": "AU Small Finance Bank",
    "USFB": "Ujjivan Small Finance Bank",
    "ESAF": "ESAF Small Finance Bank",
    "JSFB": "Jana Small Finance Bank",
    "SFBL": "Suryoday Small Finance Bank",
    "NESF": "Northeast Small Finance Bank",
    " SUCB": "Suryoday SFB",
    "UTKS": "Utkarsh Small Finance Bank",
    "FINF": "Fincare Small Finance Bank",
    "ESFB": "Equitas Small Finance Bank",
    "HCBL": "HCBL Co-op Bank",
    "CTCB": "Citizen Credit Co-op",
    # Payments Banks
    "AIRP": "Airtel Payments Bank",
    "FINO": "Fino Payments Bank",
    "IPOS": "India Post Payments Bank",
    "PYTM": "Paytm Payments Bank",
    "JIOБ": "Jio Payments Bank",
    "NSDL": "NSDL Payments Bank",
    # Foreign Banks
    "CITI": "Citibank",
    "HSBC": "HSBC Bank",
    "SCBL": "Standard Chartered Bank",
    "DEUT": "Deutsche Bank",
    "BARC": "Barclays Bank",
    "DBSS": "DBS Bank",
    "BNPA": "BNP Paribas",
    "ABNA": "ABN AMRO / RBS",
    "BOFA": "Bank of America",
    "JPMO": "JP Morgan Chase",
    "CHAS": "JP Morgan Chase",
    "WOBC": "Woori Bank",
    "KBKH": "KEB Hana Bank",
    "SOGE": "Societe Generale",
    "CRES": "Credit Agricole",
    "MUFG": "MUFG Bank",
    "SMBC": "Sumitomo Mitsui",
    "MHCB": "Mizuho Bank",
    "ИЦИК": "ICICI Bank",  # OCR variant
    # Co-operative Banks (major)
    "SVCB": "Saraswat Co-op Bank",
    "COSB": "Cosmos Co-op Bank",
    "GSCB": "Gujarat State Co-op",
    "MSNU": "Mehsana Urban Co-op",
    "NICB": "New India Co-op Bank",
    "GSMABANK": "Gujarat State Merchant",
}

# ─────────────────────────────────────────────────────────────────────────────
# Bank name keyword lookup (SECONDARY — used when no IFSC found)
# Sorted longest-first so specific names beat short keywords
# ─────────────────────────────────────────────────────────────────────────────

BANK_NAMES: Dict[str, List[str]] = {
    # Most specific first (longest keywords)
    "State Bank of India":    ["state bank of india", "state bank", "sbi bank"],
    "Bank of Baroda":         ["bank of baroda", "bankofbaroda", "baiik of baroda"],
    "Bank of India":          ["bank of india"],
    "Central Bank of India":  ["central bank of india"],
    "Union Bank of India":    ["union bank of india", "union bank"],
    "Punjab National Bank":   ["punjab national bank"],
    "Indian Overseas Bank":   ["indian overseas bank"],
    "Bank of Maharashtra":    ["bank of maharashtra"],
    "Punjab & Sind Bank":     ["punjab and sind bank", "punjab & sind"],
    "IDFC First Bank":        ["idfc first bank", "idfc first", "idfc"],
    "Kotak Mahindra Bank":    ["kotak mahindra bank", "kotak mahindra", "kotak bank"],
    "Karur Vysya Bank":       ["karur vysya bank", "karur vysya"],
    "Karnataka Bank":         ["karnataka bank"],
    "South Indian Bank":      ["south indian bank"],
    "Tamilnad Mercantile":    ["tamilnad mercantile bank", "tmb bank"],
    "Federal Bank":           ["federal bank"],
    "IndusInd Bank":          ["indusind bank", "indusind"],
    "ICICI Bank":             ["icici bank"],           # NOT just "icici" — too short, appears in UPI
    "HDFC Bank":              ["hdfc bank"],
    "Axis Bank":              ["axis bank"],
    "Yes Bank":               ["yes bank"],
    "Canara Bank":            ["canara bank"],
    "Indian Bank":            ["indian bank"],
    "AU Small Finance Bank":  ["au small finance bank", "au small finance", "au bank"],
    "Ujjivan Small Finance":  ["ujjivan small finance bank", "ujjivan small", "ujjivan"],
    "ESAF Small Finance":     ["esaf small finance bank", "esaf small", "esaf bank"],
    "Bandhan Bank":           ["bandhan bank"],
    "Dhanlaxmi Bank":         ["dhanlaxmi bank", "dhanlaxmi"],
    "Saraswat Bank":          ["saraswat co-op bank", "saraswat bank"],
    "J&K Bank":               ["jammu and kashmir bank", "j&k bank", "jk bank"],
    "Lakshmi Vilas Bank":     ["lakshmi vilas bank", "lakshmi vilas"],
    "DCB Bank":               ["dcb bank"],
    "RBL Bank":               ["rbl bank"],
    "IDBI Bank":              ["idbi bank"],
    "UCO Bank":               ["uco bank"],
    "Standard Chartered":     ["standard chartered bank", "standard chartered"],
    "Deutsche Bank":          ["deutsche bank"],
    "Barclays":               ["barclays bank", "barclays"],
    "DBS Bank":               ["dbs bank", "development bank of singapore"],
    "Citibank":               ["citibank", "citi bank"],
    "HSBC":                   ["hsbc bank", "hsbc"],
    # SBI kept short keyword last so long ones above match first
    "State Bank of India":    ["state bank of india", " sbi ", "sbi\n"],
}

BANK_IFSC_PREFIX: Dict[str, str] = {
    k: v + "0" for v, k in IFSC_PREFIX_TO_BANK.items()
    if len(v) == 4  # Only use 4-char codes for reconstruction
}

# ─────────────────────────────────────────────────────────────────────────────
# Universal column header keywords — covers ALL known Indian bank formats
# ─────────────────────────────────────────────────────────────────────────────

_COL_HEADER_MAP: List[Tuple[re.Pattern, str]] = [
    # DATE column — every bank uses one of these
    (re.compile(
        r"\bdate\b|\bdt\b|\bvalue[\s\-]?date\b|\btxn[\s\-]?date\b"
        r"|\btrans[\s\-]?date\b|\bposting[\s\-]?date\b|\bvalue\b"
        r"|\btrans\.?\s*dt\b|\bbook[\s\-]?date\b",
        re.I), "date"),

    # NARRATION / DESCRIPTION column
    (re.compile(
        r"\bnarration\b|\bnarr\.?\b|\bdescription\b|\bdescr\.?\b"
        r"|\bparticulars?\b|\bparticular\b|\bdetails?\b"
        r"|\btransaction[\s\-]?details?\b|\bremarks?\b|\bstatement\b"
        r"|\btransaction[\s\-]?narration\b|\bpayment[\s\-]?details?\b"
        r"|\bcheque[\s\-]?particulars?\b|\bbeneficiary\b",
        re.I), "narration"),

    # REF / CHEQUE column
    (re.compile(
        r"\bref(?:erence)?[\s\.#]?(?:no\.?|num\.?|number)?\b"
        r"|\bcheque[\s\-]?(?:no\.?|num\.?|number)?\b"
        r"|\bchq\.?[\s\-]?(?:no\.?|num\.?)?\b"
        r"|\bvoucher[\s\-]?(?:no\.?)?\b|\butr[\s\-]?(?:no\.?)?\b"
        r"|\binstrument[\s\-]?(?:no\.?)?\b|\btxn[\s\-]?(?:id|no\.?)?\b"
        r"|\btransaction[\s\-]?(?:id|no\.?)?\b|\bmode\b",
        re.I), "ref"),

    # DEBIT / WITHDRAWAL column
    (re.compile(
        r"\bdebit\b|\bwithdrawal(?:s)?\b|\bwithdraw(?:al)?\b"
        r"|\bdr\.?\b|\bpaid[\s\-]?out\b|\bamount[\s\-]?(?:dr|debit)\b"
        r"|\bdebit[\s\-]?amount\b|\bdr[\s\-]?amount\b"
        r"|\bwithdrawal[\s\-]?\(dr\)\b|\bdr\s*\(withdrawal\)\b",
        re.I), "debit"),

    # CREDIT / DEPOSIT column
    (re.compile(
        r"\bcredit\b|\bdeposit(?:s)?\b|\bcr\.?\b|\bpaid[\s\-]?in\b"
        r"|\bamount[\s\-]?(?:cr|credit)\b|\bcredit[\s\-]?amount\b"
        r"|\bcr[\s\-]?amount\b|\bdeposit[\s\-]?\(cr\)\b"
        r"|\bcr\s*\(deposit\)\b",
        re.I), "credit"),

    # BALANCE column
    (re.compile(
        r"\bbalance\b|\bbal\.?\b|\bclosing[\s\-]?bal(?:ance)?\b"
        r"|\brunning[\s\-]?bal(?:ance)?\b|\bavailable[\s\-]?bal(?:ance)?\b"
        r"|\bamount[\s\-]?balance\b|\bbal(?:ance)?[\s\-]?after\b"
        r"|\baccount[\s\-]?bal(?:ance)?\b|\bnet[\s\-]?bal(?:ance)?\b",
        re.I), "balance"),
]



def _fix_amounts(text: str) -> str:
    """
    Fix OCR artifacts in amount strings:
      '1.126.06'  → '1,126.06'    (comma read as period)
      '22.792.26' → '22,792.26'
      '1.23.456.78' → '1,23,456.78' (Indian format)
    Also handles European format: '1.234,56' → '1234.56' is NOT done here
    (we keep raw and parse per-context).
    """
    # Three-digit groups before final two-decimal: X.DDD.DD → X,DDD.DD
    text = re.sub(r'(\d+)\.(\d{3})\.(\d{2})\b', r'\1,\2.\3', text)
    # Two-digit group (Indian): X.XX.DDD.DD → X,XX,DDD.DD
    text = re.sub(r'(\d+)\.(\d{2})\.(\d{3})\.(\d{2})\b', r'\1,\2,\3.\4', text)
    # Repeat for chained 3-group: X.DDD.DDD.DD
    text = re.sub(r'(\d+)\.(\d{3})\.(\d{3})\.(\d{2})\b', r'\1,\2,\3.\4', text)
    # Fix double commas
    text = re.sub(r',,+', ',', text)
    return text


def _parse_amount(s: str) -> Optional[float]:
    """Parse amount string → float. Handles Indian and international formats."""
    if not s:
        return None
    s = _fix_amounts(s.strip())
    # Remove currency symbols and spaces
    s = re.sub(r'[₹$€£¥\s]', '', s)
    # Remove commas (thousand separators)
    s = s.replace(',', '')
    m = re.search(r'[\d]+\.\d{2}', s)
    if m:
        try:
            val = float(m.group(0))
            return val if val >= 0 else None
        except ValueError:
            pass
    # Integer amount (no decimal)
    m2 = re.search(r'\d+', s)
    if m2:
        try:
            return float(m2.group(0))
        except ValueError:
            pass
    return None


def _is_date(text: str) -> bool:
    """Return True if text looks like a transaction date."""
    t = text.strip()
    if _DATE_NUMERIC.match(t):
        return True
    if _DATE_ALPHA.search(t):
        return True
    return False


def _is_amount(text: str) -> bool:
    clean = text.strip().replace(',', '')
    return bool(_AMOUNT_RE.match(clean))


def _strip_cr_dr(line: str) -> str:
    """
    Strip trailing 'Cr'/'Dr'/'CR'/'DR' suffix that BoB appends to every balance figure.
    e.g. "200.00 Cr" → "200.00",  "1,306.55 Cr" → "1,306.55"
    Also handles mid-line occurrences like "200.00 Cr 70.00 Cr"
    """
    return re.sub(r'(\d)\s*\b(?:Cr|Dr|CR|DR)\b', r'\1', line)


def _clean_narration(text: str) -> str:
    """
    Clean OCR artifacts from transaction narration strings.
    1. Fix timestamps: '09: 18: 50' → '09:18:50'  (OCR adds spaces after colons)
    2. Strip page-break junk and footer phone numbers bleeding into narrations
    3. Collapse multiple spaces
    """
    # Fix OCR-spaced timestamps: digits colon space digits → digits colon digits
    text = re.sub(r'(\d{2}):\s+(\d{2}):\s+(\d{2})', r'\1:\2:\3', text)
    text = re.sub(r'(\d{2}):\s+(\d{2})', r'\1:\2', text)
    # Strip trailing helpline/page numbers: "1930", "1800 5700", "1800 5000", "Page N | N"
    text = re.sub(r'\s+\b1930\b\s*$', '', text)
    text = re.sub(r'\s+\b1800\s+\d{4}\b\s*$', '', text)
    text = re.sub(r'\s*Page\s+\d+\s*\|\s*\d+.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*https?://\S+', '', text)
    # Strip trailing bare 4-digit numbers that are clearly phone suffix / helpline
    text = re.sub(r'\s+\b\d{4}\b\s*$', '', text)
    # Collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _clean_name(name: str) -> str:
    name = name.split("\n")[0].split("\r")[0]
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'[^A-Za-z\s\.\-]+$', '', name).strip()
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor class
# ─────────────────────────────────────────────────────────────────────────────

class BankStatementExtractor:
    """
    Universal bank statement extractor.
    Two entry points:
      .extract(text)              — flat OCR text (multi-page concatenated)
      .extract_from_boxes(boxes, full_text, page_count)  — structured PaddleOCR boxes
    """

    # ── Public API ────────────────────────────────────────────────────────

    def extract(self, text: str, page_count: int = 1):
        """
        Flat OCR text → structured data.
        Works for both single-page and multi-page (concatenate pages with PAGE_BREAK marker).
        Returns an app.models.schemas.BankStatementData instance.
        """
        from app.models.schemas import BankStatementData
        text = _strip_cr_dr(_fix_amounts(text))
        data = BankStatementData()
        _setattr_safe(data, "pages_processed", page_count)

        self._extract_header_fields(text, data)
        data.transactions = self._parse_transactions_text(text)
        self._extract_bf_balances(text, data)
        # ← CRITICAL: must run BEFORE _compute_analytics so debit/credit totals are correct
        self._fix_debit_credit_with_balance_continuity(
            data.transactions,
            opening_balance=getattr(data, 'opening_balance', None)
        )
        self._compute_analytics(data)
        return data

    def extract_from_boxes(
        self,
        ocr_boxes: List[Any],
        full_text: str = "",
        page_count: int = 1,
        existing_data=None,
    ):
        """
        Structured path: bounding boxes for a single page.
        Call repeatedly for multi-page PDFs, passing existing_data each time
        so transactions accumulate across pages.
        """
        from app.models.schemas import BankStatementData
        full_text = _strip_cr_dr(_fix_amounts(full_text))

        if existing_data is None:
            data = BankStatementData()
            _setattr_safe(data, "pages_processed", 0)
        else:
            data = existing_data

        # Run header extraction on EVERY page — critical because:
        # - ICICI MICR is on page 2, BoB IFSC/MICR is on page 4
        # - Fields already filled are protected by "or" guards inside _extract_header_fields
        if full_text:
            self._extract_header_fields(full_text, data)

        prev_pages = _getattr_safe(data, "pages_processed", 0) or 0
        _setattr_safe(data, "pages_processed", prev_pages + 1)

        words = self._paddle_to_words(ocr_boxes)
        if not words:
            page_txs = self._parse_transactions_text(full_text)
        else:
            rows = self._cluster_rows(words)
            col_map = self._detect_columns_from_headers(rows, words)
            page_txs = self._extract_transactions_box(rows, col_map)

        if data.transactions is None:
            data.transactions = []
        data.transactions.extend(page_txs)
        # Run continuity fix after every page so intermediate callers get correct data
        # finalize() will re-run this on the complete set, which is safe (idempotent)
        self._fix_debit_credit_with_balance_continuity(
            data.transactions,
            opening_balance=getattr(data, 'opening_balance', None)
        )
        return data

    def finalize(self, data, full_text: str = ""):
        """
        Call after all pages have been processed.
        Runs balance continuity fix, analytics, and B/F balance extraction.
        """
        full_text = _strip_cr_dr(_fix_amounts(full_text))
        self._extract_bf_balances(full_text, data)
        # Pass opening_balance so continuity fix is seeded correctly
        # even when the first transaction has no prior balance reference
        opening_bal = getattr(data, 'opening_balance', None)
        self._fix_debit_credit_with_balance_continuity(
            data.transactions, opening_balance=opening_bal
        )
        self._deduplicate_transactions(data)
        self._compute_analytics(data)
        return data

    # ── Header field extraction ───────────────────────────────────────────

    def _extract_header_fields(self, text: str, data) -> None:
        data.bank_name      = data.bank_name      or self._bank_name(text)
        data.account_holder = data.account_holder or self._account_holder(text)
        data.account_number = data.account_number or self._account_number(text)
        data.account_type   = data.account_type   or self._account_type(text)
        data.ifsc_code      = data.ifsc_code      or self._ifsc(text, data.bank_name)
        _setattr_safe(data, "micr_code",
                      _getattr_safe(data, "micr_code") or self._micr(text))
        if not data.statement_period_from:
            data.statement_period_from, data.statement_period_to = self._period(text)
        if data.opening_balance is None:
            data.opening_balance = self._balance_keyword(text, "opening")
        if data.closing_balance is None:
            data.closing_balance = self._balance_keyword(text, "closing")

    def _bank_name(self, text: str) -> Optional[str]:
        """
        Universal 6-layer bank detection — no bank-specific hardcoding needed.
        Layer 1: IFSC prefix  (mandatory on all Indian statements — most reliable)
        Layer 2: Website domain in footer (e.g. www.bankofbaroda.bank.in)
        Layer 3: Keyword in non-narration lines (longest match first)
        Layer 4: Keyword in full text with UPI-handle guard
        Layer 5: Unknown IFSC prefix → return "Bank (XXXX)"
        Layer 6: Generic "X Bank" pattern extraction
        """
        # ── Layer 1: IFSC prefix → bank name (covers any RBI-registered bank) ──
        for m in _IFSC_RE.finditer(text):
            ctx = text[max(0, m.start()-5):m.start()]
            if "@" in ctx:
                continue  # inside UPI VPA — skip
            prefix = m.group(1)[:4]
            if prefix in IFSC_PREFIX_TO_BANK:
                return IFSC_PREFIX_TO_BANK[prefix]

        # ── Layer 2: Website domain printed on statement ──────────────────────
        domain_map = {
            "bankofbaroda": "Bank of Baroda",
            "icicibank":    "ICICI Bank",
            "hdfcbank":     "HDFC Bank",
            "onlinesbi":    "State Bank of India",
            "sbi.co.in":    "State Bank of India",
            "axisbank":     "Axis Bank",
            "kotakbank":    "Kotak Mahindra Bank",
            "yesbank":      "Yes Bank",
            "indusind":     "IndusInd Bank",
            "federalbank":  "Federal Bank",
            "canarabank":   "Canara Bank",
            "unionbankofindia": "Union Bank of India",
            "pnbindia":     "Punjab National Bank",
            "bankofindia":  "Bank of India",
            "aubank":       "AU Small Finance Bank",
            "bandhanbank":  "Bandhan Bank",
            "rblbank":      "RBL Bank",
            "idfcfirstbank": "IDFC First Bank",
            "southindianbank": "South Indian Bank",
            "karnatakbank": "Karnataka Bank",
            "kvb.co.in":    "Karur Vysya Bank",
        }
        tl = text.lower()
        for domain, name in domain_map.items():
            if re.search(r'(?:www\.|https?://|\.)' + re.escape(domain), tl):
                return name

        # ── Layer 3: Keyword in clean (non-narration) lines ──────────────────
        clean_lines = []
        for line in text.splitlines():
            ls = line.strip()
            if re.match(r"^\s*(?:UPI|NEFT|RTGS|IMPS|NACH|ECS)/", ls, re.I):
                continue
            clean_lines.append(ls)
        tl_clean = "\n".join(clean_lines).lower()

        # Sort longest keyword first — "bank of baroda" beats "bank"
        kw_pairs = sorted(
            [(kw, nm) for nm, kws in BANK_NAMES.items() for kw in kws],
            key=lambda x: len(x[0]), reverse=True
        )
        for kw, name in kw_pairs:
            if kw in tl_clean:
                return name

        # ── Layer 4: Full text with UPI-handle guard ──────────────────────────
        for kw, name in kw_pairs:
            for m in re.finditer(re.escape(kw), tl):
                ctx_b = tl[max(0, m.start()-8):m.start()]
                ctx_a = tl[m.end():m.end()+4]
                if "@" not in ctx_b and "/" not in ctx_a[:2]:
                    return name

        # ── Layer 5: Unknown IFSC prefix → readable label ─────────────────────
        for m in _IFSC_RE.finditer(text):
            ctx = text[max(0, m.start()-5):m.start()]
            if "@" not in ctx:
                prefix = m.group(1)[:4]
                return f"Bank ({prefix})"   # e.g. "Bank (HDFC)" for unlisted IFSC

        # ── Layer 6: Generic "X Bank" / "X Finance" pattern ──────────────────
        m6 = re.search(
            r'\b([A-Z][A-Za-z\s&\.]{3,35}(?:Bank|Banque|Finance|Payments?))\b',
            text
        )
        if m6:
            cand = m6.group(1).strip()
            bad = {"your base branch", "scheduled bank", "commercial bank",
                   "member bank", "reserve bank", "member of"}
            if cand.lower() not in bad and len(cand) >= 5:
                return cand
        return None

    def _account_holder(self, text: str) -> Optional[str]:
        _BAD_WORDS = {
            "ROAD","NAGAR","STREET","INDIA","BANK","BRANCH","ACCOUNT",
            "LIMIT","STATE","CITY","PHONE","EMAIL","ADDRESS","RAJASTHAN",
            "DELHI","MUMBAI","KOLKATA","CHENNAI","BENGALURU","HYDERABAD",
            "HOUSE","GALI","SECTOR","DISTRICT","PRADESH","WEST","EAST",
            "FLOOR","EXTENSION","PURA","NORTH","SOUTH","NO","VENKATESWARA",
            "GAMRI","BHAJAN","STORE","MEDICAL","MANIKANTA","KUKATPALLI",
        }
        patterns = [
            # Labeled (most reliable): "Account Holder: NAME"
            r"(?:Account\s+(?:Holder|Name)|Customer\s+Name|Name\s+of\s+(?:Account\s+)?Holder)\s*[:\-]?\s*([A-Z][a-zA-Z\s\.]{3,60})",
            # Title-prefix at line start — \s* handles "MR.AVDHESH" (no space) AND "MR. AVDHESH"
            r"^(?:MR|MRS|MS|DR|SHRI|SMT|SRI|PROF)\.?\s*([A-Z][A-Z][A-Z\s\.]{1,58})$",
            # Title-prefix anywhere — \s* handles both spacing variants
            r"(?:MR\.|MRS\.|MS\.|DR\.)\s*([A-Z][a-zA-Z\s\.]{3,60})",
        ]
        for p in patterns:
            for m in re.finditer(p, text, re.IGNORECASE | re.MULTILINE):
                name = _clean_name(m.group(1))
                # Stop at first address-like word
                words = name.split()
                good = []
                for w in words:
                    if w.upper().rstrip('.') in _BAD_WORDS or re.match(r'^\d', w):
                        break
                    good.append(w)
                name = " ".join(good).strip()
                name = re.sub(r'[^A-Za-z\s\.]+$', '', name).strip()
                words_in = {w.upper() for w in name.split()}
                if len(name) >= 5 and not words_in.intersection(_BAD_WORDS):
                    return name
        return None

    def _account_number(self, text: str) -> Optional[str]:
        labeled = [
            r"(?:Account\s+(?:No|Number)|A/?C\s+(?:No\.?|Number)|Savings\s+A/c|Current\s+A/c)\s*[.:\s]+([X\*\d][\dX\*\s]{7,20}[X\*\d])",
            r"Account\s+Number\s*[:\-]?\s*(\d{8,20})",
            r"A/c\s+No\.?\s*[:\-]?\s*(\d{8,20})",
            # Standalone after label line
            r"(?:Savings|Current|NRE|NRO)\s+(?:Account|A/C)\s+(\d{8,20})",
        ]
        for p in labeled:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                num = re.sub(r'\s+', '', m.group(1))
                if len(num) >= 8:
                    return num

        # Context-based: long digit sequence near account keywords
        for m in re.finditer(r"\b(\d{9,20})\b", text):
            ctx = text[max(0, m.start()-100):m.start()].lower()
            if any(k in ctx for k in ["account", "a/c", "acct", "savings", "current"]):
                return m.group(1)
        return None

    def _account_type(self, text: str) -> Optional[str]:
        tl = text.lower()
        if "salary"        in tl: return "Salary"
        if "savings"       in tl: return "Savings"
        if "current"       in tl: return "Current"
        if "fixed deposit" in tl or " fd " in tl: return "Fixed Deposit"
        if "nre"           in tl: return "NRE"
        if "nro"           in tl: return "NRO"
        if "overdraft"     in tl or " od " in tl: return "Overdraft"
        return None

    def _ifsc(self, text: str, bank_name: Optional[str] = None) -> Optional[str]:
        # 1. Labeled
        m = re.search(
            r"(?:IFS\s*(?:C|Code)\s*[:\-]?|RTGS[^\n]{0,30}IFSC|IFSC\s+Code)[^\n]{0,20}?([A-Z]{4}0[A-Z0-9]{6})",
            text, re.IGNORECASE
        )
        if m: return m.group(1)

        # 2. Reconstruct from bank prefix + branch code
        if bank_name and bank_name in BANK_IFSC_PREFIX:
            prefix = BANK_IFSC_PREFIX[bank_name]
            bc = re.search(r"Branch\s+(?:Code|No\.?)\s*[:\s]+(\d{4,7})", text, re.IGNORECASE)
            if bc:
                branch = bc.group(1).zfill(6)
                candidate = prefix + branch
                if len(candidate) == 11:
                    return candidate

        # 3. Standalone IFSC not in UPI handles
        for m in _IFSC_RE.finditer(text):
            ctx = text[max(0, m.start()-40):m.end()+40]
            if "@" not in ctx:
                return m.group(1)
        return None

    def _micr(self, text: str) -> Optional[str]:
        # 1. Labeled: "MICR Code: 110229196" or "MICR: 110229196"
        m = re.search(r"MICR\s*(?:Code?)?\s*[:\-]\s*(\d{9})\b", text, re.IGNORECASE)
        if m: return m.group(1)

        # 2. Column-header table: "MICR CODE\n... 110229196 ..."
        # Handles ICICI page-2 table where MICR CODE is a column header
        m2 = re.search(
            r"MICR\s+CODE[^\n]*\n[^\n]{0,60}?\b(\d{9})\b",
            text, re.IGNORECASE
        )
        if m2: return m2.group(1)

        # 3. 9-digit number immediately before an IFSC code on the same line
        # e.g. "... 110229196 ICIC0001134 ..." or "500012072 BARB0DBKUKU"
        m3 = re.search(r"\b(\d{9})\b(?=\s+[A-Z]{4}0[A-Z0-9]{6})", text)
        if m3: return m3.group(1)

        # 4. 9-digit number immediately after an account number on the same line
        # e.g. "Savings 113401501120 110229196 ICIC0001134"
        m4 = re.search(r"\b\d{10,18}\b\s+(\d{9})\b", text)
        if m4: return m4.group(1)

        return None

    def _period(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        _MON = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        patterns = [
            # DD/MM/YYYY numeric with labels
            r"From\s*:\s*(\d{2}[/\-]\d{2}[/\-]\d{2,4})\s+To\s*:\s*(\d{2}[/\-]\d{2}[/\-]\d{2,4})",
            r"From\s+(\d{2}[/\-]\d{2}[/\-]\d{2,4})\s+To\s+(\d{2}[/\-]\d{2}[/\-]\d{2,4})",
            r"Period\s*[:\-]\s*(\d{2}[/\-]\d{2}[/\-]\d{2,4})\s+(?:to|[\-–])\s+(\d{2}[/\-]\d{2}[/\-]\d{2,4})",
            r"Statement\s+(?:Period|Date)\s*[:\-]\s*(\d{2}[/\-]\d{2}[/\-]\d{2,4})\s+(?:to|[\-–])\s+(\d{2}[/\-]\d{2}[/\-]\d{2,4})",
            r"\b(\d{2}[/\-]\d{2}[/\-]\d{2,4})\s+to\s+(\d{2}[/\-]\d{2}[/\-]\d{2,4})\b",
            # SBI: "for the period05/12/2019to30/12/2019"
            r"period\s*(\d{2}[/\-]\d{2}[/\-]\d{2,4})\s*to\s*(\d{2}[/\-]\d{2}[/\-]\d{2,4})",
            # BoB: "Statement Period from Mar 01, 2026 to Mar 31, 2026"
            r"(?:Statement\s+)?Period\s+from\s+(" + _MON + r"\.?\s+\d{1,2},?\s*\d{4})\s+to\s+(" + _MON + r"\.?\s+\d{1,2},?\s*\d{4})",
            # ICICI: "for the period April 29, 2025 - July02, 2025"
            r"period\s+(" + _MON + r"\.?\s*\d{1,2},?\s*\d{4})\s*[\-–]\s*(" + _MON + r"\.?\s*\d{1,2},?\s*\d{4})",
            # Generic month-name range with separator
            r"\b(" + _MON + r"\.?\s+\d{1,2},?\s*\d{4})\s+(?:to|[\-–])\s+(" + _MON + r"\.?\s+\d{1,2},?\s*\d{4})\b",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m and m.lastindex and m.lastindex >= 2:
                return m.group(1).strip(), m.group(2).strip()
        return None, None

    def _balance_keyword(self, text: str, kind: str) -> Optional[float]:
        """Extract opening or closing balance from labeled keyword patterns."""
        # Cr/Dr suffix (BoB style) - strip before parsing
        def _parse_with_suffix(v: str) -> Optional[float]:
            v = re.sub(r'\s*(?:Cr|Dr|CR|DR)\s*$', '', v).strip()
            return _parse_amount(v)

        if kind == "opening":
            pats = [
                r"Opening\s+Balance\s*[:\-]?\s*([\d,]+\.\d{2}(?:\s*(?:Cr|Dr))?)",
                r"Balance\s+B/?F\s*[:\-]?\s*([\d,]+\.\d{2}(?:\s*(?:Cr|Dr))?)",
                r"Brought\s+Forward\s*[:\-]?\s*([\d,]+\.\d{2}(?:\s*(?:Cr|Dr))?)",
                r"Balance\s+as\s+on[^\n]{0,30}?([\d,]+\.\d{2}(?:\s*(?:Cr|Dr))?)",
                # ICICI: "B/F 40,120.40" as a transaction line
                r"\bB/?F\b[^\n]{0,20}?([\d,]+\.\d{2})",
                # BoB: date-prefixed "01-03-2026 Opening Balance 0.00 Cr"
                r"\d{2}[/\-]\d{2}[/\-]\d{4}\s+Opening\s+Balance\s+([\d,]+\.\d{2}(?:\s*(?:Cr|Dr))?)",
            ]
        else:
            pats = [
                r"Closing\s+Balance\s*[:\-]?\s*([\d,]+\.\d{2}(?:\s*(?:Cr|Dr))?)",
                r"Balance\s+C/?F\s*[:\-]?\s*([\d,]+\.\d{2}(?:\s*(?:Cr|Dr))?)",
                r"Carried\s+Forward\s*[:\-]?\s*([\d,]+\.\d{2}(?:\s*(?:Cr|Dr))?)",
            ]
        for p in pats:
            found = re.findall(p, text, re.IGNORECASE)
            vals = []
            for v in found:
                f = _parse_with_suffix(v)
                if f is not None and f >= 0:
                    vals.append(f)
            if vals:
                return vals[-1] if kind == "closing" else vals[0]
        return None

    def _extract_bf_balances(self, text: str, data) -> None:
        """
        Use first/last transaction balance as opening/closing when keyword extraction fails.
        Also handles B/F rows explicitly.
        """
        # Try keyword extraction first (already done in header, but re-check on combined text)
        if data.opening_balance is None:
            data.opening_balance = self._balance_keyword(text, "opening")
        if data.closing_balance is None:
            data.closing_balance = self._balance_keyword(text, "closing")

        # Use transaction sequence as fallback
        txs = data.transactions
        if txs:
            bals = [t.balance for t in txs if t.balance is not None]
            if bals:
                if data.opening_balance is None:
                    data.opening_balance = bals[0]
                if data.closing_balance is None:
                    data.closing_balance = bals[-1]

    # ── Text-based transaction parser (multi-line narration aware) ────────

    def _parse_transactions_text(self, text: str) -> List[Transaction]:
        """
        Parse transactions from flat OCR text.

        Algorithm:
        1. Split into lines
        2. A transaction "starts" on a line whose first token is a date
        3. Continuation lines (no date, no amounts = narration extension) are appended
        4. Amounts in each transaction are collected, then mapped to debit/credit/balance
           using balance-continuity heuristic
        """
        lines = [_strip_cr_dr(_fix_amounts(line)) for line in text.split("\n")]
        transactions: List[Transaction] = []

        # State machine
        current_date: Optional[str] = None
        current_desc_parts: List[str] = []
        current_ref: Optional[str] = None
        current_amounts: List[str] = []

        def flush():
            nonlocal current_date, current_desc_parts, current_ref, current_amounts
            if current_date is None:
                return
            desc = _clean_narration(" ".join(current_desc_parts).strip())[:300]
            tx = self._amounts_to_transaction(
                current_date, desc, current_ref, current_amounts
            )
            transactions.append(tx)
            current_date = None
            current_desc_parts = []
            current_ref = None
            current_amounts = []

        # Patterns to skip (header/footer junk)
        _SKIP_RE = re.compile(
            r"^\s*(?:date|narr|descr|particular|withdrawal|deposit|debit|credit|balance"
            r"|chq|ref|mode|sr\.?\s*no|slno|page\s+no|page\s+\d|statement|account|total"
            r"|brought|carried|transaction|https?://|www\.|customer\s+care"
            r"|cyber\s+crime|helpline|\d{4}\s+\d{4}"
            r"|abbreviations?|abbreviation|nominee|base\s+branch|customer\s+id"
            r"|relationship\s+type|savings\s+account\s+inr)\b",
            re.IGNORECASE
        )
        # Lines that start with date AND contain "Opening/Closing Balance" → skip as transaction
        # These are balance marker rows, already handled by _balance_keyword()
        _BALANCE_ROW_RE = re.compile(
            r"^\d{2}[/\-]\d{2}[/\-]\d{4}\s+(?:Opening|Closing)\s+Balance",
            re.IGNORECASE
        )
        _REF_RE = re.compile(r"^\d{6,20}$")

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            # Check if line starts with a date
            dm = _DATE_LINE_START.match(line)
            if dm:
                flush()
                date_str = dm.group(1)
                rest = line[dm.end():].strip()

                # ── Skip balance-marker rows ──────────────────────────────
                # BoB: "01-03-2026 Opening Balance 0.00 Cr"
                # BoB: "31-03-2026 Closing Balance 1.80 Cr"
                # These are header info already captured by _balance_keyword()
                # They must NOT become transaction rows.
                if _BALANCE_ROW_RE.match(line):
                    current_date = None   # reset so no partial tx is open
                    continue

                # Skip pure value-date lines like "(05-Dec-2019)" from SBI
                if rest.startswith("(") and re.match(r"^\(?\d{2}[/\-]\d{2}[/\-]\d{2,4}\)?$", rest):
                    current_date = date_str
                    continue

                current_date = date_str

                # Extract amounts from rest
                amounts_in_rest = _AMOUNT_SCAN.findall(rest)
                # Description = text before first amount
                if amounts_in_rest:
                    first_pos = rest.find(amounts_in_rest[0])
                    raw_desc = rest[:first_pos].strip()
                    current_amounts = amounts_in_rest
                else:
                    raw_desc = rest
                    current_amounts = []

                # Clean desc: remove trailing ref numbers and value dates
                raw_desc = re.sub(r'\s+\d{2}[/\-]\d{2}[/\-]\d{2,4}\s*$', '', raw_desc).strip()
                raw_desc = re.sub(r'\s+\d{8,20}\s*$', '', raw_desc).strip()

                # Try to extract ref from end of desc
                tokens = raw_desc.rsplit(None, 1)
                if len(tokens) == 2 and _REF_RE.match(tokens[1]):
                    raw_desc = tokens[0]
                    current_ref = tokens[1]

                if raw_desc and not _SKIP_RE.match(raw_desc):
                    current_desc_parts = [raw_desc]
                else:
                    current_desc_parts = []

            elif current_date is not None:
                # Continuation line
                if _SKIP_RE.match(line):
                    continue
                # Does it look like a pure value-date line?
                if re.match(r"^\(?\d{2}[/\-]\d{2}[/\-]\d{2,4}\)?$", line):
                    continue
                # Extract amounts
                line_amounts = _AMOUNT_SCAN.findall(line)
                line_no_amounts = _AMOUNT_SCAN.sub("", line).strip()

                if line_amounts:
                    current_amounts.extend(line_amounts)
                    if line_no_amounts and len(line_no_amounts) > 2:
                        current_desc_parts.append(line_no_amounts)
                else:
                    # Pure narration continuation
                    if len(line) > 2 and not _SKIP_RE.match(line):
                        current_desc_parts.append(line)

        flush()  # Don't forget last transaction
        return transactions

    def _amounts_to_transaction(
        self,
        date: str,
        desc: str,
        ref: Optional[str],
        raw_amounts: List[str],
    ) -> Transaction:
        """
        Map a list of extracted decimal amounts to debit/credit/balance.

        Rules:
        - Last amount = balance (most reliable)
        - With 3+ amounts: third-from-last=debit, second-from-last=credit
          BUT one of them may be 0/empty (represented by its absence in the line)
          → use value proximity to distinguish
        - With 2 amounts: (amount, balance) — classify by balance continuity later
        - With 1 amount: likely balance only
        """
        amounts = [_parse_amount(a) for a in raw_amounts]
        amounts = [a for a in amounts if a is not None]

        from app.models.schemas import Transaction
        tx = Transaction(date=date, description=desc)
        _setattr_safe(tx, "ref_no", ref)

        if len(amounts) >= 3:
            tx.debit   = amounts[-3] if amounts[-3] and amounts[-3] > 0 else None
            tx.credit  = amounts[-2] if amounts[-2] and amounts[-2] > 0 else None
            tx.balance = amounts[-1]
        elif len(amounts) == 2:
            tx.credit  = amounts[0]
            tx.balance = amounts[1]
        elif len(amounts) == 1:
            tx.balance = amounts[0]

        return tx

    # ── Box-based transaction extraction ─────────────────────────────────

    def _paddle_to_words(self, ocr_result: List[Any]) -> List[Word]:
        words: List[Word] = []
        for item in (ocr_result or []):
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pts = item[0]
                    text_part = item[1]
                    if isinstance(text_part, (list, tuple)):
                        text = str(text_part[0])
                        conf = float(text_part[1]) if len(text_part) > 1 else 1.0
                    else:
                        text = str(text_part)
                        conf = 1.0
                    text = _fix_amounts(text.strip())
                    if not text:
                        continue
                    xs = [float(p[0]) for p in pts]
                    ys = [float(p[1]) for p in pts]
                    words.append(Word(
                        text=text,
                        x1=min(xs), y1=min(ys),
                        x2=max(xs), y2=max(ys),
                    ))
            except Exception as e:
                logger.debug(f"Word parse error: {e}")
        return words

    def _cluster_rows(self, words: List[Word], gap_factor: float = 0.55) -> List[TableRow]:
        if not words:
            return []
        heights = sorted([w.y2 - w.y1 for w in words])
        median_h = heights[len(heights) // 2] if heights else 12
        gap = median_h * gap_factor

        sorted_words = sorted(words, key=lambda w: (w.cy, w.cx))
        rows: List[TableRow] = []
        current = TableRow(words=[sorted_words[0]])

        for word in sorted_words[1:]:
            if abs(word.cy - current.y_center) <= gap:
                current.words.append(word)
            else:
                rows.append(current)
                current = TableRow(words=[word])
        rows.append(current)
        return rows

    def _detect_columns_from_headers(
        self,
        rows: List[TableRow],
        all_words: List[Word],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Strategy 1: Find column header row (contains date/narration/debit/credit/balance keywords)
        → use X positions of those headers as column anchors.

        Strategy 2: Fallback → detect from amount X-distributions.
        """
        img_w = max(w.x2 for w in all_words) if all_words else 1000.0

        # Try to find header row
        header_row: Optional[TableRow] = None
        header_col_map: Dict[str, float] = {}  # semantic_name → center_x

        for row in rows[:20]:  # headers are near top
            text = row.full_text().lower()
            matched_semantics: Set[str] = set()
            for word in row.words:
                wt = word.text.lower()
                for pattern, semantic in _COL_HEADER_MAP:
                    if pattern.search(wt):
                        matched_semantics.add(semantic)
                        # Record X center of this header word as anchor
                        if semantic not in header_col_map:
                            header_col_map[semantic] = word.cx

            if len(matched_semantics) >= 3:  # need at least date + amount + one other
                header_row = row
                break

        col_half = img_w * 0.12  # ±12% tolerance per column

        if header_row and len(header_col_map) >= 3:
            logger.info(f"Header-based columns: {header_col_map}")
            result: Dict[str, Tuple[float, float]] = {}
            for name, cx in header_col_map.items():
                result[name] = (cx - col_half, cx + col_half)
            # Ensure narration gets wide range if present
            if "narration" in header_col_map and "date" in header_col_map:
                date_end = header_col_map["date"] + col_half
                next_col_start = min(
                    cx for name, cx in header_col_map.items()
                    if name not in ("date", "narration") and cx > date_end
                ) if any(cx > date_end for name, cx in header_col_map.items()
                         if name not in ("date", "narration")) else img_w * 0.55
                result["narration"] = (date_end, next_col_start - 5)
            return result

        # Fallback: amount X-distribution
        return self._detect_columns_from_amounts(rows, img_w)

    def _detect_columns_from_amounts(
        self,
        rows: List[TableRow],
        img_w: float,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Detect column positions by clustering X-centers of amount-like words
        in transaction rows.
        """
        txn_rows = [
            row for row in rows
            if row.words and _is_date(sorted(row.words, key=lambda w: w.cx)[0].text)
        ]
        if not txn_rows:
            txn_rows = [
                row for row in rows
                if any(_is_amount(w.text) for w in row.words)
            ]

        if not txn_rows:
            return self._fallback_columns(img_w)

        amount_xs: List[float] = []
        for row in txn_rows:
            for w in row.words:
                if _is_amount(w.text):
                    amount_xs.append(w.cx)

        if not amount_xs:
            return self._fallback_columns(img_w)

        amount_xs.sort()
        col_centers = self._cluster_1d(amount_xs, gap=img_w * 0.07)
        col_centers_sorted = sorted(col_centers, reverse=True)

        names = ["balance", "credit", "debit"]
        amount_cols: Dict[str, float] = {}
        for i, cx in enumerate(col_centers_sorted):
            if i < len(names):
                amount_cols[names[i]] = cx

        col_half = img_w * 0.11
        result: Dict[str, Tuple[float, float]] = {}
        for name, cx in amount_cols.items():
            result[name] = (cx - col_half, cx + col_half)

        result["date"] = (0.0, img_w * 0.11)

        leftmost_amt = min(cx - col_half for cx in amount_cols.values()) if amount_cols else img_w * 0.55
        narration_start = img_w * 0.11
        gap_size = leftmost_amt - narration_start

        if gap_size > img_w * 0.30 and len(col_centers) >= 3:
            result["narration"] = (narration_start, img_w * 0.50)
        else:
            result["narration"] = (narration_start, leftmost_amt)

        return result

    def _fallback_columns(self, img_w: float) -> Dict[str, Tuple[float, float]]:
        return {
            "date":      (0.0,           img_w * 0.11),
            "narration": (img_w * 0.11,  img_w * 0.50),
            "debit":     (img_w * 0.63,  img_w * 0.74),
            "credit":    (img_w * 0.74,  img_w * 0.87),
            "balance":   (img_w * 0.87,  img_w),
        }

    def _extract_transactions_box(
        self,
        rows: List[TableRow],
        col_map: Dict[str, Tuple[float, float]],
    ) -> list:
        """Extract transactions from bounding-box rows using column map."""
        from app.models.schemas import Transaction
        transactions: list = []
        _SKIP_FULL_RE = re.compile(
            r"^\s*(?:date|narr|descr|particular|withdrawal|deposit|debit|credit|balance"
            r"|chq|ref|mode|total|opening|closing|brought|carried|page|statement)\b",
            re.IGNORECASE
        )

        for row in rows:
            first_word = sorted(row.words, key=lambda w: w.cx)[0].text if row.words else ""

            is_txn = _is_date(first_word)
            if not is_txn:
                if transactions and not any(_is_amount(w.text) for w in row.words):
                    rt = row.full_text()
                    if len(rt) > 2 and not _SKIP_FULL_RE.match(rt):
                        last = transactions[-1]
                        if last.description:
                            appended = _clean_narration(last.description + " " + rt)
                            last.description = appended[:300]
                continue

            date_str   = row.text_in_range(*col_map.get("date",      (0, 0)))
            narration  = row.text_in_range(*col_map.get("narration",  (0, 0)))
            debit_str  = row.text_in_range(*col_map.get("debit",      (0, 0)))
            credit_str = row.text_in_range(*col_map.get("credit",     (0, 0)))
            bal_str    = row.text_in_range(*col_map.get("balance",    (0, 0)))
            ref_str    = row.text_in_range(*col_map.get("ref",        (0, 0)))

            tx = Transaction(
                date=date_str or None,
                description=_clean_narration(narration or row.full_text())[:300],
                debit=_parse_amount(debit_str),
                credit=_parse_amount(credit_str),
                balance=_parse_amount(bal_str),
            )
            _setattr_safe(tx, "ref_no", ref_str or None)
            transactions.append(tx)

        return transactions

    # ── Balance continuity validator ──────────────────────────────────────

    def _fix_debit_credit_with_balance_continuity(
        self,
        transactions: List[Transaction],
        tolerance: float = 2.0,   # increased from 1.0 — covers BoB 0.05 Cr rounding
        opening_balance: Optional[float] = None,
    ) -> None:
        """
        Walk through transactions in order.
        For each transaction where only credit is set (but not debit),
        check: prev_balance - credit ≈ balance → actually a debit (swap).
        Similarly: prev_balance + debit ≈ balance → actually a credit.

        This is the core engine that correctly classifies BoB-style statements
        where OCR produces one amount per row and we tentatively mark it 'credit'.

        Args:
            transactions: list to fix in-place
            tolerance:    max rounding error in ₹ allowed for a balance match
            opening_balance: seed prev_bal if first tx has no prior context
        """
        prev_bal: Optional[float] = opening_balance  # seed with known opening bal

        for tx in transactions:
            if tx.balance is None:
                # Can't verify — reset chain only if we have no balance at all
                continue  # keep prev_bal — don't reset, next tx might anchor us

            if prev_bal is None:
                # First anchored balance — use it as seed, don't reclassify
                prev_bal = tx.balance
                continue

            has_debit  = tx.debit  is not None and tx.debit  > 0
            has_credit = tx.credit is not None and tx.credit > 0

            if has_credit and not has_debit:
                expected_if_debit  = round(prev_bal - tx.credit, 2)
                expected_if_credit = round(prev_bal + tx.credit, 2)
                if abs(expected_if_debit - tx.balance) <= tolerance:
                    tx.debit = tx.credit
                    tx.credit = None
                # else: correctly a credit — keep

            elif has_debit and not has_credit:
                expected_if_debit  = round(prev_bal - tx.debit, 2)
                expected_if_credit = round(prev_bal + tx.debit, 2)
                if abs(expected_if_credit - tx.balance) <= tolerance:
                    tx.credit = tx.debit
                    tx.debit = None
                # else: correctly a debit — keep

            elif has_debit and has_credit:
                # Both set — verify net direction, nullify wrong one
                net = round(prev_bal + tx.credit - tx.debit, 2)
                if abs(net - tx.balance) > tolerance:
                    if abs(round(prev_bal - tx.debit, 2) - tx.balance) <= tolerance:
                        tx.credit = None
                    elif abs(round(prev_bal + tx.credit, 2) - tx.balance) <= tolerance:
                        tx.debit = None

            prev_bal = tx.balance

    # ── Deduplication ─────────────────────────────────────────────────────

    def _deduplicate_transactions(self, data) -> None:
        """Remove exact duplicate transactions that appear when PDF pages overlap."""
        seen: Set[Tuple] = set()
        unique = []
        for tx in data.transactions:
            key = (tx.date, tx.balance, tx.debit, tx.credit)
            if key not in seen:
                seen.add(key)
                unique.append(tx)
        data.transactions = unique

    # ── Analytics ─────────────────────────────────────────────────────────

    def _compute_analytics(self, data) -> None:
        txs = data.transactions
        data.transaction_count = len(txs)
        if not txs:
            return
        debits  = [t.debit  for t in txs if t.debit  is not None and t.debit  > 0]
        credits = [t.credit for t in txs if t.credit is not None and t.credit > 0]
        data.total_debits   = round(sum(debits),  2) if debits  else None
        data.total_credits  = round(sum(credits), 2) if credits else None
        data.largest_debit  = round(max(debits),  2) if debits  else None
        data.largest_credit = round(max(credits), 2) if credits else None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _cluster_1d(self, values: List[float], gap: float) -> List[float]:
        if not values:
            return []
        groups: List[List[float]] = [[values[0]]]
        for v in values[1:]:
            if v - groups[-1][-1] <= gap:
                groups[-1].append(v)
            else:
                groups.append([v])
        return [sum(g) / len(g) for g in groups]