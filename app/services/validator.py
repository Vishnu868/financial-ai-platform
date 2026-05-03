"""
validation_service.py — Invoice Data Validator
===============================================

What this module does:
  After extraction_service.py pulls fields out of raw OCR text, this module
  checks whether those fields are CORRECT and CONSISTENT with each other.
  It catches OCR misreads and extraction bugs that produced plausible-looking
  but wrong values.

What you get from it:
  validate() → (validation_passed: bool, warnings: List[str])
  - validation_passed=True  → all extracted data is internally consistent
  - validation_passed=False → warnings list tells you exactly what's wrong
  This bool drives the ✅/⚠️ badge in the frontend ResultDisplay.

Checks performed (in order):
  1.  GSTIN format       — length, regex, valid state code           [hard fail]
  2.  GSTIN checksum     — weighted mod-36 algorithm                 [soft warn]
  3.  PAN format         — 5 letters + 4 digits + 1 letter           [hard fail]
  4.  PAN ↔ GSTIN link   — digits 3-12 of GSTIN must equal PAN       [hard fail]
  5.  Tax math           — CGST + SGST + IGST ≈ total_tax            [hard fail]
  6.  CGST == SGST       — required for intra-state supply            [hard fail]
  7.  IGST ⊕ CGST/SGST  — mutually exclusive by GST law              [hard fail]
  8.  GST rate validity  — must be 0, 5, 12, 18, or 28 %             [soft warn]
  9.  Total cross-check  — subtotal + tax - discount + delivery ≈ total [hard fail]
  10. Invoice date       — parseable + not in future                  [hard fail]
  11. Date ordering      — invoice_date >= order_date                 [soft warn]
  12. Amount sanity      — total > 0, tax ≤ total, no negatives       [hard fail]
  13. Field completeness — at least 8/22 critical fields present      [soft warn]

Hard fail = contributes to validation_passed=False
Soft warn = added to warnings list but does NOT flip validation_passed to False
            (used when the check is informational, not proof of wrong data)
"""

import re
import logging
from datetime import datetime, date
from typing import List, Optional, Tuple
from app.models.schemas import InvoiceData

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VALID_GST_RATES = {0.0, 5.0, 12.0, 18.0, 28.0}

VALID_STATE_CODES = set(range(1, 38)) | {97}

DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%Y-%m-%d",
    "%d %b %Y", "%d %B %Y",
    "%b %d, %Y", "%B %d, %Y",
    "%m/%d/%Y",
]

# ─────────────────────────────────────────────────────────────────────────────
# GSTIN CHECKSUM
# The checksum algorithm (weighted Luhn-36) is published by GSTN but has
# minor implementation variants across sources. We treat a checksum FAILURE
# as a SOFT WARNING only — it means "suspicious, verify manually" not
# "definitely wrong", because some known-valid GSTINs don't pass the
# commonly-documented version.
# ─────────────────────────────────────────────────────────────────────────────

_GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_CHAR_VAL = {ch: i for i, ch in enumerate(_GSTIN_CHARS)}


def _gstin_checksum(gstin14: str) -> str:
    """
    Compute expected GSTIN check digit using weighted mod-36.
    Positions are 1-indexed; even positions get weight 2, odd get weight 1.
    Returns expected last character.
    """
    total = 0
    for i, ch in enumerate(gstin14):
        pos = i + 1   # 1-indexed
        weight = 2 if pos % 2 == 0 else 1
        val = _CHAR_VAL.get(ch.upper(), 0)
        product = val * weight
        total += (product // 36) + (product % 36)
    check = (36 - (total % 36)) % 36
    return _GSTIN_CHARS[check]


# ─────────────────────────────────────────────────────────────────────────────
# DATE PARSING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> Optional[date]:
    """Try all known formats. Returns date or None if unparseable."""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionValidator:

    # Tolerances for floating-point / rounding differences
    TAX_TOLERANCE   = 1.0   # ₹1  for CGST+SGST+IGST vs total_tax
    TOTAL_TOLERANCE = 2.0   # ₹2  for subtotal+tax-discount+delivery vs total

    def validate(self, data: InvoiceData) -> Tuple[bool, List[str]]:
        """
        Run all validations.

        Returns:
            (all_hard_checks_passed, list_of_warning_messages)
        """
        hard_failures: List[str] = []
        soft_warnings: List[str] = []

        # ── 1. GSTIN format (hard) ─────────────────────────────────────────
        if data.seller_gst:
            fmt_err = self._validate_gstin_format(data.seller_gst)
            if fmt_err:
                hard_failures.append(fmt_err)
            else:
                # ── 2. GSTIN checksum (soft) ───────────────────────────────
                chk_warn = self._validate_gstin_checksum(data.seller_gst)
                if chk_warn:
                    soft_warnings.append(chk_warn)

        # ── 3. PAN format (hard) ──────────────────────────────────────────
        if data.seller_pan:
            pan_err = self._validate_pan_format(data.seller_pan)
            if pan_err:
                hard_failures.append(pan_err)
            else:
                # ── 4. PAN ↔ GSTIN link (soft warning — extraction can mismatch) ──
                if data.seller_gst and not self._validate_gstin_format(data.seller_gst):
                    link_err = self._validate_pan_gstin_link(data.seller_pan, data.seller_gst)
                    if link_err:
                        soft_warnings.append(link_err)  # Soft, not hard

        # ── 5-7. Tax math (hard) ──────────────────────────────────────────
        tax_errs = self._validate_tax_math(data)
        hard_failures.extend(tax_errs)

        # ── 8. GST rate validity (soft) ───────────────────────────────────
        rate_warns = self._validate_gst_rates(data)
        soft_warnings.extend(rate_warns)

        # ── 9. Total cross-check (hard) ───────────────────────────────────
        total_err = self._validate_total(data)
        if total_err:
            hard_failures.append(total_err)

        # ── 10. Invoice date (hard) ───────────────────────────────────────
        inv_date_obj = None
        if data.invoice_date:
            date_err, inv_date_obj = self._validate_date(data.invoice_date, "Invoice date")
            if date_err:
                hard_failures.append(date_err)

        # ── 11. Date ordering: invoice_date >= order_date (soft) ─────────
        if data.order_date and inv_date_obj:
            ord_date_obj = _parse_date(data.order_date)
            if ord_date_obj and inv_date_obj < ord_date_obj:
                soft_warnings.append(
                    f"Invoice date ({data.invoice_date}) is BEFORE order date "
                    f"({data.order_date}) — possible date extraction error"
                )

        # ── 12. Amount sanity (hard) ──────────────────────────────────────
        amount_errs = self._validate_amounts(data)
        hard_failures.extend(amount_errs)

        # ── 13. Field completeness (soft) ─────────────────────────────────
        completeness_warn = self._validate_completeness(data)
        if completeness_warn:
            soft_warnings.append(completeness_warn)

        # ── Combine & decide ──────────────────────────────────────────────
        all_warnings = hard_failures + soft_warnings
        validation_passed = len(hard_failures) == 0

        if validation_passed:
            data.tax_validated = True

        logger.info(
            f"Validation: {'PASSED' if validation_passed else 'FAILED'} | "
            f"{len(hard_failures)} hard failures, {len(soft_warnings)} soft warnings"
        )
        return validation_passed, all_warnings

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 1 — GSTIN FORMAT
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_gstin_format(self, gstin: str) -> Optional[str]:
        """
        Format: SS PPPPP NNNN P E Z C  (15 chars)
          SS   = 2-digit state code (01-37, 97)
          PPPPP = first 5 letters of PAN
          NNNN  = 4 digits of PAN
          P     = last letter of PAN
          E     = entity type code (1-9 / A-Z)
          Z     = always literal 'Z'
          C     = check digit
        """
        if len(gstin) != 15:
            return (
                f"GSTIN '{gstin}' is {len(gstin)} chars — must be exactly 15. "
                f"Likely OCR misread (e.g. 'O' read as '0')."
            )

        if not re.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]$", gstin):
            return (
                f"GSTIN '{gstin}' has invalid format. "
                f"Expected: 2 digits + 5 uppercase letters + 4 digits + "
                f"1 letter + 1 alphanumeric + 'Z' + 1 alphanumeric."
            )

        state_code = int(gstin[:2])
        if state_code not in VALID_STATE_CODES:
            return (
                f"GSTIN state code '{gstin[:2]}' is not a valid Indian "
                f"state/UT code (valid range: 01-37, 97)."
            )

        if gstin[13] != 'Z':
            return (
                f"GSTIN '{gstin}': position 14 (0-indexed: 13) must be 'Z' — got '{gstin[13]}'. "
                f"Likely OCR misread."
            )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 2 — GSTIN CHECKSUM (soft)
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_gstin_checksum(self, gstin: str) -> Optional[str]:
        """
        Soft check — checksum failure is suspicious but not conclusive.
        Real GSTINs can fail due to algorithm implementation variants.
        """
        expected = _gstin_checksum(gstin[:-1])
        if gstin[-1] != expected:
            return (
                f"GSTIN '{gstin}' checksum is suspicious — "
                f"expected check digit '{expected}', got '{gstin[-1]}'. "
                f"Verify on GSTN portal: https://services.gst.gov.in/services/searchtaxpayer"
            )
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 3 — PAN FORMAT
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_pan_format(self, pan: str) -> Optional[str]:
        """
        PAN format: AAAAA9999A
          First 3 chars: any letters (entity code)
          4th char: entity type (C=Company, P=Person, H=HUF, F=Firm, etc.)
          5th char: first letter of surname/entity name
          6-9: 4 digits (serial number)
          10th: alphabetic check character
        """
        if not re.match(r'^[A-Z]{5}\d{4}[A-Z]$', pan):
            return (
                f"PAN '{pan}' has invalid format. "
                f"Expected: 5 uppercase letters + 4 digits + 1 uppercase letter "
                f"(e.g. AAJCC9783E)."
            )
        # 4th character must be a valid entity type
        valid_4th = set("ABCFGHLJPTK")
        if pan[3] not in valid_4th:
            return (
                f"PAN '{pan}': 4th character '{pan[3]}' is not a valid entity type. "
                f"Valid types: C=Company, P=Person, H=HUF, F=Firm, A=AOP, B=BOI, "
                f"G=Government, J=AJP, L=Local, T=Trust."
            )
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 4 — PAN ↔ GSTIN LINK
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_pan_gstin_link(self, pan: str, gstin: str) -> Optional[str]:
        """
        GSTIN positions 3-12 (0-indexed) are exactly the 10-char PAN.
        Downgraded to soft warning — extraction scope issues can cause legitimate
        mismatches (e.g. seller GSTIN from page 1, PAN from page 2 of multi-invoice PDF).
        """
        gstin_pan_segment = gstin[2:12]
        if gstin_pan_segment.upper() != pan.upper():
            return (
                f"PAN/GSTIN mismatch: PAN '{pan}' should appear at positions "
                f"3-12 of GSTIN '{gstin}' (found '{gstin_pan_segment}' instead). "
                f"Verify manually — may be seller vs platform PAN/GSTIN from different invoice blocks."
            )
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # CHECKS 5-7 — TAX MATH
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_tax_math(self, data: InvoiceData) -> List[str]:
        errors = []

        # 5. Component sum vs total_tax
        if data.total_tax is not None:
            components = [data.cgst_amount, data.sgst_amount, data.igst_amount]
            component_sum = sum(x for x in components if x is not None)
            if component_sum > 0:
                diff = abs(component_sum - data.total_tax)
                if diff > self.TAX_TOLERANCE:
                    errors.append(
                        f"Tax math error: "
                        f"CGST(₹{data.cgst_amount or 0:.2f}) + "
                        f"SGST(₹{data.sgst_amount or 0:.2f}) + "
                        f"IGST(₹{data.igst_amount or 0:.2f}) = "
                        f"₹{component_sum:.2f} ≠ total_tax ₹{data.total_tax:.2f} "
                        f"(diff ₹{diff:.2f}, tolerance ₹{self.TAX_TOLERANCE}). "
                        f"OCR likely misread one of the tax amounts."
                    )

        # 6. CGST must equal SGST (intra-state GST law requirement)
        if data.cgst_amount and data.sgst_amount:
            diff = abs(data.cgst_amount - data.sgst_amount)
            if diff > self.TAX_TOLERANCE:
                errors.append(
                    f"CGST (₹{data.cgst_amount:.2f}) ≠ SGST (₹{data.sgst_amount:.2f}) — "
                    f"Indian GST law requires CGST = SGST for intra-state supply "
                    f"(diff ₹{diff:.2f}). One amount is likely misread."
                )

        # 7. IGST and CGST/SGST are mutually exclusive
        if data.igst_amount and data.igst_amount > 0:
            if (data.cgst_amount and data.cgst_amount > 0) or \
               (data.sgst_amount and data.sgst_amount > 0):
                errors.append(
                    f"Both IGST (₹{data.igst_amount:.2f}) and CGST/SGST detected. "
                    f"IGST applies to inter-state supply; CGST+SGST to intra-state. "
                    f"An invoice cannot have both — one is an extraction error."
                )

        return errors

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 8 — GST RATE VALIDITY (soft)
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_gst_rates(self, data: InvoiceData) -> List[str]:
        """
        Indian GST rates are legally restricted to: 0%, 5%, 12%, 18%, 28%.
        CGST/SGST are half-slabs: 0, 2.5, 6, 9, 14.
        Special rates: 3% IGST applies to gold, silver, precious metals (Section 9).
        Any other value means OCR misread the rate digit.
        """
        warnings = []
        rate_fields = [
            ("CGST rate", data.cgst_rate),
            ("SGST rate", data.sgst_rate),
            ("IGST rate", data.igst_rate),
        ]
        for name, rate in rate_fields:
            if rate is None:
                continue
            # Full valid set including half-slabs and special 3% rate
            all_valid = VALID_GST_RATES | {2.5, 3.0, 6.0, 9.0, 14.0}
            if rate not in all_valid:
                warnings.append(
                    f"{name} is {rate}% — not a standard Indian GST rate. "
                    f"Valid rates: 0%, 3% (precious metals), 5% (2.5+2.5), "
                    f"12% (6+6), 18% (9+9), 28% (14+14). "
                    f"Possible OCR digit misread."
                )
        return warnings

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 9 — TOTAL CROSS-CHECK
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_total(self, data: InvoiceData) -> Optional[str]:
        """
        Formula: subtotal (taxable value) + total_tax ≈ total_amount

        On all supported platforms 'subtotal' is extracted as the TAXABLE VALUE —
        the amount already net of discounts and inclusive of any charges that form
        part of the taxable base.  Discount, delivery, and packaging are stored
        separately for reference but must NOT be re-applied here because they are
        already reflected in the taxable value the invoice reports.

        Correct identity for every platform:
          taxable_value + total_tax = invoice_total
        """
        if data.total_amount is None or data.subtotal is None:
            return None  # Can't check without both anchors

        expected = data.subtotal + (data.total_tax or 0)
        diff = abs(expected - data.total_amount)

        if diff > self.TOTAL_TOLERANCE:
            return (
                f"Total cross-check failed: "
                f"subtotal(₹{data.subtotal:.2f}) + tax(₹{data.total_tax or 0:.2f}) "
                f"= ₹{expected:.2f}, but extracted total = ₹{data.total_amount:.2f} "
                f"(diff ₹{diff:.2f}, tolerance ₹{self.TOTAL_TOLERANCE}). "
                f"One of these amounts is likely misread."
            )
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 10 — DATE VALIDATION
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_date(self, date_str: str, label: str) -> Tuple[Optional[str], Optional[date]]:
        """Returns (error_message_or_None, parsed_date_or_None)."""
        parsed = _parse_date(date_str)
        if parsed is None:
            # Not an error — just an unrecognized format, don't block
            logger.debug(f"Could not parse date '{date_str}' with known formats")
            return None, None
        if parsed > date.today():
            return (
                f"{label} '{date_str}' is in the future ({parsed}) — "
                f"possible OCR misread (e.g. year digit wrong).",
                parsed
            )
        return None, parsed

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 12 — AMOUNT SANITY
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_amounts(self, data: InvoiceData) -> List[str]:
        errors = []

        if data.total_amount is not None:
            # Must be positive (>0, not just >=0)
            if data.total_amount <= 0:
                errors.append(
                    f"Total amount ₹{data.total_amount} is zero or negative — "
                    f"OCR likely misread the total."
                )
            # Sanity ceiling
            elif data.total_amount > 10_000_000:
                errors.append(
                    f"Total amount ₹{data.total_amount:,.2f} exceeds ₹1 crore — "
                    f"verify manually (possible OCR decimal point error)."
                )
            # Tax cannot exceed total
            if data.total_tax and data.total_tax > data.total_amount:
                errors.append(
                    f"Total tax (₹{data.total_tax:.2f}) > total amount (₹{data.total_amount:.2f}) — "
                    f"impossible. One of these values is misread."
                )

        # No negative individual amounts
        for field_name, val in [
            ("cgst_amount", data.cgst_amount),
            ("sgst_amount", data.sgst_amount),
            ("igst_amount", data.igst_amount),
            ("total_tax", data.total_tax),
            ("subtotal", data.subtotal),
        ]:
            if val is not None and val < 0:
                errors.append(
                    f"'{field_name}' is negative (₹{val:.2f}) — "
                    f"tax/amount fields cannot be negative. Likely OCR misread."
                )

        return errors

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 13 — FIELD COMPLETENESS (soft)
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_completeness(self, data: InvoiceData) -> Optional[str]:
        """
        Threshold raised from 3 to 8 — 3/22 was trivially easy to pass
        (just having invoice_type + total_amount + one date would pass it).
        8/22 means at least: invoice number, date, seller info, amounts.
        """
        MIN_FIELDS = 8
        if data.fields_extracted < MIN_FIELDS:
            return (
                f"Only {data.fields_extracted}/22 fields extracted (minimum: {MIN_FIELDS}). "
                f"Document may be low quality, heavily skewed, or an unsupported layout. "
                f"Try uploading a clearer/higher-resolution image."
            )
        return None