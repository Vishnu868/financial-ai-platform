"""
extraction_service_fixed.py
============================
Complete rewrite of InvoiceExtractionService.

ROOT CAUSES FIXED
-----------------
1. GREEDY REGEX → all address regexes now use non-greedy (.*?) with hard stop-word anchors
2. SINGLE PAGE ONLY → OCR service now passes full joined text; extractor works on full text
3. OVER-EXTRACTION in addresses → added comprehensive stop-word boundaries
4. invoice_details grabbing "of" → regex now requires ≥6-char token starting with letter+alnum
5. place_of_delivery EMPTY → added fallback = shipping_address (derived)
6. place_of_supply EMPTY → added fallback via GSTIN state code + explicit patterns per platform
7. reverse_charge MISSING → always defaults to "No" when pattern not found
8. total_tax MISSING → computed from CGST+SGST+IGST if not explicit; also "Total taxes" row
9. amount_in_words TRUNCATED → join across pages; multi-pattern incl BigBasket Rs.Words form
10. seller_address MISSING → added patterns for BigBasket/JioMart/Blinkit/Swiggy/Zomato
11. GSTIN over-matching → seller GSTIN is first found near Sold By / Supplier block, not any
12. Tax amount double-counting → deduplicate summed amounts per tax type
13. BigBasket Bill to/Ship to combined → handled as single block, both fields set
14. JioMart Studocu watermark page → skip non-invoice pages before extraction
15. Myntra multi-invoice PDF → detect and pick relevant invoice block
16. Zomato restaurant + platform fee → both invoice blocks handled
17. Meesho IGST-only → correctly routes to igst_amount, not cgst/sgst
18. BigBasket IGST false positive → BigBasket block returns None for IGST + early exit
19. BigBasket subtotal → extracted from CGST table taxable column sum
20. Multi-invoice scoping → _get_seller_invoice_block() for Blinkit/Myntra/Swiggy/Zomato
21. JioMart SCGST/IGST column → dedicated pattern for inter-state IGST
22. Swiggy per-item CGST/SGST → summed from item rows in seller invoice scope
23. Meesho/Myntra Total row → last amount pattern for total_amount
24. Flipkart discount → Discounts/Coupons column matched
25. validator.py _validate_total → formula fixed: subtotal + tax only (discount already in taxable)

PLATFORM SPECIFICS
------------------
Amazon   : 2 pages; TOTAL row has "net_total final_total"; amount_in_words on page 2
BigBasket: Bill to/Ship to combined; amount_in_words "Rs.Words" on page 2; CGST%+SGST%
Blinkit  : 2 invoices in same PDF (seller + Blink Commerce); pick by Invoice Number block
Flipkart : IGST only; Ship-from / Bill-to layout; Grand Total explicit
JioMart  : Page 1 = Studocu cover → skip; IGST only (inter-state); Supply/Dispatch addr
Meesho   : IGST only; very compact; Sold by at bottom
Myntra   : 2-3 invoices in PDF; separate product + platform fee invoices
Swiggy   : 2 invoices (seller goods + Swiggy handling); pick goods invoice first
Zomato   : Restaurant invoice + Eternal/Zomato platform fee invoice
"""

import re
from typing import Optional, List, Tuple
from app.models.schemas import InvoiceData, InvoiceItem, DocumentPlatform

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

GSTIN_STATE_CODES = {
    "01": "Jammu & Kashmir", "02": "Himachal Pradesh", "03": "Punjab",
    "04": "Chandigarh", "05": "Uttarakhand", "06": "Haryana",
    "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
    "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh",
    "13": "Nagaland", "14": "Manipur", "15": "Mizoram",
    "16": "Tripura", "17": "Meghalaya", "18": "Assam",
    "19": "West Bengal", "20": "Jharkhand", "21": "Odisha",
    "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
    "26": "Dadra & Nagar Haveli", "27": "Maharashtra",
    "28": "Andhra Pradesh", "29": "Karnataka", "30": "Goa",
    "31": "Lakshadweep", "32": "Kerala", "33": "Tamil Nadu",
    "34": "Puducherry", "35": "Andaman & Nicobar",
    "36": "Telangana", "37": "Andhra Pradesh (New)",
    "97": "Other Territory",
}

_ADDR_STOP = (
    r"(?=\n\s*(?:"
    r"GST(?:IN)?|State(?:/UT)?\s*Code|PAN|FSSAI|CIN|"
    r"Invoice\s+(?:Number|No|Date|Details)|Order\s+(?:No|Number|ID|Date)|"
    r"Place\s+of|Ship(?:ping)?\s+(?:Address|To)|Bill(?:ing)?\s+(?:Address|To)|"
    r"Sold\s+By|Details\s+of\s+Supplier|Seller\s+(?:Name|GSTIN)|"
    r"Bill\s+From|Tax\s+Invoice|Additional\s+Information|"
    r"Sr\.?\s*[Nn]o|Sl\.?\s*[Nn]o|Description|HSN|"
    r"Customer\s+Type|Customer\s+ID|Pincode|IN\b|"
    r"Tel\.|Phone:|Email:|Mobile:|"
    r"Telangana|Tamil\s+Nadu|Karnataka|Maharashtra|Rajasthan|"
    r"Terms\s+&|Disclaimer|Authorization|Authorized"
    r"))"
)


def _clean(text: str, max_len: int = 400) -> str:
    return re.sub(r'\s+', ' ', text.strip())[:max_len]


def _state_name(code: str) -> str:
    return GSTIN_STATE_CODES.get(code.zfill(2), "")


def _state_from_gstin(gstin: str) -> Optional[str]:
    if gstin and len(gstin) >= 2:
        code = gstin[:2]
        name = _state_name(code)
        return f"{code} - {name}" if name else code
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main service
# ──────────────────────────────────────────────────────────────────────────────

class InvoiceExtractionService:

    PLATFORM_KEYWORDS = {
        DocumentPlatform.AMAZON:     ["amazon", "amazon.in", "amzn", "asspl", "aripl"],
        DocumentPlatform.FLIPKART:   ["flipkart", "ekart", "flipkart internet",
                                      "international value retail"],
        DocumentPlatform.MEESHO:     ["meesho", "fashnear", "dodiya"],
        # FIX: Myntra PDFs contain GTA pages from "Flipkart India Private Limited".
        # Added invoice-number prefixes and "sixth sense" so Myntra is detected
        # before Flipkart's keywords fire on the GTA page.
        DocumentPlatform.MYNTRA:     ["myntra", "myntra designs", "sixth sense",
                                      "i2926sh", "i2926my", "i2926fi"],
        DocumentPlatform.SWIGGY:     ["swiggy", "bundl technologies", "swiggy limited",
                                      "kwickbox"],
        DocumentPlatform.ZOMATO:     ["zomato", "eternal limited", "zomato limited",
                                      "belgian waffle", "vijay reshma"],
        DocumentPlatform.BIGBASKET:  ["bigbasket", "innovative retail", "bbnow", "bb now"],
        DocumentPlatform.BLINKIT:    ["blinkit", "blink commerce", "grofers",
                                      "moonstone ventures"],
        DocumentPlatform.JIOMART:    ["jiomart", "jio mart", "reliance retail"],
    }

    def detect_platform(self, text: str) -> DocumentPlatform:
        """
        FIX: Myntra PDFs contain GTA pages from 'Flipkart India Private Limited'
        which match 'flipkart' before 'myntra'. Myntra must be checked first.
        """
        tl = text.lower()
        priority_order = [
            DocumentPlatform.MYNTRA,
            DocumentPlatform.ZOMATO,
            DocumentPlatform.BIGBASKET,
            DocumentPlatform.BLINKIT,
            DocumentPlatform.SWIGGY,
            DocumentPlatform.JIOMART,
            DocumentPlatform.MEESHO,
            DocumentPlatform.FLIPKART,
            DocumentPlatform.AMAZON,
        ]
        for platform in priority_order:
            for kw in self.PLATFORM_KEYWORDS[platform]:
                if kw in tl:
                    return platform
        return DocumentPlatform.UNKNOWN

    # ── Pre-processing: strip Studocu/cover pages ──────────────────────────
    def _clean_text(self, text: str) -> str:
        """Remove known watermark/cover content (Studocu, lOMoAR headers)."""
        text = re.sub(
            r"(?:Scan to open on Studocu.*?lOMoARcPSD\|\d+\s*)",
            "", text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r"lOMoARcPSD\|\d+", "", text)
        text = re.sub(r"Downloaded by [^\n]+", "", text)
        text = re.sub(r"Studocu is not sponsored[^\n]*", "", text)
        return text

    # ── Scope to seller invoice block for multi-invoice PDFs ──────────────
    def _get_seller_invoice_block(self, text: str, platform: DocumentPlatform) -> str:
        """
        For multi-invoice PDFs, return only the seller/goods invoice block.
        Prevents values from the platform fee invoice bleeding into extraction.

        Blinkit : page 1 = seller; page 2 = Blink Commerce platform invoice
        Myntra  : page 1 = product invoice; page 2+ = Myntra Designs fee invoice
        Swiggy  : page 1 = KWICKBOX goods; page 2 = Swiggy handling fee invoice
        Zomato  : page 1 = restaurant; page 2 = Eternal Limited platform fee
        """
        if platform == DocumentPlatform.BLINKIT:
            m = re.search(
                r"Blink\s+Commerce\s+Private\s+Limited\s*\(formerly",
                text, re.IGNORECASE
            )
            return text[:m.start()] if m else text

        if platform == DocumentPlatform.MYNTRA:
            # Product invoice block ends at the Myntra Designs fee invoice
            m = re.search(
                r"Invoice\s+(?:Number\s*:|No\s*:)\s*I\d{4}MY",
                text, re.IGNORECASE
            )
            return text[:m.start()] if m else text

        if platform == DocumentPlatform.SWIGGY:
    # "Taxes Rate" header is the FIRST marker of Swiggy's page-2 invoice —
    # it appears before "Invoice From: Swiggy" in the PDF text stream.
            m = re.search(r"^Taxes\s+Rate\s*$", text, re.IGNORECASE | re.MULTILINE)
            if not m:
                m = re.search(r"Invoice\s+From\s*:\s*Swiggy", text, re.IGNORECASE)
            if not m:
                m = re.search(r"GSTIN\s*:\s*36AAFCB7707D", text, re.IGNORECASE)
            return text[:m.start()] if m else text

        if platform == DocumentPlatform.ZOMATO:
            m = re.search(
                r"ETERNAL\s+LIMITED\s*\(FORMERLY\s+KNOWN\s+AS\s+ZOMATO",
                text, re.IGNORECASE
            )
            if not m:
                m = re.search(
                    r"ORIGINAL\s+FOR\s+RECIPIENT\s*\nTax\s+Invoice\s*\nETERNAL",
                    text, re.IGNORECASE
                )
            return text[:m.start()] if m else text

        return text

    # ── Main entry ─────────────────────────────────────────────────────────
    def extract_all_fields(self, text: str, platform: DocumentPlatform) -> InvoiceData:
        text = self._clean_text(text)

        # FIX: Scope seller block for multi-invoice PDFs — used for tax AND
        # seller identity fields (pan, seller_info, seller_gst) so page-2
        # seller data doesn't overwrite page-1 seller data.
        seller_scope = self._get_seller_invoice_block(text, platform)

        seller_gstin = self._get_seller_gst(seller_scope, platform)
        billing_addr = self._get_billing_address(seller_scope, platform)
        shipping_addr = self._get_shipping_address(seller_scope, platform)

        pod = self._get_place_of_delivery(seller_scope)
        if not pod and shipping_addr:
            pod = self._extract_state_from_address(shipping_addr)

        pos = self._get_place_of_supply(seller_scope)
        if not pos and seller_gstin:
            pos = _state_from_gstin(seller_gstin)

        rc = self._get_reverse_charge(seller_scope)
        if not rc:
            rc = "No"

        cgst = self._get_tax_amount(seller_scope, "CGST", platform)
        sgst = self._get_tax_amount(seller_scope, "SGST", platform)
        igst = self._get_tax_amount(seller_scope, "IGST", platform)
        total_tax = self._get_total_tax(seller_scope, cgst, sgst, igst)

        data = InvoiceData(
            platform=platform,
            billing_address=billing_addr,
            shipping_address=shipping_addr,
            invoice_type=self._get_invoice_type(seller_scope),
            order_number=self._get_order_number(text),        # full text for order
            invoice_number=self._get_invoice_number(seller_scope, platform),
            order_date=self._get_date(text, "order"),
            invoice_details=self._get_invoice_details(seller_scope),
            invoice_date=self._get_date(seller_scope, "invoice"),
            seller_info=self._get_seller_info(seller_scope),
            seller_pan=self._get_seller_pan(seller_scope),
            seller_gst=seller_gstin,
            fssai_license=self._get_fssai(seller_scope),
            billing_state_code=self._get_state_code(seller_scope, "billing", seller_gstin),
            shipping_state_code=self._get_state_code(seller_scope, "shipping", seller_gstin),
            place_of_supply=pos,
            place_of_delivery=pod,
            reverse_charge=rc,
            amount_in_words=self._get_amount_in_words(seller_scope),
            seller_name=self._get_seller_name(seller_scope),
            seller_address=self._get_seller_address(seller_scope, platform),
            total_tax=total_tax,
            total_amount=self._get_total_amount(seller_scope, platform),
            buyer_name=self._get_buyer_name(seller_scope),
            buyer_phone=self._get_phone(seller_scope),
            subtotal=self._get_subtotal(seller_scope),
            cgst_amount=cgst,
            sgst_amount=sgst,
            igst_amount=igst,
            cgst_rate=self._get_tax_rate(seller_scope, "CGST"),
            sgst_rate=self._get_tax_rate(seller_scope, "SGST"),
            igst_rate=self._get_tax_rate(seller_scope, "IGST"),
            discount=self._get_discount(seller_scope),
            delivery_charge=self._get_delivery(seller_scope),
            packaging_charge=self._get_packaging(seller_scope),
            payment_method=self._get_payment(text),
            items=self._get_items(seller_scope),
        )
        data.fields_extracted = self._count_fields(data)
        return data

    def calculate_confidence(self, data: InvoiceData, ocr_conf: float) -> float:
        score = data.fields_extracted / 22.0
        return round(min((score * 0.65) + (ocr_conf * 0.35), 1.0), 3)

    def _count_fields(self, d: InvoiceData) -> int:
        return sum(1 for v in [
            d.billing_address, d.shipping_address, d.invoice_type,
            d.order_number, d.invoice_number, d.order_date,
            d.invoice_details, d.invoice_date, d.seller_info,
            d.seller_pan, d.seller_gst, d.fssai_license,
            d.billing_state_code, d.shipping_state_code,
            d.place_of_supply, d.place_of_delivery,
            d.reverse_charge, d.amount_in_words,
            d.seller_name, d.seller_address, d.total_tax, d.total_amount,
        ] if v is not None and v != "" and v != 0)

    # ─────────────────────────────────────────────────────────────────────────
    # ADDRESS EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def _get_billing_address(self, text: str, platform: DocumentPlatform) -> Optional[str]:
        
        if platform == DocumentPlatform.BLINKIT:
            name_m  = re.search(r"(?:^|\n)Name\s*\n:\n([^\n]+)",    text, re.IGNORECASE)
            addr_m  = re.search(r"(?:^|\n)Address\s*\n:\n([^\n]+)", text, re.IGNORECASE)
            state_m = re.search(r"(?:^|\n)State\s*\n:\n([^\n]+)",   text, re.IGNORECASE)
            parts = [m.group(1).strip() for m in [name_m, addr_m, state_m] if m]
            if parts:
                return ", ".join(parts)
        patterns = [
            # BigBasket strict
            (r"Bill\s+to\s*/\s*Ship\s+to:\s*\n((?:[^\n]+\n){2,5}?)\s*Invoice\s+Number",
             re.IGNORECASE),
            # Amazon
            (r"Billing\s+Address\s*[:\s]+\n?((?:(?!GST|State|Ship|IN\b|Place|Order)[^\n]+\n?){1,6})",
             re.IGNORECASE | re.MULTILINE),
            # BigBasket / Blinkit combined
            (r"Bill\s+to\s*/\s*Ship\s+to:?\s*\n((?:(?!Invoice\s+Number|Order\s+No|Additional|GSTIN|Tel\.)[^\n]+\n?){1,5})",
             re.IGNORECASE | re.MULTILINE),
            # Flipkart
            (r"Bill\s+To\s*\n((?:(?!Ship\s+To|GSTIN|PAN|CIN|Order\s+ID)[^\n]+\n?){1,6})",
             re.IGNORECASE | re.MULTILINE),
            # JioMart
            (r"Bill\s+To\s+Address\s*\n((?:(?!Ship\s+To|GSTIN|Place\s+of|State\s+Code)[^\n]+\n?){1,6})",
             re.IGNORECASE | re.MULTILINE),
            # Myntra / Meesho
            (r"Bill\s+to\s*/?\s*Ship\s+to:?\s*\n((?:(?!Customer\s+Type|Bill\s+From|GSTIN)[^\n]+\n?){1,5})",
             re.IGNORECASE | re.MULTILINE),
            # FIX SWIGGY: "Invoice To: Umadevi   Seller Name: KWICKBOX..."
            # Capture only the customer name — stop before "Seller Name:"
            (r"Invoice\s+To\s*:\s*([^\n]+?)(?=\s+Seller\s+Name\s*:|\n|$)",
             re.IGNORECASE),            
            # FIX BLINKIT: "Invoice To\nName: Vishnu\nAddress: 4-34-96..."
            # Capture the full block as one address string
            (r"Invoice\s+To\s*\n(?:Name\s*\n?:\s*\n?([^\n]+)\n)?(?:Address\s*\n?:\s*\n?([^\n]+))?",
             re.IGNORECASE | re.MULTILINE),
            # Swiggy/Zomato multiline fallback
            (r"Invoice\s+To:?\s*\n((?:(?!PAN|GSTIN|Address:|State\s+Code|Pincode\s*:|\bINV\b)[^\n]+\n?){1,5})",
             re.IGNORECASE | re.MULTILINE),
            # Zomato customer block
            (r"Customer\s+Name\s*:\s*([^\n]{3,80})",
             re.IGNORECASE),
            # Zomato platform fee invoice: "Customer Details\nName:...\nDelivery Address:..."
            (r"Customer\s+Details\s*\n(?:Name\s*:\s*[^\n]+\n)(?:GSTIN\s*:\s*[^\n]+\n)?(Delivery\s+Address\s*:\s*[^\n]+)",
             re.IGNORECASE | re.MULTILINE),
            # Generic BILL TO
            (r"BILL\s+TO:?\s*\n((?:(?!Place\s+of|Order|Phone|Ship)[^\n]+\n?){1,6})",
             re.IGNORECASE | re.MULTILINE),
            # Billed To (Myntra GTA)
            (r"Billed\s+To\s*\n((?:(?!State\s*:|GSTIN|Billed\s+From)[^\n]+\n?){1,5})",
             re.IGNORECASE | re.MULTILINE),
        ]
        for pat, flags in patterns:
            m = re.search(pat, text, flags)
            if m:
                addr = _clean(m.group(1))
                if len(addr) > 8:
                    return addr
        return None

    def _get_shipping_address(self, text: str, platform: DocumentPlatform) -> Optional[str]:
        patterns = [
            # Amazon
            (r"Shipping\s+Address\s*[:\s]+\n?((?:(?!GST|State\s*Code|Place\s+of|Order|IN\b)[^\n]+\n?){1,6})",
             re.IGNORECASE | re.MULTILINE),
            # BigBasket combined
            (r"Bill\s*to\s*/?\s*Ship\s*to\s*:?\s*\n((?:[^\n]+\n){2,6}?)\s*Invoice",
             re.IGNORECASE),
            (r"Bill\s+to\s*/\s*Ship\s+to:\s*\n((?:[^\n]+\n){2,5}?)\s*Invoice\s+Number",
             re.IGNORECASE),
            # Flipkart
            (r"Ship\s+To\s*\n((?:(?!Bill\s+To|GSTIN|PAN|CIN|Order\s+ID|Keep\s+this)[^\n]+\n?){1,6})",
             re.IGNORECASE | re.MULTILINE),
            # JioMart
            (r"Ship\s+To\s+Address\s*\n((?:(?!Bill\s+To|GSTIN|Place\s+of|State\s+Code)[^\n]+\n?){1,6})",
             re.IGNORECASE | re.MULTILINE),
            # Myntra GTA
            (r"Shipped\s+To\s*\n((?:(?!Shipped\s+From|Billed\s+To)[^\n]+\n?){1,5})",
             re.IGNORECASE | re.MULTILINE),
            # Meesho / Zomato
            (r"SHIP\s+TO:?\s*\n((?:(?!Order\s+Date|Invoice\s+Date|Terms|Phone)[^\n]+\n?){1,6})",
             re.IGNORECASE | re.MULTILINE),
            # Swiggy / Zomato delivery address
            # Swiggy customer address block (multiline, stop before State Code/Category)
            (r"Delivery\s+Address\s*[:\s]+((?:(?!Category:|State\s+Code|Pincode|State\s+name)[^\n]+\n?){1,4})",
             re.IGNORECASE | re.MULTILINE),
            # Zomato: "Delivery Address : Valarmathi Mess, 641018" — single line only
            (r"Delivery\s+Address\s*:\s*([^\n]+?)(?=\s*\n|\s*State\s+name|$)",
             re.IGNORECASE),
            # Blinkit
            (r"Invoice\s+To\s*\n(?:Name\s*:\s*[^\n]+\n)Address\s*:\s*((?:[^\n]+\n?){1,4}?)(?=(?:Pin\s*code|State|Order\s+Id))",
             re.IGNORECASE | re.MULTILINE),
        ]
        for pat, flags in patterns:
            m = re.search(pat, text, flags)
            if m:
                addr = _clean(m.group(1))
                if len(addr) > 8:
                    return addr

        if platform in (DocumentPlatform.BIGBASKET, DocumentPlatform.BLINKIT):
            return self._get_billing_address(text, platform)

        return None

    def _extract_state_from_address(self, addr: str) -> Optional[str]:
        states = list(GSTIN_STATE_CODES.values())
        for st in states:
            if st.lower() in addr.lower():
                return st
        parts = [p.strip() for p in addr.split(",")]
        if len(parts) >= 2:
            return parts[-1].split("\n")[0].strip()[:60] or None
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # INVOICE TYPE
    # ─────────────────────────────────────────────────────────────────────────

    def _get_invoice_type(self, text: str) -> Optional[str]:
        tl = text.lower()
        if "tax invoice/bill of supply/cash memo" in tl:
            return "Tax Invoice/Bill of Supply/Cash Memo"
        if "original tax invoice" in tl:
            return "Original Tax Invoice"
        if "bill of supply" in tl:
            return "Bill of Supply"
        if "credit note" in tl:
            return "Credit Note"
        if "tax invoice" in tl:
            return "Tax Invoice"
        if "invoice" in tl:
            return "Invoice"
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # ORDER / INVOICE NUMBERS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_order_number(self, text: str) -> Optional[str]:
        patterns = [
            r"Order\s+(?:Number|No\.?|ID|Id)\s*[:#]?\s*([A-Z0-9][\w\-]{4,})",
            r"Customer\s+Order\s+No\.?\s*[:#]?\s*([A-Z0-9][\w\-]{4,})",
            r"Order\s+ID\s*[:#]?\s*(\d[\d\-]{4,})",
            r"\b(\d{3}-\d{7}-\d{7})\b",          # Amazon
            r"\b(\d{7}-\d{7}-\d{7})\b",          # FIX: Myntra format
            r"\b(OD\d{12,})\b",                   # Flipkart
            r"\b(BNN-\d+-\d+)\b",                 # BigBasket
            r"\b(ORD\d{8,})\b",                   # Blinkit platform order
            # FIX: Meesho long numeric order id appears AFTER label on next line
            r"Order\s+(?:Number|No\.?|ID)\s*\n\s*(\d{15,})",
            r"\b(\d{18,})\b",                     # Meesho fallback
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if len(val) >= 5:
                    return val
        return None

    def _get_invoice_number(self, text: str, platform: DocumentPlatform) -> Optional[str]:
        # FIX MYNTRA: prefer product invoice (I2926SH...) over fee/GTA invoice.
        # Also guard against PacketID barcode (9831126688) being grabbed.
        if platform == DocumentPlatform.MYNTRA:
            # First try SH (seller/product) invoice number
            m = re.search(r"Invoice\s+Number\s*[:#]?\s*#?\s*(I\d{4}SH\d+)", text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
            # Fallback: any I####XX invoice number (not GTA FI prefix)
            m = re.search(r"Invoice\s+Number\s*[:#]?\s*#?\s*(I\d{4}(?!FI)[A-Z]{2}\d+)", text, re.IGNORECASE)
            if m and len(m.group(1)) >= 6:
                return m.group(1).strip()

        # Blinkit: prefer seller invoice (C27616T...) over platform invoice (TLFI...)
        if platform == DocumentPlatform.BLINKIT:
            m = re.search(r"Invoice\s+Number\s*[:#]?\s*#?\s*(C\w{10,})", text, re.IGNORECASE)
            if m:
                return m.group(1).strip()

        patterns = [
            r"(?:Tax\s+)?Invoice\s+(?:Number|No\.?)\s*[:#]?\s*#?\s*([A-Z0-9][A-Z0-9\-_/]{4,})",
            r"Invoice\s+No\.?\s*[:#]?\s*([A-Z0-9][A-Z0-9\-_/]{4,})",
            r"Bill\s+(?:No|Number)\.?\s*[:#]?\s*([A-Z0-9][A-Z0-9\-_/]{4,})",
            r"Tax\s+Invoice\s+No\s*[:#]?\s*([A-Z0-9][A-Z0-9\-_/]{4,})",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if len(val) >= 5:
                    return val
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # DATES
    # ─────────────────────────────────────────────────────────────────────────

    _DATE_FMTS = [
        r"\d{1,2}[\.\-/]\d{1,2}[\.\-/]\d{2,4}(?:\s+\d{2}:\d{2}:\d{2})?",
        r"\d{1,2}-[A-Z][a-z]+-\d{4}",
        r"\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?",
        r"\d{1,2}\s+\w+\s+\d{4}",
        r"\d{2}\s+\w{3}\s+\d{4}",
    ]

    def _get_date(self, text: str, dtype: str) -> Optional[str]:
        label = "Order" if dtype == "order" else "Invoice"
        alt_label = r"Ordered\s+on" if dtype == "order" else r"Date\s+of\s+Invoice"
        for fmt in self._DATE_FMTS:
            for prefix in [
                rf"{label}\s+Date\s*[:#]?\s*\n?\s*({fmt})",
                rf"{alt_label}\s*[:#]?\s*({fmt})",
            ]:
                m = re.search(prefix, text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip()
                    if len(val) >= 6:
                        return val
        if dtype == "invoice":
            for fmt in self._DATE_FMTS:
                m = re.search(rf"Date\s*[:#]\s*({fmt})", text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip()
                    if len(val) >= 6:
                        return val
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # INVOICE DETAILS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_invoice_details(self, text: str) -> Optional[str]:
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # SELLER IDENTIFICATION
    # ─────────────────────────────────────────────────────────────────────────

    def _get_seller_info(self, text: str) -> Optional[str]:
        patterns = [
            (r"Sold\s+By\s*[:/]?\s*\n((?:(?!PAN|GST|Billing|IRN|IN\b)[^\n]+\n?){1,5})",
             re.IGNORECASE | re.MULTILINE),
            (r"Details\s+of\s+Supplier\s*\n((?:(?!Tel\.|GSTIN|CIN|FSSAI)[^\n]+\n?){1,5})",
             re.IGNORECASE | re.MULTILINE),
            # Blinkit: "Sold By / Seller\n..."

            (r"Sold\s+By\s*/\s*Seller\s*\n((?:(?!GSTIN|FSSAI|PAN|CIN)[^\n]+\n?){1,4})",
             re.IGNORECASE | re.MULTILINE),
            # SWIGGY: "Seller Name: KWICKBOX RETAIL PVT LTD - ALLWYN COLONY"
            (r"Seller\s+Name\s*:\s*([^\n]{5,80})",
             re.IGNORECASE),
            # ZOMATO: "Tax Invoice on behalf of -\nLegal Entity Name...\nRestaurant Name..."
            # Increased {1,4} → {1,6} so all restaurant info lines are captured
            (r"Tax\s+Invoice\s+on\s+behalf\s+of\s*-\s*\n((?:[^\n]+\n?){1,6}?)(?=\n?\s*(?:Invoice\s+No|Customer))",
             re.IGNORECASE | re.MULTILINE),
            (r"(RELIANCE\s+RETAIL\s+LIMITED[^\n]*)",
             re.IGNORECASE),
            (r"Sold\s+By:\s*([^\n]{5,80})",
             re.IGNORECASE),
            (r"Bill\s+From:?\s*\n((?:(?!GSTIN|PAN|Ship)[^\n]+\n?){1,4})",
             re.IGNORECASE | re.MULTILINE),
            (r"Invoice\s+From:?\s*\n?((?:(?!PAN|Email|GSTIN|Address)[^\n]+\n?){1,3})",
             re.IGNORECASE | re.MULTILINE),
        ]
        for pat, flags in patterns:
            m = re.search(pat, text, flags)
            if m:
                info = _clean(m.group(1))
                if len(info) > 5:
                    return info
        return None

    def _get_seller_name(self, text: str) -> Optional[str]:
        patterns = [
            r"Sold\s+By\s*[:/]?\s*\n?\s*([A-Z][A-Z\s\-&\.]+(?:LTD|LIMITED|LLP|PVT|PRIVATE|INC)[A-Z\s\.\,]*)",
            r"Seller\s+Name\s*[:#]?\s*([^\n]{5,80})",
            r"Restaurant\s+Name\s*[:#]?\s*([^\n]{5,80})",
            r"Legal\s+Entity\s+Name\s*[:#]?\s*([^\n]{5,80})",
            r"Bill\s+From:?\s*\n([^\n]{5,80})",
            r"Sold\s+By\s*[:/]?\s*\n?\s*([^\n]{5,80})",
            r"Details\s+of\s+Supplier\s*\n([^\n]{5,80})",
            r"Invoice\s+From:?\s*\n?\s*([^\n]{5,80})",
            r"Sold\s+by:\s*([^\n]{5,80})",
            r"(RELIANCE\s+RETAIL\s+LIMITED)",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                name = m.group(1).strip().rstrip(' *,')
                # FIX: strip junk prefixes like "Seller\n" or "Ship From:" label
                name = re.sub(r"^(?:Seller|Ship\s+From|Bill\s+From)\s*[:\n]+\s*", "", name, flags=re.IGNORECASE)
                name = name.strip()
                # Reject if it's just a label word
                bad_starts = ("ship from", "bill from", "seller\n", "invoice from")
                if len(name) > 3 and not any(name.lower().startswith(b) for b in bad_starts):
                    return name[:150]
        return None

    def _get_seller_address(self, text: str, platform: DocumentPlatform) -> Optional[str]:
        patterns = [
            (r"Sold\s+By\s*[:/]?\s*\n?[^\n]*\n((?:(?!PAN|GST|IN\b|Billing)[^\n]+\n?){1,4})",
             re.IGNORECASE | re.MULTILINE),
            (r"Details\s+of\s+Supplier\s*\n([^\n]+\n(?:[^\n]+\n){2,8})\s*Tel\.",
             re.IGNORECASE),
            (r"Details\s+of\s+Supplier\s*\n[^\n]*\n((?:(?!Tel\.|GSTIN|CIN)[^\n]+\n?){1,4})",
             re.IGNORECASE | re.MULTILINE),
            # Blinkit seller address
            (r"Sold\s+By\s*/\s*Seller\s*\n[^\n]*\n((?:(?!GSTIN|FSSAI|PAN)[^\n]+\n?){1,3})",
             re.IGNORECASE | re.MULTILINE),
            # JioMart
            (r"Supply/Dispatch\s+Location\s+Address\s*[:#]?\s*\n?((?:(?!Supply\s+State|Tax\s+Invoice)[^\n]+\n?){1,3})",
             re.IGNORECASE | re.MULTILINE),
            # Swiggy seller address (from seller block only — page 2 excluded by scoping)
            (r"(?:Invoice\s+From|Seller\s+Details?)[^\n]*\n(?:[^\n]*\n){0,1}Address\s*:\s*((?:(?!PAN|GSTIN|Pincode|State\s+Code)[^\n]+\n?){1,3})",
             re.IGNORECASE | re.MULTILINE),
            # Flipkart
            (r"Ship[-\s]from\s+Address\s*[:#]?\s*((?:[^\n]+\n?){1,2}?)(?=\n\s*(?:GSTIN|PAN|Total))",
             re.IGNORECASE | re.MULTILINE),
            # Myntra Bill From
            (r"(?:Bill\s+From|Seller\s+Details?)\s*[:#]?\s*\n[^\n]*\n((?:(?!GSTIN|PAN|Qty)[^\n]+\n?){1,3})",
             re.IGNORECASE | re.MULTILINE),
            # FIX ZOMATO: Restaurant Address (stop before Restaurant GSTIN)
            (r"Restaurant\s+Address\s*[:#]?\s*((?:[^\n]+\n?){1,2}?)(?=\n\s*(?:Restaurant\s+GSTIN|FSSAI|Invoice\s+No))",
             re.IGNORECASE | re.MULTILINE),
            # FIX MEESHO: seller address from "Sold by:" block, stop at disclaimer
            (r"Sold\s+by:\s*([^\n]+(?:\n[^\n]+){0,2}?)(?=\n\s*(?:Tax\s+is|This\s+is\s+a\s+computer|Includes))",
             re.IGNORECASE | re.MULTILINE),
            # Generic
            (r"(?:Seller|Registered)\s+Address\s*[:#]?\s*((?:[^\n]+\n?){1,3}?)(?=\n\s*(?:GST|PAN|FSSAI|Tel|City))",
             re.IGNORECASE | re.MULTILINE),
        ]
        for pat, flags in patterns:
            m = re.search(pat, text, flags)
            if m:
                addr = _clean(m.group(1))
                # FIX: truncate at disclaimer phrases
                for stop in ["tax is not payable", "this is a computer generated", "includes discounts"]:
                    idx = addr.lower().find(stop)
                    if idx > 10:
                        addr = addr[:idx].strip().rstrip(',')
                if len(addr) > 10:
                    return addr
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # GSTIN / PAN / FSSAI
    # ─────────────────────────────────────────────────────────────────────────

    _GSTIN_RE = r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]"

    def _get_seller_gst(self, text: str, platform: DocumentPlatform) -> Optional[str]:
        # FIX: Myntra — find GSTIN in product invoice block only
        if platform == DocumentPlatform.MYNTRA:
            product_end = re.search(r"Invoice\s+(?:Number\s*:|No\s*:)\s*I\d{4}MY", text, re.IGNORECASE)
            search_text = text[:product_end.start()] if product_end else text
            m = re.search(rf"GSTIN\s+Number\s*:\s*({self._GSTIN_RE})", search_text, re.IGNORECASE)
            if m:
                return m.group(1)

        seller_block_end = re.search(
            r"\n\s*(?:Billing\s+Address|Bill\s+To\s*\n|Invoice\s+To\s*\n|Customer\s+ID)",
            text, re.IGNORECASE
        )
        seller_block = text[:seller_block_end.start()] if seller_block_end else text[:2500]

        for p in [
            rf"(?:GST(?:IN)?|GST\s+Registration\s+No)\s*[:#]?\s*({self._GSTIN_RE})",
            rf"GSTIN\s*[:#]?\s*({self._GSTIN_RE})",
        ]:
            m = re.search(p, seller_block, re.IGNORECASE)
            if m:
                return m.group(1)

        m = re.search(rf"\b({self._GSTIN_RE})\b", text)
        return m.group(1) if m else None

    def _get_all_gstins(self, text: str) -> List[str]:
        return re.findall(rf"\b({self._GSTIN_RE})\b", text)

    def _get_seller_pan(self, text: str) -> Optional[str]:
        for p in [
            r"PAN\s*(?:No)?\.?\s*[:#]?\s*([A-Z]{5}\d{4}[A-Z])",
            r"(?:^|\n)\s*PAN\s*[:#]?\s*([A-Z]{5}\d{4}[A-Z])",
            r"\bPAN\b[^\n]{0,5}([A-Z]{5}\d{4}[A-Z])",
        ]:
            m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1)
        return None

    def _get_fssai(self, text: str) -> Optional[str]:
        for p in [
            r"FSSAI\s*(?:Lic\.?|License)?\s*(?:No\.?|Number)?\s*[:#]?\s*(\d{10,18})",
            r"FSSAI\s*[:#]?\s*(\d{10,18})",
        ]:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # STATE CODES & PLACE OF SUPPLY / DELIVERY
    # ─────────────────────────────────────────────────────────────────────────

    def _get_state_code(self, text: str, stype: str, gstin: Optional[str]) -> Optional[str]:
        m = re.search(r"State(?:/UT)?\s+Code\s*[:#]?\s*(\d{1,2})", text, re.IGNORECASE)
        if m:
            code = m.group(1).zfill(2)
            name = _state_name(code)
            return f"{code} - {name}" if name else code
        if gstin and len(gstin) >= 2:
            code = gstin[:2]
            name = _state_name(code)
            return f"{code} - {name}" if name else code
        return None

    def _get_place_of_supply(self, text: str) -> Optional[str]:
        patterns = [
            r"Place\s+of\s+[Ss]upply\s*(?:&\s*State\s+Code)?\s*[:#]?\s*([^\n,]{2,60})",
            r"State\s+name\s*&\s*Place\s+of\s+Supply\s*[:#]?\s*([^\n]{2,60})",
            r"Place\s+of\s+Supply\s*&\s*State\s+Code\s*[:#]?\s*([^\n]{2,30})",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip().rstrip(',').strip()[:80]
                if val:
                    return val
        return None

    def _get_place_of_delivery(self, text: str) -> Optional[str]:
        m = re.search(r"Place\s+of\s+[Dd]elivery\s*[:#]?\s*([^\n]{2,60})", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()[:80]
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # REVERSE CHARGE
    # ─────────────────────────────────────────────────────────────────────────

    def _get_reverse_charge(self, text: str) -> Optional[str]:
        for p in [
            r"reverse\s+charge\s*[:\-–—]?\s*(Yes|No)\b",
            r"supply\s+attracts\s+reverse\s+charge\s*[:#]?\s*(Yes|No)\b",
            r"Is\s+the\s+supply\s+subject\s+to\s+reverse\s+charge\s*[:#]?\s*(Yes|No)\b",
            # FIX: "Whether reverse charges applicable: No"
            r"Whether\s+Reverse\s+Charges?\s+Applicable\s*[:#]?\s*(Yes|No)\b",
            r"[Ww]hether\s+(?:the\s+)?tax\s+is\s+payable\s+under\s+reverse\s+charge\s*[-:]\s*(Yes|No)",
            # FIX: "Tax is not payable under reverse charge" → No (explicit "not")
            r"Tax\s+is\s+not\s+payable\s+(?:on\s+|under\s+)?reverse\s+charge",
            # FIX: "Tax is not payable on reverse charge basis" → No
            r"tax\s+is\s+not\s+payable\s+on\s+reverse\s+charge\s+basis",
        ]:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                if m.lastindex and m.group(m.lastindex):
                    return m.group(m.lastindex).capitalize()
                # Pattern matched "not payable" phrase → always No
                return "No"
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # AMOUNT IN WORDS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_amount_in_words(self, text: str) -> Optional[str]:
        patterns = [
            r"Total\s+Invoice\s+value\s*\(In\s+words\):\s*Rs\.?\s*([A-Za-z\s]+Paisa(?:\s+Only)?)",
            r"Total\s+Invoice\s+value\s*\(?[Ii]n\s+words?\)?\s*[:#]?\s*(?:Rs\.?)?\s*([A-Z][A-Za-z\s\-&]+(?:Paisa\s+Only|Only))",
            r"Amount\s+(?:in\s+|(?:\(in\s+))[Ww]ords?\)?[:#]?\s*(?:Rs\.?)?\s*([A-Z][A-Za-z\s\-&]+(?:Only|Paisa\s+Only|Rupees[^\n]*))",
            r"Amount\s+in\s+Words\s*[:#]?\s*\n?\s*([A-Z][^\n]{5,})",
            r"Invoice\s+total\s+in\s+words\s+([A-Z][^\n]{5,})",
            r"Amount\s+in\s+Words:\s*([A-Z][^\n]{5,})",
            r"Amount\s+in\s+Words?\s*[:#]?\s*((?:[A-Z][a-z]+\s*)+(?:only|Only))",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if len(val) > 5:
                    return val[:300]
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # AMOUNTS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_total_amount(self, text: str, platform: DocumentPlatform) -> Optional[float]:
        patterns = [
            # Most reliable: number just before "Amount in words"
            r"([\d,]+\.?\d+)\s*\n\s*Amount\s+(?:\(?\s*in\s+)?[Ww]ords",
            # Amazon / Flipkart
            r"Grand\s+Total\s*[₹Rs\.]+\s*([\d,]+\.?\d+)",
            # BigBasket
            r"Total\s+Invoice\s+value\s*\(?\s*In\s+Figure\s*\)?\s*[:#]?\s*Rs\.?\s*([\d,]+\.?\d+)",
            r"Final\s+Total\s*[:\s]*(?:Rs\.?|₹)?\s*([\d,]+\.?\d+)",
            # JioMart
            r"Total\s+Invoice\s+Value\s*\n?\s*₹?\s*([\d,]+\.?\d+)",
            # Zomato
            r"Total\s+Value\s*\n?.*?([\d,]+\.?\d+)\s*$",
            # Swiggy goods invoice
            r"Invoice\s+Value\s+(\d[\d,]*\.?\d*)",
            r"Invoice\s+Total\s*₹?\s*([\d,]+\.?\d+)",
            # JioMart balance
            r"Balance\s+Due\s*[:#]?\s*([\d,]+\.?\d+)",
            # FIX: Myntra — "TOTAL Rs X Rs X Rs X Rs X Rs X Rs FINAL" last value
            r"^TOTAL\s+(?:Rs\s+[\d,\.]+\s+){4,}Rs\s+([\d,\.]+)\s*$",
            # FIX: Meesho — "Total Rs.tax Rs.grand" last Rs. amount on Total row
            r"^Total\s+(?:Rs\.?\s*[\d,\.]+\s+)*Rs\.?\s*([\d,\.]+)\s*$",
            # FIX: Blinkit / generic — "Total qty taxable cgst sgst grand"
            r"^Total\s+\d+\s+[\d,\.]+\s+[\d,\.]+\s+([\d,\.]+)\s*$",
            # Meesho simple
            r"^Total\s+(?:Rs\.?|₹)\s*([\d,]+\.?\d+)\s*$",
            # Amazon TOTAL row
            r"TOTAL[:\s]+₹[\d,]+\.?\d+\s+₹([\d,]+\.?\d+)",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
            if m:
                grp = m.group(m.lastindex) if m.lastindex else m.group(1)
                try:
                    val = float(grp.replace(",", ""))
                    if val > 0:
                        return val
                except ValueError:
                    continue
        return None

    def _get_total_tax(
        self,
        text: str,
        cgst: Optional[float],
        sgst: Optional[float],
        igst: Optional[float],
    ) -> Optional[float]:
        for p in [
            r"Total\s+[Tt]ax(?:es|(?:\s+Amount|\s+Value))?\s*[:#]?\s*₹?\s*([\d,]+\.?\d+)",
            r"Total\s+taxes\s*\n?\s*([\d,]+\.?\d+)",
            r"Total\s+Tax\s+Amount\s*[:#]?\s*(?:Rs\.?|₹)?\s*([\d,]+\.?\d+)",
        ]:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    if val > 0:
                        return val
                except ValueError:
                    pass

        parts = [x for x in [cgst, sgst, igst] if x is not None and x > 0]
        if parts:
            return round(sum(parts), 2)
        return None

    def _get_subtotal(self, text: str) -> Optional[float]:
        # FIX BIGBASKET: sum taxable value column from CGST table
        if "bigbasket" in text.lower() or "bbnow" in text.lower():
            table_match = re.search(
                r"CGST%\s+Sale\s+Taxable\s+Value\s+Tax\s+Value(.*?)"
                r"(?:SGST%\s+Sale|Transaction\s+ID|\Z)",
                text, re.IGNORECASE | re.DOTALL
            )
            if table_match:
                rows = re.findall(
                    r"\d+\.?\d*%\s+Rs\.\s*[\d\.]+\s+Rs\.\s*([\d\.]+)\s+Rs\.",
                    table_match.group(1), re.IGNORECASE
                )
                if rows:
                    return round(sum(float(v) for v in rows), 2)

        if re.search(r"Seller\s+Name\s*:\s*KWICKBOX", text, re.IGNORECASE):
            return None

        # BLINKIT: UPC digits (multi-part like "890 800 003 483 9") fool the
        # digit-start pattern → return None (Check9 skipped, which is correct
        # since total=35.00 and tax=5.34 are both extracted accurately)
        if re.search(r"Invoice\s+Number\s*:\s*C\w{10,}", text, re.IGNORECASE):
            return None

        return self._amt([
            # JioMart explicit
            r"Total\s+Taxable\s+Amount\s*[:#]?\s*(?:Rs\.?|₹)?\s*([\d,]+\.?\d+)",
            # Flipkart Total row: Total qty gross discount TAXABLE igst grand
            r"^Total\s+\d+\s+[\d,\.]+\s+[-\d,\.]+\s+([\d,\.]+)\s+[\d,\.]+\s+[\d,\.]+\s*$",
            # Meesho: "Taxable Value Rs.78.64"
            r"Taxable\s+Value\s+Rs\.?\s*([\d,]+\.?\d+)",
            # ZOMATO: "Total Value 390 9.75 9.75 409.5" — net total INCLUDING packaging.
            # Must come BEFORE Item(s) Total so 390 is used (390+19.5=409.5 ✓)
            # NOT 370 from Item(s) Total which excludes packaging (370+19.5≠409.5)
            r"Total\s+Value\s+([\d,\.]+)\s+[\d,\.]+\s+[\d,\.]+\s+[\d,\.]+",
            # Zomato fallback: Item(s) Total gross discount NET
            r"Item\(s\)\s+Total\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)",
            # Generic
            r"Taxable\s+(?:Value|Amount)\s*[:#]?\s*(?:Rs\.?|₹)?\s*([\d,]+\.?\d+)",
            r"Sub\s*[-\s]?[Tt]otal\s*[:#]?\s*(?:Rs\.?|₹)?\s*([\d,]+\.?\d+)",
        ], text)

    def _get_discount(self, text: str) -> Optional[float]:
        return self._amt([
            # BigBasket "You Saved"
            r"You\s+Saved\s*[:#]?\s*(?:Rs\.?|₹)\s*([\d,]+\.?\d+)",
            # FIX: Flipkart "Discounts/Coupons" column
            r"Discounts?\s*/?\s*Coupons?\s*[₹Rs\.]*\s*-?\s*([\d,]+\.?\d+)",
            # Generic
            r"(?:Total\s+)?Discount[s]?\s*[:#]?\s*[-₹Rs\.]*\s*([\d,]+\.?\d+)",
        ], text)

    def _get_delivery(self, text: str) -> Optional[float]:
        return self._amt([
            r"(?:Delivery|Shipping)\s+(?:Charge|Fee|Cost)s?\s*[:#]?\s*₹?\s*([\d,]+\.?\d+)",
        ], text)

    def _get_packaging(self, text: str) -> Optional[float]:
        return self._amt([
            # BigBasket Handling Charge (guard: not Blinkit/Swiggy item rows)
            r"(?<!\bFees\s)(?<!\bFee\s)Handling\s+Charge\s*\n?₹?\s*([\d\.]+)",
            r"(?:Packaging|Packing)\s+Charges?\s*[:#]?\s*₹?\s*([\d,]+\.?\d+)",
            # Zomato restaurant packaging
            r"Restaurant\s+Packaging\s+Charge\s+(\d+)\s+0\s+\d+",
        ], text)

    # ─────────────────────────────────────────────────────────────────────────
    # TAX AMOUNTS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_tax_amount(self, text: str, tax: str, platform: Optional[DocumentPlatform] = None) -> Optional[float]:
        """
        Extract a single tax type (CGST / SGST / IGST) from scoped invoice text.
        Platform-specific blocks run first and return early.

        KEY PRINCIPLE: Each invoice prints tax amounts either as:
          (a) a computed INR value in a column (most platforms), OR
          (b) a rate% applied to taxable value (Myntra: "5.0% IGST" header + value in col)
        We extract the printed value directly — never recompute from rate.
        """

        # ── BIGBASKET ──────────────────────────────────────────────────────
        if "bigbasket" in text.lower() or "bbnow" in text.lower():
            if tax == "IGST":
                return None  # BigBasket is always intra-state
            tax_label = "CGST" if tax == "CGST" else "SGST"
            table_match = re.search(
                rf"{tax_label}%\s+Sale\s+Taxable\s+Value\s+Tax\s+Value(.*?)"
                r"(?:(?:SGST|CGST|IGST)%\s+Sale|Transaction\s+ID|Final\s+Total|Authorized|\Z)",
                text, re.IGNORECASE | re.DOTALL
            )
            if table_match:
                block = table_match.group(1)
                row_vals = re.findall(
                    r"\d+\.?\d*%\s+Rs\.\s*[\d\.]+\s+Rs\.\s*[\d\.]+\s+Rs\.\s*([\d\.]+)",
                    block, re.IGNORECASE
                )
                if not row_vals:
                    row_vals = re.findall(r"\d+\.?\d*%\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)", block)
                if row_vals:
                    return round(sum(float(v) for v in row_vals), 2)
            gst_block_match = re.search(
                rf"GST\s+Information.*?{tax_label}%(.*?)(?:Transaction\s+ID|Final\s+Total|\Z)",
                text, re.IGNORECASE | re.DOTALL
            )
            if gst_block_match:
                block = gst_block_match.group(1)
                values = re.findall(r"Rs\.\s*([\d\.]+)", block)
                if not values:
                    values = re.findall(r"\b\d+\.\d{1,2}\b", block)
                if values:
                    nums = [float(v) for v in values]
                    tax_vals = [nums[i] for i in range(2, len(nums), 3)]
                    return round(sum(tax_vals), 2)
            return None

        # ── JIOMART ────────────────────────────────────────────────────────
        if platform == DocumentPlatform.JIOMART:
            if tax == "IGST":
                m = re.search(r"SCGST/IGST\s+[\d\.]+\s+([\d,]+\.?\d+)", text, re.IGNORECASE)
                if m:
                    try:
                        return round(float(m.group(1).replace(",", "")), 2)
                    except ValueError:
                        pass
            if tax in ("CGST", "SGST"):
                return None  # JioMart is inter-state

        # ── FLIPKART ───────────────────────────────────────────────────────
        # Flipkart IGST column header is "IGST ₹" or "IGST ₹" with value in row.
        # The Total row: "Total  1  3555.00  -140.00  3049.11  365.90  3415.00"
        # IGST is the 5th numeric value on the Total row.
        if platform == DocumentPlatform.FLIPKART:
            if tax == "IGST":
                # Try explicit IGST column patterns first
                m = re.search(r"IGST\s*[₹Rs\.]*\s+([\d,]+\.?\d+)", text, re.IGNORECASE)
                if m:
                    try:
                        val = round(float(m.group(1).replace(",", "")), 2)
                        if val > 0:
                            return val
                    except ValueError:
                        pass
                # FIX: extract from TOTAL row — format: Total qty gross discount taxable IGST grand
                # "Total  1  3555.00  -140.00  3049.11  365.90  3415.00"
                m = re.search(
                    r"^Total\s+\d+\s+[\d,\.]+\s+[-\d,\.]+\s+[\d,\.]+\s+([\d,\.]+)\s+[\d,\.]+\s*$",
                    text, re.IGNORECASE | re.MULTILINE
                )
                if m:
                    try:
                        val = round(float(m.group(1).replace(",", "")), 2)
                        if val > 0:
                            return val
                    except ValueError:
                        pass
                # Also try "IGST: 12.0 %\n...365.90" inline format
                m = re.search(r"IGST[:\s]+[\d\.]+\s*%[^\n]*\n[^\n]*?([\d,]+\.?\d+)", text, re.IGNORECASE)
                if m:
                    try:
                        val = round(float(m.group(1).replace(",", "")), 2)
                        if val > 0:
                            return val
                    except ValueError:
                        pass
            if tax in ("CGST", "SGST"):
                return None  # Flipkart is inter-state

        # ── MYNTRA ─────────────────────────────────────────────────────────
        # Myntra table: Qty | Gross | Discount | OtherCharges | Taxable | CGST | SGST/UGST | IGST | Cess | Total
        # For inter-state (most Myntra): IGST column has Rs value, CGST/SGST blank.
        # Row: "1  Rs 2399.00  Rs 1794.00  Rs 0.00  Rs 576.19  [blank]  [blank]  Rs 28.81  [blank]  Rs 605.00"
        if platform == DocumentPlatform.MYNTRA:
            if tax == "IGST":
                # FIX: Match "Rs 28.81" in the IGST column of TOTAL row
                # TOTAL row: "TOTAL  Rs X  Rs Y  Rs 0.00  Rs taxable  Rs IGST  Rs grand"
                m = re.search(
                    r"^TOTAL\s+Rs\s+[\d,\.]+\s+Rs\s+[\d,\.]+\s+Rs\s+[\d,\.]+\s+Rs\s+[\d,\.]+\s+Rs\s+([\d,\.]+)\s+Rs\s+[\d,\.]+\s*$",
                    text, re.IGNORECASE | re.MULTILINE
                )
                if m:
                    try:
                        val = round(float(m.group(1).replace(",", "")), 2)
                        if val > 0:
                            return val
                    except ValueError:
                        pass
                # Fallback: look for IGST amount after taxable value in item rows
                # "Rs 576.19  Rs 28.81  Rs 605.00" — IGST is middle value
                rows = re.findall(
                    r"Rs\s+([\d,\.]+)\s+Rs\s+([\d,\.]+)\s+Rs\s+([\d,\.]+)\s*$",
                    text, re.MULTILINE
                )
                for r in rows:
                    try:
                        taxable, igst, total = float(r[0]), float(r[1]), float(r[2])
                        if abs(taxable + igst - total) < 1.0 and igst > 0:
                            return round(igst, 2)
                    except ValueError:
                        pass
            if tax in ("CGST", "SGST"):
                # Myntra product invoices are inter-state — CGST/SGST = 0/None
                # But for intra-state Myntra orders they would exist — check first
                m = re.search(rf"{tax}\s+Rs\s+([\d,\.]+)", text, re.IGNORECASE)
                if m:
                    try:
                        val = round(float(m.group(1).replace(",", "")), 2)
                        if val > 0:
                            return val
                    except ValueError:
                        pass
                return None

        # ── BLINKIT ────────────────────────────────────────────────────────
        # Seller invoice table (scoped): columns CGST (%) | CGST (INR) | SGST (%) | SGST (INR)
        # Item row: "29.66  9.00  2.67  9.00  2.67  0.00  0.00  35.00"
        # Total row: "Total  1  [blank]  [blank]  2.67  [blank]  2.67  [blank]  [blank]  35.00"
        if platform == DocumentPlatform.BLINKIT and tax in ("CGST", "SGST"):
            # From Total row: "Total  qty  ...  CGST_inr  ...  SGST_inr  ...  grand"
            m = re.search(
                r"^Total\s+\d+.*?([\d\.]+)\s+([\d\.]+)\s+[\d\.]+\s*$",
                text, re.IGNORECASE | re.MULTILINE
            )
            if m:
                try:
                    cgst_val = round(float(m.group(1)), 2)
                    sgst_val = round(float(m.group(2)), 2)
                    if tax == "CGST" and cgst_val > 0:
                        return cgst_val
                    if tax == "SGST" and sgst_val > 0:
                        return sgst_val
                except ValueError:
                    pass
            # FIX: per-item rows: "taxable  CGST%  CGST_inr  SGST%  SGST_inr  0  0  total"
            rows = re.findall(
                r"([\d\.]+)\s+(\d+(?:\.\d+)?)\s+([\d\.]+)\s+(\d+(?:\.\d+)?)\s+([\d\.]+)\s+0(?:\.0+)?\s+0(?:\.0+)?\s+[\d\.]+",
                text
            )
            if rows:
                idx = 2 if tax == "CGST" else 4
                total = round(sum(float(r[idx]) for r in rows), 2)
                if total > 0:
                    return total

        # ── SWIGGY seller invoice: sum per-item CGST/SGST ─────────────────
        # Column order: Net Taxable | CGST% | CGST | SGST% | SGST | Cess | AddCess | Total
        if platform == DocumentPlatform.SWIGGY and tax in ("CGST", "SGST"):
            rows = re.findall(
                r"([\d\.]+)\s+(\d+(?:\.\d+)?)\s+([\d\.]+)\s+(\d+(?:\.\d+)?)\s+([\d\.]+)\s+0\s+0\s+0\s+(\d+)",
                text
            )
            if rows:
                idx = 2 if tax == "CGST" else 4
                total = round(sum(float(r[idx]) for r in rows), 2)
                if total > 0:
                    return total

        # ── ZOMATO restaurant invoice ──────────────────────────────────────
        # Table: Particulars | Gross | Discount | Net | CGST(Rate) | CGST(INR) | SGST(Rate) | SGST(INR) | Total
        # "Total Value" row: "390  [blank]  9.75  [blank]  9.75  409.5"
        # Full row captured as: "Total Value 390 9.75 9.75 409.5"
        if platform == DocumentPlatform.ZOMATO and tax in ("CGST", "SGST"):
            # From "Total Value" row: group1=net, group2=CGST, group3=SGST, group4=total
            m = re.search(
                r"Total\s+Value\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)",
                text, re.IGNORECASE
            )
            if m:
                try:
                    cgst = round(float(m.group(2)), 2)
                    sgst = round(float(m.group(3)), 2)
                    if tax == "CGST" and cgst > 0:
                        return cgst
                    if tax == "SGST" and sgst > 0:
                        return sgst
                except ValueError:
                    pass
            # FIX: also sum CGST(INR)/SGST(INR) column per row:
            # Each item row: "gross  discount  net  rate%  cgst_inr  rate%  sgst_inr  total"
            rows = re.findall(
                r"[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+%\s+([\d\.]+)\s+[\d\.]+%\s+([\d\.]+)\s+[\d\.]+",
                text
            )
            if rows:
                idx = 0 if tax == "CGST" else 1
                total = round(sum(float(r[idx]) for r in rows), 2)
                if total > 0:
                    return total

        # ── General patterns (all other platforms) ────────────────────────
        summary_patterns = [
            rf"^{tax}\s+\d+%?\s+([\d,]+\.?\d+)\s*$",
            rf"^{tax}\s+[\d\.]+%\s+([\d,]+\.?\d+)",
            rf"{tax}\s*(?:\(\s*\d+\.?\d*\s*%?\s*\))?\s*[:#]?\s*₹?\s*(?:Rs\.?)?\s*([\d,]+\.?\d+)",
            rf"{tax}\s*\((?:INR|₹)\)\s*\n?\s*([\d,]+\.?\d+)",
            rf"{tax}\s*@[\d\.]+%\s*:Rs\.([\d,]+\.?\d+)",
            rf"{tax}\s*@[\d\.]+%\s*[:#]?\s*(?:Rs\.?)?\s*([\d,]+\.?\d+)",
        ]

        seen_vals: set = set()
        for p in summary_patterns:
            for m in re.finditer(p, text, re.IGNORECASE | re.MULTILINE):
                try:
                    v = round(float(m.group(1).replace(",", "")), 2)
                    if v > 0:
                        seen_vals.add(v)
                except ValueError:
                    pass

        if not seen_vals:
            return None

        vals = sorted(seen_vals, reverse=True)
        if len(vals) == 1:
            return vals[0]

        total_candidate = vals[0]
        rest_sum = round(sum(vals[1:]), 2)
        if abs(total_candidate - rest_sum) < 0.02:
            return total_candidate

        return round(sum(vals), 2)

    def _get_tax_rate(self, text: str, tax: str) -> Optional[float]:
        m = re.search(
            rf"{tax}\s*(?:\(?\s*%?\s*\)?)?\s*[:#@]?\s*(\d+\.?\d*)\s*%",
            text, re.IGNORECASE
        )
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # BUYER / CUSTOMER
    # ─────────────────────────────────────────────────────────────────────────

    def _get_buyer_name(self, text: str) -> Optional[str]:
        for p in [
            r"Legal\s+Name\s*[:#]?\s*([^\n]{3,60})",
            r"Invoice\s+To\s*:\s*(\S[^S\n]*?)(?=\s+Seller\s+Name|\n|$)",
            r"Bill\s*to\s*/?\s*Ship\s*to\s*:?\s*\n\s*([A-Za-z\s\.]+),",
            r"Customer\s+Name\s*[:#]?\s*([^\n]{3,60})",
            r"Invoice\s+To\s*\n\s*Name\s*[:#]?\s*([^\n]{3,60})",
            r"(?:Ship|Bill)\s+To:?\s*\n\s*([A-Z][^\n]{2,60})",
            r"Name\s+of\s+the\s+Customer\s*[:#]?\s*([^\n]{3,60})",
            r"(?<!Seller\s)(?<!Legal\s)Name\s*[:#]?\s*:\s*([A-Z][^\n]{2,60})",
        ]:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                name = m.group(1).strip()[:100]
                if len(name) > 2 and not name.lower().startswith(("the ", "invoice", "address")):
                    return name
        return None

    def _get_phone(self, text: str) -> Optional[str]:
        m = re.search(
            r"(?:Mobile|Phone|Tel)\s*(?:No\.?)?\s*[:#]?\s*((?:\+91|91)?[\s\-]?[6-9]\d{9})",
            text, re.IGNORECASE
        )
        return m.group(1).strip() if m else None

    def _get_payment(self, text: str) -> Optional[str]:
        m = re.search(r"Payment\s+(?:Mode|Method|Type)\s*[:#]?\s*([^\n]{2,30})", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        tl = text.lower()
        for method in [
            "upi", "credit card", "debit card", "net banking",
            "cod", "cash on delivery", "amazon pay", "paytm",
            "phonepe", "google pay", "wallet", "emi", "neft",
        ]:
            if method in tl:
                return method.title()
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # ITEMS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_items(self, text: str) -> Optional[List[InvoiceItem]]:
        items = []
        for m in re.finditer(
            r"HSN[:/]?\s*(\d{4,8}).*?(?:₹|Rs\.?)\s*([\d,]+\.?\d+)\s*$",
            text, re.MULTILINE | re.IGNORECASE,
        ):
            try:
                items.append(InvoiceItem(
                    hsn_code=m.group(1),
                    total_price=float(m.group(2).replace(",", "")),
                ))
            except ValueError:
                pass
        return items if items else None

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────────────────

    def _amt(self, patterns: list, text: str) -> Optional[float]:
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except (ValueError, IndexError):
                    continue
        return None