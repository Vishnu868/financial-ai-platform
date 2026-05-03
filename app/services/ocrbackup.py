"""
OCR Service — RapidOCR + EasyOCR confidence-based fallback.

Install: pip install rapidocr-onnxruntime

PDF  → PyMuPDF direct text (98%, instant)
Image → RapidOCR pass 1 (original) → pass 2 (enhanced) → EasyOCR fallback
"""

import cv2
import numpy as np
import time
import re
import logging
from typing import Tuple
from app.services.preprocess import file_to_images, preprocess_image
from app.config import settings

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self):
        logger.info("Initializing OCR engines...")
        self._rapid = None
        self._easy = None
        self._init_rapid()
        self._init_easy()
        logger.info("OCR engines ready.")

    def _init_rapid(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._rapid = RapidOCR()
            logger.info("RapidOCR loaded successfully")
        except ImportError:
            logger.warning("RapidOCR not installed. Run: pip install rapidocr-onnxruntime")
        except Exception as e:
            logger.warning(f"RapidOCR init failed: {e}")

    def _init_easy(self):
        try:
            import easyocr
            self._easy = easyocr.Reader(["en"], gpu=settings.ocr_use_gpu)
            logger.info("EasyOCR loaded successfully")
        except ImportError:
            logger.warning("EasyOCR not installed")
        except Exception as e:
            logger.warning(f"EasyOCR init failed: {e}")

    def extract_from_file(
        self, file_bytes: bytes, content_type: str
    ) -> Tuple[str, float, dict]:
        start = time.time()

        # Strategy 1: Direct PDF text
        if content_type == "application/pdf":
            text, pages = self._extract_pdf_text(file_bytes)
            if len(text.strip()) > 50:
                elapsed = (time.time() - start) * 1000
                logger.info(f"Direct PDF text: {len(text)} chars, {pages} pages, {elapsed:.0f}ms")
                return text, 0.98, {
                    "engine_used": "pdf_text_direct", "confidence": 0.98,
                    "paddle_regions": 0, "easy_regions": 0, "merged_regions": 0,
                    "processing_time_ms": round(elapsed, 1),
                    "pages_processed": pages,
                    "fallback_used": False, "image_preprocessed": False,
                }

        # Strategy 2: Smart OCR
        images = file_to_images(file_bytes, content_type)
        if not images:
            return "", 0.0, {"error": "Could not read file"}

        all_text, all_conf = [], []
        rapid_total, easy_total = 0, 0
        engine_used = "none"

        for img in images:
            text, conf, engine, count = self._smart_ocr(img)
            all_text.append(text)
            all_conf.append(conf)
            engine_used = engine
            if engine == "rapidocr":
                rapid_total += count
            else:
                easy_total += count

        raw_text = "\n\n".join(all_text)
        raw_text = self._cleanup_ocr_text(raw_text)
        confidence = sum(all_conf) / len(all_conf) if all_conf else 0.0
        elapsed = (time.time() - start) * 1000

        logger.info(f"OCR: {engine_used}, conf={confidence:.1%}, rapid={rapid_total} easy={easy_total}, {elapsed:.0f}ms")

        return raw_text, round(confidence, 3), {
            "engine_used": engine_used, "confidence": round(confidence, 3),
            "paddle_regions": rapid_total, "easy_regions": easy_total,
            "merged_regions": rapid_total + easy_total,
            "processing_time_ms": round(elapsed, 1),
            "pages_processed": len(images),
            "fallback_used": engine_used == "easyocr",
            "image_preprocessed": True,
        }

    def _smart_ocr(self, image: np.ndarray) -> Tuple[str, float, str, int]:
        # Pass 1: RapidOCR on original
        t1, c1, n1 = self._run_rapid(image)
        if c1 >= 0.85 and n1 > 5:
            logger.info(f"Pass 1 (RapidOCR original): {c1:.1%}, {n1} regions")
            return t1, c1, "rapidocr", n1

        # Pass 2: RapidOCR on enhanced
        enhanced = self._enhance_image(image)
        t2, c2, n2 = self._run_rapid(enhanced)
        if c2 >= 0.70 and n2 > 5:
            logger.info(f"Pass 2 (RapidOCR enhanced): {c2:.1%}, {n2} regions")
            return t2, c2, "rapidocr", n2

        # Best RapidOCR so far
        if c1 >= c2 and n1 > 0:
            best_t, best_c, best_n = t1, c1, n1
        elif n2 > 0:
            best_t, best_c, best_n = t2, c2, n2
        else:
            best_t, best_c, best_n = "", 0.0, 0

        # Pass 3: EasyOCR fallback
        if best_c < 0.70 or best_n < 5:
            preprocessed = preprocess_image(image)
            t3, c3, n3 = self._run_easy(preprocessed)
            if c3 > best_c or best_n < 5:
                logger.info(f"Pass 3 (EasyOCR): {c3:.1%}, {n3} regions")
                return t3, c3, "easyocr", n3

        if best_n > 0:
            return best_t, best_c, "rapidocr", best_n
        return "", 0.0, "none", 0

    def _run_rapid(self, image: np.ndarray) -> Tuple[str, float, int]:
        if self._rapid is None:
            return "", 0.0, 0
        try:
            result, _ = self._rapid(image)
            if not result:
                return "", 0.0, 0
            lines, confidences = [], []
            for item in result:
                if len(item) >= 3:
                    text = str(item[1]).strip()
                    conf = float(item[2])
                    if text:
                        lines.append(text)
                        confidences.append(conf)
            avg = sum(confidences) / len(confidences) if confidences else 0.0
            return "\n".join(lines), round(avg, 3), len(lines)
        except Exception as e:
            logger.error(f"RapidOCR error: {e}")
            return "", 0.0, 0

    def _run_easy(self, image: np.ndarray) -> Tuple[str, float, int]:
        if self._easy is None:
            return "", 0.0, 0
        try:
            result = self._easy.readtext(image)
            lines, confidences = [], []
            for (_, text, conf) in result:
                if text.strip():
                    lines.append(text.strip())
                    confidences.append(conf)
            avg = sum(confidences) / len(confidences) if confidences else 0.0
            return "\n".join(lines), round(avg, 3), len(lines)
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return "", 0.0, 0

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w < 2000:
            scale = 2000 / w
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(enhanced, -1, kernel)

    def _extract_pdf_text(self, pdf_bytes: bytes) -> Tuple[str, int]:
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            for page in doc:
                t = str(page.get_text("text"))
                if t.strip():
                    pages.append(t)
            doc.close()
            return "\n\n".join(pages), len(pages)
        except Exception as e:
            logger.warning(f"PDF text extraction failed: {e}")
            return "", 0

    def _cleanup_ocr_text(self, text: str) -> str:
        # ── Fix ₹ symbol misread as R or T ──
        text = re.sub(r'\bR(\d+[.,]\d+)', r'₹\1', text)
        text = re.sub(r'\bT(\d+[.,]\d+)', r'₹\1', text)

        # ── Fix merged amounts: ₹173.581138.00 → ₹173.58 ₹1138.00 ──
        text = re.sub(r'(\d+\.\d{2})(\d{3,})', r'\1 ₹\2', text)

        # ── Fix amount+percentage merge: 846.629% → 846.62 9% ──
        text = re.sub(r'(\d+\.\d{2})(\d+)%', r'\1 \2%', text)
        text = re.sub(r'%([A-Z])', r'% \1', text)

        # ── Fix CamelCase labels (RapidOCR strips spaces between words) ──
        camel_fixes = [
            ('TaxInvoice', 'Tax Invoice'), ('BillofSupply', 'Bill of Supply'),
            ('CashMemo', 'Cash Memo'), ('OriginalforRecipient', 'Original for Recipient'),
            ('SoldBy', 'Sold By'), ('BillingAddress', 'Billing Address'),
            ('ShippingAddress', 'Shipping Address'), ('InvoiceNumber', 'Invoice Number'),
            ('InvoiceDate', 'Invoice Date'), ('InvoiceDetails', 'Invoice Details'),
            ('OrderNumber', 'Order Number'), ('OrderDate', 'Order Date'),
            ('Placeofsupply', 'Place of supply'), ('Placeofdelivery', 'Place of delivery'),
            ('AmountinWords', 'Amount in Words'), ('State/UTCode', 'State/UT Code'),
            ('GSTRegistrationNo', 'GST Registration No'), ('PANNo', 'PAN No'),
            ('AuthorizedSignatory', 'Authorized Signatory'),
            ('Whethertaxis', 'Whether tax is '), ('payableunder', 'payable under '),
            ('reversecharge', 'reverse charge'),
            ('SellerName', 'Seller Name'), ('BuyerName', 'Buyer Name'),
            ('TotalAmount', 'Total Amount'), ('GrandTotal', 'Grand Total'),
            ('NetAmount', 'Net Amount'), ('TotalTax', 'Total Tax'),
        ]
        for old, new in camel_fixes:
            text = text.replace(old, new)

        # ── Fix colon spacing ──
        text = re.sub(r':([A-Z0-9])', r': \1', text)

        # ── Normalize GST/PAN/FSSAI labels ──
        text = re.sub(r'GSTIN\s*[;:]\s*', 'GSTIN: ', text)
        text = re.sub(r'PAN\s*(?:No\.?)?\s*[;:]\s*', 'PAN: ', text)
        text = re.sub(r'FSSAI\s*[;:]\s*', 'FSSAI: ', text)

        # ── Fix common OCR word errors ──
        text = re.sub(r'\bInvo1ce\b', 'Invoice', text)
        text = re.sub(r'\bTota1\b', 'Total', text)

        return text