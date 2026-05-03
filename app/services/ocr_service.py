"""
ocr_service.py  —  Universal OCR with 4-layer extraction pipeline
"""

from __future__ import annotations

import cv2
import fitz
import logging
import numpy as np
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from app.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Universal field synonym dictionary
# ─────────────────────────────────────────────────────────────────────────────

FIELD_SYNONYMS: Dict[str, List[str]] = {
    "invoice_number": [
        "invoice no", "invoice number", "invoice #", "invoice no.", "inv no",
        "inv number", "bill no", "bill number", "bill no.", "tax invoice no",
        "tax invoice number", "document no", "document number", "doc no",
        "receipt no", "receipt number", "voucher no", "ref no", "reference no",
        "invoice ref", "bill ref", "challan no", "order invoice", "inv#",
    ],
    "invoice_date": [
        "invoice date", "date of invoice", "bill date", "date", "invoice dt",
        "tax invoice date", "document date", "receipt date", "issued date",
        "date of issue", "billing date", "transaction date", "inv date",
        "date of bill", "order date",
    ],
    "order_number": [
        "order no", "order number", "order id", "order #", "purchase order",
        "po number", "po no", "customer order no", "customer order number",
        "order ref", "so number", "so no", "sales order", "booking no",
        "booking number", "consignment no",
    ],
    "order_date": [
        "order date", "date of order", "purchase date", "booking date",
        "order placed", "ordered on",
    ],
    "seller_gst": [
        "gstin", "gst no", "gst number", "gst registration no",
        "gst registration number", "gstin no", "seller gstin",
        "supplier gstin", "vendor gstin", "our gstin", "company gstin",
        "gstin/uin", "tin no", "vat no", "tax registration no",
        "supply state gstin", "gstn",
    ],
    "seller_pan": [
        "pan", "pan no", "pan number", "pan no.", "permanent account number",
        "seller pan", "vendor pan", "supplier pan", "income tax pan", "it pan",
    ],
    "seller_name": [
        "sold by", "seller name", "seller", "vendor name", "vendor",
        "supplier name", "supplier", "company name", "firm name",
        "business name", "shop name", "merchant name", "brand name",
        "invoice from", "bill from", "details of supplier",
        "restaurant name", "legal entity name", "store name",
    ],
    "seller_info": [
        "details of supplier", "sold by", "seller details", "vendor details",
        "supplier details", "about seller", "seller info",
    ],
    "seller_address": [
        "seller address", "vendor address", "supplier address",
        "registered address", "business address", "company address",
        "ship from address", "dispatch address", "supply address",
        "place of business", "principal place",
    ],
    "billing_address": [
        "bill to", "billing address", "bill to address", "billed to",
        "invoice to", "buyer address", "customer address",
        "ship to", "shipping address", "delivery address",
        "consignee address", "deliver to",
    ],
    "shipping_address": [
        "ship to", "shipping address", "delivery address", "deliver to",
        "consignee address", "ship to address", "shipped to",
        "delivery location", "destination address",
    ],
    "fssai_license": [
        "fssai", "fssai no", "fssai license", "fssai license no",
        "fssai registration", "food license", "fssai lic no",
        "fssai number", "food safety license",
    ],
    "place_of_supply": [
        "place of supply", "state of supply", "supply state",
        "place of supply & state code", "state name & place of supply",
        "state", "supply location",
    ],
    "place_of_delivery": [
        "place of delivery", "delivery state", "destination state",
        "place of receipt",
    ],
    "reverse_charge": [
        "reverse charge", "reverse charge applicable",
        "whether tax is payable under reverse charge",
        "supply attracts reverse charge",
        "whether reverse charges applicable",
    ],
    "total_amount": [
        "total", "grand total", "invoice total", "total amount",
        "amount due", "net payable", "balance due", "net amount due",
        "total invoice value", "final total", "amount payable",
        "total value", "invoice value", "total invoice amount",
        "net total", "balance payable",
    ],
    "total_tax": [
        "total tax", "total taxes", "total gst", "tax amount",
        "total tax amount", "total tax value", "gst amount",
        "total tax charged", "tax total",
    ],
    "subtotal": [
        "subtotal", "sub total", "taxable value", "taxable amount",
        "net amount", "assessable value", "total taxable amount",
        "total taxable value", "base amount",
    ],
    "cgst_amount": [
        "cgst", "cgst amount", "cgst (inr)", "cgst value",
        "central gst", "central tax",
    ],
    "sgst_amount": [
        "sgst", "sgst amount", "sgst (inr)", "sgst value",
        "state gst", "state tax", "sgst/utgst",
    ],
    "igst_amount": [
        "igst", "igst amount", "igst (inr)", "igst value",
        "integrated gst", "integrated tax", "scgst/igst",
    ],
    "discount": [
        "discount", "total discount", "discount amount",
        "discounts", "rebate", "you saved", "savings", "discounts/coupons",
    ],
    "amount_in_words": [
        "amount in words", "amount (in words)", "rupees in words",
        "total in words", "invoice total in words",
        "total invoice value (in words)", "amount in words:",
    ],
    "invoice_type": [
        "invoice type", "type of invoice", "document type",
        "tax invoice", "bill of supply", "credit note", "debit note",
        "original for recipient", "original tax invoice",
    ],
    "payment_method": [
        "payment mode", "payment method", "payment type",
        "mode of payment", "paid by", "payment",
    ],
    "buyer_name": [
        "buyer name", "customer name", "name of customer",
        "recipient name", "legal name", "client name",
    ],
}

FORMAT_EXTRACTORS: Dict[str, str] = {
    "seller_gst":    r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9])\b",
    "seller_pan":    r"\b([A-Z]{5}\d{4}[A-Z])\b",
    "fssai_license": r"\b(\d{14,18})\b",
    "invoice_date": (
        r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b|"
        r"\b(\d{4}-\d{2}-\d{2})\b|"
        r"\b(\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{4})\b"
    ),
    "total_amount": r"(?:Total|Grand\s+Total|Invoice\s+Total)[^\d]*?([\d,]+\.\d{2})",
    "amount_in_words": (
        r"(?:Rs\.|INR|Rupees?\.?)\s*"
        r"((?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|"
        r"Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty|"
        r"Thirty|Forty|Fifty|Sixty|Seventy|Eighty|Ninety|Hundred|Thousand|"
        r"Lakh|Crore)[A-Za-z\s]+(?:Only|Paisa\s+Only)?)"
    ),
}

GSTIN_STATE_CODES: Dict[str, str] = {
    "01": "Jammu & Kashmir",    "02": "Himachal Pradesh",  "03": "Punjab",
    "04": "Chandigarh",         "05": "Uttarakhand",       "06": "Haryana",
    "07": "Delhi",              "08": "Rajasthan",          "09": "Uttar Pradesh",
    "10": "Bihar",              "11": "Sikkim",             "12": "Arunachal Pradesh",
    "13": "Nagaland",           "14": "Manipur",            "15": "Mizoram",
    "16": "Tripura",            "17": "Meghalaya",          "18": "Assam",
    "19": "West Bengal",        "20": "Jharkhand",          "21": "Odisha",
    "22": "Chhattisgarh",       "23": "Madhya Pradesh",     "24": "Gujarat",
    "26": "Dadra & Nagar Haveli", "27": "Maharashtra",      "28": "Andhra Pradesh",
    "29": "Karnataka",          "30": "Goa",                "31": "Lakshadweep",
    "32": "Kerala",             "33": "Tamil Nadu",         "34": "Puducherry",
    "35": "Andaman & Nicobar Islands",                      "36": "Telangana",
    "37": "Andhra Pradesh (New)",                           "97": "Other Territory",
}


# ─────────────────────────────────────────────────────────────────────────────
# Type alias for raw PaddleOCR box entry
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: [ [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], text_str, conf_float ]
RawBox = List[Any]

# Full waterfall return type:
# (text, conf, engine, paddle_n, rapid_n, easy_n, preprocessed, boxes)
WaterfallResult = Tuple[str, float, str, int, int, int, bool, List[RawBox]]


# ─────────────────────────────────────────────────────────────────────────────
# Image pre-processing utilities
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_ocr(image: np.ndarray, aggressive: bool = False) -> np.ndarray:
    """Standard preprocessing pipeline for OCR."""
    h, w = image.shape[:2]

    # Upscale if too small
    if w < 1200:
        scale = 1200 / w
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # Deskew
    gray = _deskew(gray)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if aggressive:
        gray = _adaptive_binarise(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        gray = cv2.filter2D(gray, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
    else:
        gray = cv2.filter2D(gray, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))

    return gray


def _deskew(gray: np.ndarray) -> np.ndarray:
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None:
            return gray
        angles = []
        for line in lines[:50]:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if abs(angle) < 10:
                angles.append(angle)
        if not angles:
            return gray
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.3:
            return gray
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


def _adaptive_binarise(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10
    )


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)  # type: ignore[attr-defined]
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        doc.close()
    except Exception as e:
        logger.error(f"PDF rasterisation failed: {e}")
    return images


def image_bytes_to_numpy(file_bytes: bytes) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            from PIL import Image
            import io
            pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        logger.error(f"Image decode failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PaddleOCR engine
# ─────────────────────────────────────────────────────────────────────────────

class PaddleEngine:
    def __init__(self) -> None:
        self._ocr: Any = None
        self._available = False
        self._init()

    def _init(self) -> None:
        try:
            import paddle  # type: ignore[import]
        except ImportError:
            logger.warning("PaddlePaddle not installed — skipping PaddleOCR")
            return

        try:
            import paddle  # type: ignore[import]
            paddle_version: str = getattr(paddle, "__version__", "0.0.0")
            logger.info(f"PaddlePaddle version: {paddle_version}")
            major = int(paddle_version.split(".")[0]) if paddle_version[0].isdigit() else 99
            if major >= 3:
                logger.warning(
                    f"PaddlePaddle v{paddle_version} is unstable on Windows CPU. "
                    "Downgrade: pip install paddlepaddle==2.6.2"
                )
                return
        except Exception as e:
            logger.warning(f"Paddle version check failed: {e}")

        try:
            from paddleocr import PaddleOCR  # type: ignore[import]
            self._ocr = PaddleOCR(
                use_angle_cls=settings.paddle_use_angle_cls,
                lang=settings.paddle_lang,
                use_gpu=False,
                show_log=False,
                enable_mkldnn=True,
                cpu_threads=4,
                use_tensorrt=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.6,
                rec_batch_num=6,
                drop_score=0.4,
            )
            self._available = True
            logger.info("PaddleOCR initialised (Windows CPU, MKL-DNN)")
        except Exception as e:
            logger.error(f"PaddleOCR init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def run(self, image: np.ndarray) -> Tuple[str, float, int, List[RawBox]]:
        """
        Returns (text, avg_confidence, num_regions, raw_boxes).
        raw_boxes: list of [pts, text, conf] — used by BankStatementExtractor.
        """
        if not self._available or self._ocr is None:
            return "", 0.0, 0, []

        try:
            result = self._ocr.ocr(image, cls=settings.paddle_use_angle_cls)

            if not result or result[0] is None:
                return "", 0.0, 0, []

            lines: List[str] = []
            confidences: List[float] = []
            raw_boxes: List[RawBox] = []

            for item in result[0]:
                if item is None:
                    continue
                if len(item) >= 2:
                    box = item[0]
                    text_tuple = item[1]
                    if isinstance(text_tuple, (list, tuple)) and len(text_tuple) >= 2:
                        text = str(text_tuple[0]).strip()
                        conf = float(text_tuple[1])
                        if text:
                            lines.append(text)
                            confidences.append(conf)
                            raw_boxes.append([box, text, conf])

            if not lines:
                return "", 0.0, 0, []

            avg_conf = round(sum(confidences) / len(confidences), 3)
            return "\n".join(lines), avg_conf, len(lines), raw_boxes

        except Exception as e:
            logger.error(f"PaddleOCR.run error: {e}")
            return "", 0.0, 0, []


# ─────────────────────────────────────────────────────────────────────────────
# RapidOCR engine
# ─────────────────────────────────────────────────────────────────────────────

class RapidEngine:
    def __init__(self) -> None:
        self._ocr: Any = None
        self._available = False
        self._init()

    def _init(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore[import]
            self._ocr = RapidOCR()
            self._available = True
            logger.info("RapidOCR initialised")
        except ImportError:
            logger.warning("rapidocr-onnxruntime not installed")
        except Exception as e:
            logger.warning(f"RapidOCR init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def run(self, image: np.ndarray) -> Tuple[str, float, int]:
        if not self._available or self._ocr is None:
            return "", 0.0, 0
        try:
            result, _ = self._ocr(image)
            if not result:
                return "", 0.0, 0
            lines: List[str] = []
            confidences: List[float] = []
            for item in result:
                if len(item) >= 3:
                    text = str(item[1]).strip()
                    conf = float(item[2])
                    if text:
                        lines.append(text)
                        confidences.append(conf)
            if not lines:
                return "", 0.0, 0
            return "\n".join(lines), round(sum(confidences) / len(confidences), 3), len(lines)
        except Exception as e:
            logger.error(f"RapidOCR error: {e}")
            return "", 0.0, 0


# ─────────────────────────────────────────────────────────────────────────────
# EasyOCR engine
# ─────────────────────────────────────────────────────────────────────────────

class EasyEngine:
    def __init__(self) -> None:
        self._reader: Any = None
        self._available = False
        self._init()

    def _init(self) -> None:
        try:
            import easyocr  # type: ignore[import]
            self._reader = easyocr.Reader(["en"], gpu=settings.ocr_use_gpu, verbose=False)
            self._available = True
            logger.info("EasyOCR initialised")
        except ImportError:
            logger.warning("easyocr not installed")
        except Exception as e:
            logger.warning(f"EasyOCR init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def run(self, image: np.ndarray) -> Tuple[str, float, int]:
        if not self._available or self._reader is None:
            return "", 0.0, 0
        try:
            result = self._reader.readtext(image)
            lines: List[str] = []
            confidences: List[float] = []
            for (_, text, conf) in result:
                if text.strip():
                    lines.append(text.strip())
                    confidences.append(conf)
            if not lines:
                return "", 0.0, 0
            return "\n".join(lines), round(sum(confidences) / len(confidences), 3), len(lines)
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return "", 0.0, 0


# ─────────────────────────────────────────────────────────────────────────────
# OCR result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    text: str
    confidence: float
    engine_used: str
    pages_processed: int
    processing_time_ms: float
    fallback_used: bool
    image_preprocessed: bool
    paddle_regions: int = 0
    rapid_regions: int = 0
    easy_regions: int = 0
    raw_boxes: List[RawBox] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_used":        self.engine_used,
            "confidence":         round(self.confidence, 3),
            "pages_processed":    self.pages_processed,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "fallback_used":      self.fallback_used,
            "image_preprocessed": self.image_preprocessed,
            "paddle_regions":     self.paddle_regions,
            "rapid_regions":      self.rapid_regions,
            "easy_regions":       self.easy_regions,
            "merged_regions":     self.paddle_regions + self.rapid_regions + self.easy_regions,
            "raw_boxes":          self.raw_boxes,  # consumed by BankStatementExtractor
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main OCR service
# ─────────────────────────────────────────────────────────────────────────────

class OCRService:
    """
    4-layer OCR extraction pipeline.
    Initialise once at startup — all engines are loaded eagerly.
    """

    def __init__(self) -> None:
        logger.info("Initialising OCR engines (~10-30s on first run)...")
        t0 = time.time()
        self._paddle = PaddleEngine()
        self._rapid  = RapidEngine()
        self._easy   = EasyEngine()
        elapsed = time.time() - t0
        logger.info(
            f"OCR ready in {elapsed:.1f}s — "
            f"paddle={'✓' if self._paddle.available else '✗'}  "
            f"rapid={'✓' if self._rapid.available else '✗'}  "
            f"easy={'✓' if self._easy.available else '✗'}"
        )

    # ── Public entry point ────────────────────────────────────────────────

    def extract_from_file(
        self,
        file_bytes: bytes,
        content_type: str,
    ) -> Tuple[str, float, Dict[str, Any]]:
        t_start = time.time()

        if content_type == "application/pdf":
            result = self._handle_pdf(file_bytes, t_start)
        elif (content_type or "").startswith("image/"):
            result = self._handle_image(file_bytes, t_start)
        else:
            result = self._handle_pdf(file_bytes, t_start)
            if not result.text.strip():
                result = self._handle_image(file_bytes, t_start)

        text = self._cleanup_text(result.text)
        return text, result.confidence, result.to_dict()

    # ── PDF handling ──────────────────────────────────────────────────────

    def _handle_pdf(self, pdf_bytes: bytes, t_start: float) -> OCRResult:
        text, num_pages = self._pymupdf_text(pdf_bytes)
        if len(text.strip()) >= settings.pdf_min_chars:
            elapsed = (time.time() - t_start) * 1000
            logger.info(f"PDF fast path: {len(text)} chars, {num_pages} pages, {elapsed:.0f}ms")
            return OCRResult(
                text=text, confidence=0.98, engine_used="pdf_text_direct",
                pages_processed=num_pages, processing_time_ms=elapsed,
                fallback_used=False, image_preprocessed=False,
            )
        logger.info("PDF insufficient text → rasterising")
        images = pdf_to_images(pdf_bytes, dpi=300)
        if not images:
            elapsed = (time.time() - t_start) * 1000
            return OCRResult(
                text="", confidence=0.0, engine_used="none",
                pages_processed=0, processing_time_ms=elapsed,
                fallback_used=False, image_preprocessed=False,
            )
        return self._process_images(images, t_start)

    # ── Image handling ────────────────────────────────────────────────────

    def _handle_image(self, file_bytes: bytes, t_start: float) -> OCRResult:
        img = image_bytes_to_numpy(file_bytes)
        if img is None:
            elapsed = (time.time() - t_start) * 1000
            return OCRResult(
                text="", confidence=0.0, engine_used="error",
                pages_processed=0, processing_time_ms=elapsed,
                fallback_used=False, image_preprocessed=False,
            )
        return self._process_images([img], t_start)

    # ── Core pipeline ─────────────────────────────────────────────────────

    def _process_images(self, images: List[np.ndarray], t_start: float) -> OCRResult:
        all_text:   List[str]     = []
        all_conf:   List[float]   = []
        all_boxes:  List[RawBox]  = []
        paddle_total = rapid_total = easy_total = 0
        engine_used  = "none"
        any_pre      = False

        for page_idx, img in enumerate(images):
            (
                page_text, page_conf, page_engine,
                p_cnt, r_cnt, e_cnt, preprocessed, page_boxes,
            ) = self._ocr_waterfall(img, page_idx)

            all_text.append(page_text)
            all_conf.append(page_conf)
            all_boxes.extend(page_boxes)
            paddle_total += p_cnt
            rapid_total  += r_cnt
            easy_total   += e_cnt
            any_pre       = any_pre or preprocessed
            if page_engine != "none":
                engine_used = page_engine

        final_text = "\n\n".join(t for t in all_text if t.strip())
        avg_conf   = round(sum(all_conf) / len(all_conf), 3) if all_conf else 0.0
        elapsed    = (time.time() - t_start) * 1000

        logger.info(
            f"OCR done: engine={engine_used} conf={avg_conf:.1%} "
            f"paddle={paddle_total} rapid={rapid_total} easy={easy_total} "
            f"boxes={len(all_boxes)} {elapsed:.0f}ms"
        )

        return OCRResult(
            text=final_text, confidence=avg_conf, engine_used=engine_used,
            pages_processed=len(images), processing_time_ms=elapsed,
            fallback_used=(easy_total > 0), image_preprocessed=any_pre,
            paddle_regions=paddle_total, rapid_regions=rapid_total,
            easy_regions=easy_total, raw_boxes=all_boxes,
        )

    # ── OCR waterfall ─────────────────────────────────────────────────────

    def _ocr_waterfall(self, image: np.ndarray, page_idx: int) -> WaterfallResult:
        """
        5-pass waterfall. Returns WaterfallResult:
        (text, conf, engine, paddle_n, rapid_n, easy_n, preprocessed, boxes)

        Pass 1: PaddleOCR on original
        Pass 2: PaddleOCR on preprocessed
        Pass 3: RapidOCR  on original
        Pass 4: RapidOCR  on preprocessed
        Pass 5: EasyOCR   on aggressively preprocessed (final fallback)
        """
        HIGH   = settings.rapid_confidence_threshold   # 0.85
        MEDIUM = settings.ocr_confidence_threshold     # 0.75
        MIN_N  = settings.ocr_min_text_regions         # 5

        # Initialise all variables so nothing is ever undefined below
        t1, c1, n1, b1 = "", 0.0, 0, []
        t2, c2, n2, b2 = "", 0.0, 0, []
        t3, c3, n3     = "", 0.0, 0
        t4, c4, n4     = "", 0.0, 0

        # ── Pass 1: PaddleOCR original ────────────────────────────────────
        if self._paddle.available:
            t1, c1, n1, b1 = self._paddle.run(image)
            logger.debug(f"Page {page_idx} Pass1 Paddle orig: {c1:.1%} {n1}r")
            if c1 >= HIGH and n1 >= MIN_N:
                return t1, c1, "paddleocr", n1, 0, 0, False, b1

        # ── Pass 2: PaddleOCR preprocessed ───────────────────────────────
        if self._paddle.available:
            pre2 = preprocess_for_ocr(image, aggressive=False)
            t2, c2, n2, b2 = self._paddle.run(pre2)
            logger.debug(f"Page {page_idx} Pass2 Paddle pre: {c2:.1%} {n2}r")
            if c2 >= MEDIUM and n2 >= MIN_N:
                return t2, c2, "paddleocr", n2, 0, 0, True, b2

        # Best paddle result so far (used later if all passes fail threshold)
        if c1 >= c2:
            best_paddle = (t1, c1, n1, b1)
        else:
            best_paddle = (t2, c2, n2, b2)

        # ── Pass 3: RapidOCR original ─────────────────────────────────────
        if self._rapid.available:
            t3, c3, n3 = self._rapid.run(image)
            logger.debug(f"Page {page_idx} Pass3 Rapid orig: {c3:.1%} {n3}r")
            if c3 >= HIGH and n3 >= MIN_N:
                return t3, c3, "rapidocr", 0, n3, 0, False, []

        # ── Pass 4: RapidOCR preprocessed ────────────────────────────────
        if self._rapid.available:
            pre4 = preprocess_for_ocr(image, aggressive=False)
            t4, c4, n4 = self._rapid.run(pre4)
            logger.debug(f"Page {page_idx} Pass4 Rapid pre: {c4:.1%} {n4}r")
            if c4 >= MEDIUM and n4 >= MIN_N:
                return t4, c4, "rapidocr", 0, n4, 0, True, []

        # ── Best non-EasyOCR candidate ────────────────────────────────────
        # Each tuple: (conf, count, engine, text, paddle_n, rapid_n, boxes)
        candidates: List[Tuple[float, int, str, str, int, int, List[RawBox]]] = []

        if best_paddle[2] > 0:   # paddle had results
            candidates.append((best_paddle[1], best_paddle[2], "paddleocr",
                                best_paddle[0], best_paddle[2], 0, best_paddle[3]))
        if n3 > 0:
            candidates.append((c3, n3, "rapidocr", t3, 0, n3, []))
        if n4 > 0:
            candidates.append((c4, n4, "rapidocr", t4, 0, n4, []))

        if candidates:
            best = max(candidates, key=lambda x: (x[0], x[1]))
            b_conf, b_n, b_eng, b_text, b_p, b_r, b_boxes = best
            if b_conf >= MEDIUM and b_n >= MIN_N:
                logger.debug(f"Page {page_idx} best non-easy ({b_eng}) {b_conf:.1%} {b_n}r")
                return b_text, b_conf, b_eng, b_p, b_r, 0, True, b_boxes

        # ── Pass 5: EasyOCR (final fallback) ─────────────────────────────
        if self._easy.available:
            agg = preprocess_for_ocr(image, aggressive=True)
            t5, c5, n5 = self._easy.run(agg)
            logger.debug(f"Page {page_idx} Pass5 EasyOCR: {c5:.1%} {n5}r")
            if n5 > 0:
                return t5, c5, "easyocr", 0, 0, n5, True, []

        logger.warning(f"Page {page_idx}: all OCR passes failed")
        return "", 0.0, "none", 0, 0, 0, False, []

    # ── PyMuPDF direct text ───────────────────────────────────────────────

    def _pymupdf_text(self, pdf_bytes: bytes) -> Tuple[str, int]:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages: List[str] = []
            for page in doc:
                t = page.get_text("text")  # type: ignore[attr-defined]
                if t.strip():
                    pages.append(t)
            doc.close()
            return "\n\n".join(pages), len(pages)
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
            return "", 0

    # ── Text cleanup ──────────────────────────────────────────────────────

    def _cleanup_text(self, text: str) -> str:
        if not text:
            return text

        # Currency symbol misreads
        text = re.sub(r'\bR\s*(\d+[.,]\d)', r'₹\1', text)
        text = re.sub(r'\bT\s*(\d+[.,]\d)', r'₹\1', text)
        text = re.sub(r'\bz\s*(\d+[.,]\d)', r'₹\1', text)

        # Merged numbers
        text = re.sub(r'(\d+\.\d{2})(\d{3,})', r'\1 \2', text)
        text = re.sub(r'(\d+\.\d{2})(\d+)%',  r'\1 \2%', text)

        # GSTIN/PAN/FSSAI normalisation
        text = re.sub(r'GSTIN\s*[;,]\s*', 'GSTIN: ', text)
        text = re.sub(r'PAN\s*(?:No\.?)?\s*[;,]\s*', 'PAN: ', text)
        text = re.sub(r'FSSAI\s*[;,]\s*', 'FSSAI: ', text)

        # Colon spacing
        text = re.sub(r':([A-Z0-9₹])', r': \1', text)

        # Common OCR errors
        for bad, good in {
            'Invo1ce': 'Invoice', 'Tota1': 'Total', 'lnvoice': 'Invoice',
            'lnv': 'Inv', '0rder': 'Order', 'G5TIN': 'GSTIN',
            'GST1N': 'GSTIN', 'Supp1ier': 'Supplier', 'Supp|ier': 'Supplier',
        }.items():
            text = text.replace(bad, good)

        # CamelCase merges
        for old, new in [
            ('TaxInvoice', 'Tax Invoice'),     ('BillofSupply', 'Bill of Supply'),
            ('SoldBy', 'Sold By'),             ('BillingAddress', 'Billing Address'),
            ('ShippingAddress', 'Shipping Address'), ('InvoiceNumber', 'Invoice Number'),
            ('InvoiceDate', 'Invoice Date'),   ('OrderNumber', 'Order Number'),
            ('GrandTotal', 'Grand Total'),     ('TotalAmount', 'Total Amount'),
            ('PlaceofSupply', 'Place of Supply'), ('PlaceofDelivery', 'Place of Delivery'),
            ('AmountinWords', 'Amount in Words'), ('SellerName', 'Seller Name'),
            ('reversecharge', 'reverse charge'), ('payableunder', 'payable under'),
        ]:
            text = text.replace(old, new)

        return text


# ─────────────────────────────────────────────────────────────────────────────
# Universal key-value extractor
# ─────────────────────────────────────────────────────────────────────────────

class UniversalExtractor:
    """
    Platform-agnostic invoice field extractor.
    Three strategies: label matching, format heuristics, table detection.
    """

    def __init__(self) -> None:
        self._syn_lookup: Dict[str, str] = {}
        for field_name, synonyms in FIELD_SYNONYMS.items():
            for syn in synonyms:
                self._syn_lookup[syn.lower().strip()] = field_name

    def extract(self, text: str) -> Dict[str, Any]:
        results: Dict[str, Dict[str, Any]] = {}

        def _set(field: str, value: str, conf: float, strategy: str) -> None:
            value = value.strip()
            if not value:
                return
            existing = results.get(field)
            if existing is None or conf > existing["confidence"]:
                results[field] = {"value": value, "confidence": conf, "strategy": strategy}

        for label, value, conf in self._extract_label_values(text):
            f = self._match_field(label)
            if f:
                _set(f, value, conf, "label_match")

        for f, value, conf in self._extract_by_format(text):
            _set(f, value, conf, "format_match")

        for f, value, conf in self._extract_from_tables(text):
            _set(f, value, conf, "table_match")

        return {k: v["value"] for k, v in results.items()}

    def _extract_label_values(self, text: str) -> List[Tuple[str, str, float]]:
        pairs: List[Tuple[str, str, float]] = []

        # Pattern 1+2: "Label : Value" on same line
        for m in re.finditer(
            r"^([A-Za-z][A-Za-z0-9\s/\.\-\(\)]{1,50}?)\s*[:\-–]\s*(.{1,200})$",
            text, re.MULTILINE
        ):
            label, value = m.group(1).strip(), m.group(2).strip()
            if len(label) >= 2 and value:
                pairs.append((label, value, 0.90))

        # Pattern 3: label \n : \n value
        for m in re.finditer(
            r"^([A-Za-z][A-Za-z0-9\s/\.\-\(\)]{1,50}?)\s*\n\s*:\s*\n\s*(.{1,200})$",
            text, re.MULTILINE
        ):
            label, value = m.group(1).strip(), m.group(2).strip()
            if len(label) >= 2 and value:
                pairs.append((label, value, 0.85))

        # Pattern 4: consecutive lines
        lines = text.split("\n")
        for i in range(len(lines) - 1):
            a, b = lines[i].strip(), lines[i + 1].strip()
            if (
                a and b and len(a) <= 60
                and not a[-1].isdigit()
                and re.match(r"^[A-Za-z][A-Za-z0-9\s/\.\-\(\)]*$", a)
            ):
                pairs.append((a, b, 0.65))

        return pairs

    def _match_field(self, label: str) -> Optional[str]:
        label_lower = label.lower().strip()

        if label_lower in self._syn_lookup:
            return self._syn_lookup[label_lower]

        try:
            from rapidfuzz import process, fuzz  # type: ignore[import]
            result = process.extractOne(
                label_lower, self._syn_lookup.keys(),
                scorer=fuzz.token_sort_ratio, score_cutoff=80,
            )
            if result:
                return self._syn_lookup[result[0]]
        except ImportError:
            pass

        for syn, field in self._syn_lookup.items():
            if syn in label_lower or label_lower in syn:
                return field

        return None

    def _extract_by_format(self, text: str) -> List[Tuple[str, str, float]]:
        results: List[Tuple[str, str, float]] = []
        for field, pattern in FORMAT_EXTRACTORS.items():
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                value = next((g for g in m.groups() if g is not None), None) or m.group(0)
                if value and value.strip():
                    results.append((field, value.strip(), 0.88))
        return results

    def _extract_from_tables(self, text: str) -> List[Tuple[str, str, float]]:
        results: List[Tuple[str, str, float]] = []
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            cells = self._split_row(lines[i].strip())
            if len(cells) >= 2:
                col_map: Dict[int, str] = {}
                for j, h in enumerate(c.lower().strip() for c in cells):
                    f = self._match_field(h)
                    if f:
                        col_map[j] = f
                if col_map:
                    j = i + 1
                    while j < len(lines):
                        data = self._split_row(lines[j].strip())
                        if not data or len(data) < 2:
                            break
                        for ci, fn in col_map.items():
                            if ci < len(data):
                                val = data[ci].strip()
                                if self._is_amount(fn):
                                    if re.match(r"^[\d,\.]+$", val.replace("₹","").replace("Rs.","").strip()):
                                        results.append((fn, val, 0.75))
                                elif len(val) > 1:
                                    results.append((fn, val, 0.75))
                        j += 1
                    i = j
                    continue
            i += 1
        return results

    @staticmethod
    def _split_row(line: str) -> List[str]:
        if "|" in line:
            return [c.strip() for c in line.split("|") if c.strip()]
        if "\t" in line:
            return [c.strip() for c in line.split("\t") if c.strip()]
        return [c.strip() for c in re.split(r" {3,}", line) if c.strip()]

    @staticmethod
    def _is_amount(field: str) -> bool:
        return field in {
            "total_amount", "total_tax", "subtotal",
            "cgst_amount", "sgst_amount", "igst_amount", "discount",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Top-level convenience function
# ─────────────────────────────────────────────────────────────────────────────

def extract_fields_universal(text: str) -> Dict[str, Any]:
    raw = UniversalExtractor().extract(text)
    amount_fields = {
        "total_amount", "total_tax", "subtotal",
        "cgst_amount", "sgst_amount", "igst_amount", "discount",
    }
    clean: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in amount_fields:
            try:
                clean[k] = float(str(v).replace(",","").replace("₹","").replace("Rs.","").strip())
            except ValueError:
                clean[k] = v
        else:
            clean[k] = v
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_ocr_service_instance: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    global _ocr_service_instance
    if _ocr_service_instance is None:
        _ocr_service_instance = OCRService()
    return _ocr_service_instance


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_test() -> None:
    print("Running OCR service self-test...")
    svc = OCRService()
    print(f"  PaddleOCR: {'✓' if svc._paddle.available else '✗'}")
    print(f"  RapidOCR:  {'✓' if svc._rapid.available else '✗'}")
    print(f"  EasyOCR:   {'✓' if svc._easy.available else '✗'}")

    sample = """
GSTIN : 36AACFY8913A1Z9
PAN   : AACFY8913A
Invoice Number : C27616T250036020
Invoice Date   : 13-May-2025
Total Amount   : 35.00
Grand Total    : Rs. 35.00
"""
    fields = extract_fields_universal(sample)
    print("\nUniversal extractor test:")
    for k, v in sorted(fields.items()):
        print(f"  {k:25s} → {v}")
    print("\nSelf-test complete.")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        print("See docstring for install instructions.")
    else:
        _run_self_test()