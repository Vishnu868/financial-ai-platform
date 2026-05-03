"""
Image preprocessing pipeline for OCR optimization.

Pipeline:
1. PDF → images (via PyMuPDF)
2. Grayscale conversion
3. Upscale small images (below min_image_width)
4. Deskew (correct rotation using minAreaRect)
5. Adaptive thresholding (handle uneven lighting/shadows)
6. Morphological operations:
   - Opening (erode→dilate): removes small noise dots
   - Closing (dilate→erode): fills small gaps in characters
7. Non-local means denoising

Each step has a specific purpose for Indian invoice processing:
- Phone photos have uneven lighting → adaptive threshold
- Scanned documents are often slightly rotated → deskew
- Swiggy/Zomato screenshots have noise → morphological ops
"""

import cv2
import numpy as np
import logging
from typing import List, Optional


logger = logging.getLogger(__name__)


def pdf_to_images(pdf_bytes: bytes) -> List[np.ndarray]:
    """
    Convert each page of a PDF to a high-resolution numpy image.
    Uses PyMuPDF (fitz) with 2x zoom for ~300 DPI equivalent.
    """
    import fitz  # PyMuPDF

    images = []
    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            # 2x zoom = approximately 300 DPI
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
        pdf_doc.close()
    except Exception as e:
        logger.error(f"PDF to images failed: {e}")

    return images


def bytes_to_image(file_bytes: bytes) -> Optional[np.ndarray]:
    """Convert raw image bytes (JPEG/PNG) to numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def file_to_images(file_bytes: bytes, content_type: str) -> List[np.ndarray]:
    """
    Convert any supported file type to a list of images.
    PDFs produce multiple images (one per page).
    Images produce a single-element list.
    """
    if content_type == "application/pdf":
        return pdf_to_images(file_bytes)
    else:
        img = bytes_to_image(file_bytes)
        if img is not None:
            return [img]
        return []


# REPLACE WITH
def preprocess_image(
    image: np.ndarray,
    min_width: int = 1200,
    deskew_threshold: float = 0.5,
) -> np.ndarray:
    # Step 1: Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 2: Upscale — always scale to at least 2x for screenshots
    height, width = gray.shape
    if width < min_width:
        scale = min_width / width
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        logger.debug(f"Upscaled from {width}px to {int(width * scale)}px")
    elif width < 2000:
        # Screenshots are often 1200-1800px — upscale to 2x for better OCR
        gray = cv2.resize(
            gray, None, fx=2.0, fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )
        logger.debug(f"2x upscale applied: {width}px → {width*2}px")

    # Step 3: Deskew
    gray = _deskew(gray, deskew_threshold)

    # Step 4: Denoise BEFORE thresholding — preserves fine character strokes
    denoised = cv2.fastNlMeansDenoising(gray, h=7)

    # Step 5: Sharpen — recovers edges lost in screenshots
    kernel_sharpen = np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ])
    sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)

    # Step 6: Adaptive threshold with larger block size for screenshots
    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,   # larger block = handles more uneven lighting
        10,   # higher constant = cleaner result for screenshots
    )

    # Step 7: Light morphological cleanup — avoid destroying thin characters
    kernel_small = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

    return cleaned


def _deskew(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Detect and correct rotation angle in scanned/photographed documents.
    Uses minAreaRect on dark pixel coordinates to find the dominant text angle.

    Args:
        image: Grayscale image
        threshold: Minimum angle in degrees to trigger correction

    Returns:
        Deskewed image (or original if angle is below threshold)
    """
    # Find all dark pixels (text)
    coords = np.column_stack(np.where(image < 128))

    if len(coords) < 100:
        return image  # Not enough text to detect angle

    # Get the minimum area rectangle around text pixels
    angle = cv2.minAreaRect(coords)[-1]

    # Normalize angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only rotate if angle exceeds threshold
    if abs(angle) < threshold:
        return image

    logger.debug(f"Deskewing by {angle:.2f} degrees")

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated