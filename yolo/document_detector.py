"""
Task 7: YOLO-based Document Region Detection

Uses YOLOv8 to detect and classify regions in invoice images:
- Header, Logo, Table, Footer, Stamp, Signature, QR Code

This helps the OCR pipeline by:
1. Identifying WHERE important data is (table region → OCR that area)
2. Classifying document type by visible elements
3. Skipping noise regions (logos, stamps, QR codes)

Install: pip install ultralytics

Usage:
    from task7_yolo.document_detector import DocumentDetector
    detector = DocumentDetector()
    regions = detector.detect("invoice.jpg")
    for r in regions:
        print(f"{r['class']}: {r['confidence']:.2f} at {r['bbox']}")
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentDetector:
    """
    YOLO-based document layout detection.
    
    Uses a pre-trained YOLOv8 model to detect common document regions.
    For invoice processing, the key regions are:
    - 'table': Contains line items, amounts — highest value for extraction
    - 'header': Contains invoice number, date, seller info
    - 'text': General text blocks
    - 'figure': Logos, images — can be skipped for OCR
    """

    # Document layout classes (DocLayNet / PubLayNet style)
    CLASSES = [
        "text", "title", "list", "table", "figure",
        "header", "footer", "caption", "page_number",
    ]

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to custom YOLO model weights.
                       If None, uses pre-trained YOLOv8 model.
        """
        self.model = None
        try:
            from ultralytics import YOLO

            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
                logger.info(f"YOLO loaded custom model: {model_path}")
            else:
                # Use pre-trained YOLOv8n for general object detection
                # For production, train on DocLayNet dataset for document layout
                self.model = YOLO("yolov8n.pt")
                logger.info("YOLO loaded pre-trained YOLOv8n")

        except ImportError:
            logger.warning("ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            logger.error(f"YOLO init failed: {e}")

    def detect(
        self, image: np.ndarray, confidence_threshold: float = 0.25
    ) -> List[Dict]:
        """
        Detect document regions in an image.

        Args:
            image: OpenCV image (BGR)
            confidence_threshold: Minimum confidence to include

        Returns:
            List of detected regions with class, confidence, bbox
        """
        if self.model is None:
            return []

        results = self.model(image, conf=confidence_threshold, verbose=False)
        regions = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

                regions.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "area": int((x2 - x1) * (y2 - y1)),
                })

        # Sort by area (largest first — tables are usually biggest)
        regions.sort(key=lambda r: r["area"], reverse=True)
        logger.info(f"YOLO detected {len(regions)} regions")

        return regions

    def crop_region(
        self, image: np.ndarray, bbox: List[int], padding: int = 10
    ) -> np.ndarray:
        """Crop a detected region from the image with optional padding."""
        h, w = image.shape[:2]
        x1 = max(0, bbox[0] - padding)
        y1 = max(0, bbox[1] - padding)
        x2 = min(w, bbox[2] + padding)
        y2 = min(h, bbox[3] + padding)
        return image[y1:y2, x1:x2]

    def detect_and_visualize(
        self, image: np.ndarray, output_path: str = "detected.jpg"
    ) -> np.ndarray:
        """Detect regions and draw bounding boxes for visualization."""
        regions = self.detect(image)
        vis = image.copy()

        colors = {
            "text": (0, 255, 0), "title": (255, 0, 0),
            "table": (0, 0, 255), "figure": (255, 255, 0),
            "header": (255, 0, 255), "footer": (0, 255, 255),
        }

        for r in regions:
            color = colors.get(r["class"], (128, 128, 128))
            x1, y1, x2, y2 = r["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{r['class']} {r['confidence']:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(output_path, vis)
        return vis


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STANDALONE DEMO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_detector.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1])
    if image is None:
        print(f"Could not read: {sys.argv[1]}")
        sys.exit(1)

    detector = DocumentDetector()
    regions = detector.detect(image)

    print(f"\nDetected {len(regions)} regions:")
    for r in regions:
        print(f"  {r['class']:15s} conf={r['confidence']:.2f} bbox={r['bbox']} area={r['area']}")

    detector.detect_and_visualize(image, "detected_output.jpg")
    print("\nVisualization saved to detected_output.jpg")
