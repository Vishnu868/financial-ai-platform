"""
OCR Engine Benchmark — compare PaddleOCR vs EasyOCR vs Ensemble.

Usage:
    python scripts/benchmark_ocr.py [test_dir]

Default test directory: data/test_invoices/
Outputs a comparison table. Screenshot for your README.
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark(test_dir: str = "data/test_invoices"):
    import cv2
    from app.services.preprocess import preprocess_image
    from app.services.ocr_service import OCRService

    if not os.path.exists(test_dir):
        print(f"Directory not found: {test_dir}")
        print("Put test invoice images in data/test_invoices/ and try again.")
        return

    files = [
        f for f in sorted(os.listdir(test_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not files:
        print(f"No image files found in {test_dir}")
        return

    print("Loading OCR engines...")
    ocr = OCRService()

    print(f"\nBenchmarking {len(files)} files from {test_dir}\n")
    print(f"{'File':<30} {'Paddle':<12} {'Easy':<12} {'Merged':<12} {'Conf':<10} {'Time':<8}")
    print("=" * 84)

    total_time = 0
    for filename in files:
        filepath = os.path.join(test_dir, filename)

        start = time.time()
        with open(filepath, "rb") as f:
            file_bytes = f.read()

        text, conf, meta = ocr.extract_from_file(file_bytes, "image/jpeg")
        elapsed = time.time() - start
        total_time += elapsed

        print(
            f"{filename[:28]:<30} "
            f"{meta.get('paddle_regions', 0):>4} rgns   "
            f"{meta.get('easy_regions', 0):>4} rgns   "
            f"{meta.get('merged_regions', 0):>4} rgns   "
            f"{conf:>6.1%}   "
            f"{elapsed:>5.1f}s"
        )

    print("=" * 84)
    print(f"Total time: {total_time:.1f}s | Avg: {total_time/len(files):.1f}s per file")


if __name__ == "__main__":
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "data/test_invoices"
    benchmark(test_dir)
