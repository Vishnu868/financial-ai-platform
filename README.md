# Financial Document AI Platform v2.0

Production-grade automated extraction of structured data from Indian financial documents using Ensemble OCR, LLM fallback, and post-extraction validation.

## Architecture

```
Image/PDF → Preprocess (OpenCV) → Ensemble OCR (PaddleOCR + EasyOCR)
  → Classify → Extract (Regex + LLM fallback) → Validate → JSON Response
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Ensemble OCR** | PaddleOCR + EasyOCR run in parallel, merged by bounding box IoU — 10-15% more accurate than single engine |
| **LLM Fallback** | Local Mistral (via Ollama) extracts fields when regex gets < 4 results |
| **GSTIN Validation** | Format check + state code verification + checksum algorithm |
| **Tax Math Check** | Verifies CGST + SGST = Total Tax, CGST == SGST for intra-state |
| **Total Verification** | Cross-checks Subtotal + Tax - Discount + Delivery = Total |
| **Batch Processing** | ZIP upload for multiple invoices with background processing |
| **RAG Chatbot** | Ask natural language questions about uploaded documents |
| **CSV Export** | Download all extracted data as spreadsheet |

## Supported Platforms

Amazon, Flipkart, Meesho, Myntra, Swiggy, Zomato, BigBasket, Blinkit, JioMart

## Tech Stack

| Layer | Technology |
|-------|-----------|
| OCR | PaddleOCR + EasyOCR (ensemble merge by IoU) |
| Preprocessing | OpenCV (adaptive threshold, erosion/dilation, deskew) |
| Backend | FastAPI + Pydantic v2 |
| Validation | Custom GSTIN checksum + tax math + total cross-check |
| RAG | LangChain + ChromaDB + sentence-transformers |
| LLM | Mistral 7B via Ollama (local, free, no API keys) |
| Chatbot UI | Streamlit |
| Testing | Pytest (30+ tests) |

**No external APIs. No API keys. Everything runs locally.**

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/financial-ai-platform.git
cd financial-ai-platform

# 2. Setup
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# 3. Install Ollama + Mistral (optional, for LLM features)
# Download from ollama.com, then:
ollama pull mistral

# 4. Run API
uvicorn app.main:app --reload

# 5. Open Swagger UI
# http://localhost:8000/docs

# 6. Run Chatbot (new terminal)
streamlit run chatbot/app.py

# 7. Run Tests
pytest tests/ -v
```

## API Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/api/v2/invoice/extract` | POST | Extract invoice data (ensemble OCR + validation) |
| `/api/v2/invoice/platforms` | GET | List supported platforms and features |
| `/api/v2/bank/extract` | POST | Extract bank statement data |
| `/api/v2/classify/document` | POST | Auto-classify document type |
| `/api/v2/batch/process` | POST | Batch process ZIP of invoices |
| `/api/v2/batch/status/{id}` | GET | Check batch job progress |
| `/api/v2/export/csv` | POST | Export invoices to CSV |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

## API Testing (curl)

### Extract an invoice:
```bash
curl -X POST "http://localhost:8000/api/v2/invoice/extract" \
  -F "file=@data/test_invoices/amazon_invoice.pdf"
```

### Classify a document:
```bash
curl -X POST "http://localhost:8000/api/v2/classify/document" \
  -F "file=@data/test_invoices/some_document.jpg"
```

### Batch process:
```bash
curl -X POST "http://localhost:8000/api/v2/batch/process" \
  -F "file=@invoices_batch.zip"
```

### Check batch status:
```bash
curl "http://localhost:8000/api/v2/batch/status/abc123def456"
```

### Export to CSV:
```bash
curl -X POST "http://localhost:8000/api/v2/export/csv" \
  -H "Content-Type: application/json" \
  -d '[{"platform":"amazon","invoice_number":"IN-123","total_amount":1294.46,"currency":"INR","fields_extracted":5,"fields_total":16,"validation_warnings":[]}]' \
  -o invoices.csv
```

### Health check:
```bash
curl http://localhost:8000/health
```

## Sample Response

```json
{
  "status": "success",
  "message": "Successfully extracted 10 fields from amazon invoice",
  "document_type": "invoice",
  "platform": "amazon",
  "confidence_score": 0.847,
  "extracted_data": {
    "invoice_number": "IN-12345678",
    "order_id": "408-1234567-8901234",
    "invoice_date": "15/01/2025",
    "seller": {
      "name": "TechStore India Pvt Ltd",
      "gstin": "29AABCT1234F1Z5",
      "state": "Karnataka"
    },
    "buyer": {
      "name": "Nani",
      "address": "Hyderabad, Telangana"
    },
    "subtotal": 1097.0,
    "tax": {
      "cgst_amount": 98.73,
      "sgst_amount": 98.73,
      "total_tax": 197.46,
      "is_validated": true
    },
    "total_amount": 1294.46,
    "payment_method": "Amazon Pay"
  },
  "ocr_metadata": {
    "engine_used": "ensemble",
    "confidence": 0.891,
    "paddle_regions": 42,
    "easy_regions": 38,
    "merged_regions": 45,
    "processing_time_ms": 2340.5,
    "fallback_used": false
  },
  "validation_passed": true,
  "warnings": []
}
```

## Validation Example

When extraction has issues, the validator catches them:

```json
{
  "validation_passed": false,
  "warnings": [
    "GSTIN '29AABCT1234F1Z3' failed checksum — expected check digit '5', got '3'. Likely OCR misread.",
    "CGST (₹98.73) ≠ SGST (₹108.73) — for intra-state supply these must be equal."
  ]
}
```

## Project Structure

```
financial-ai-platform/
├── app/                          # Main application
│   ├── config.py                 # Centralized settings
│   ├── main.py                   # FastAPI app entry point
│   ├── models/schemas.py         # All Pydantic data models
│   ├── routers/                  # API endpoints
│   │   ├── invoice.py            # Invoice extraction
│   │   ├── bank_statement.py     # Bank statement extraction
│   │   ├── classify.py           # Document classification
│   │   ├── batch.py              # Batch processing
│   │   ├── export.py             # CSV export
│   │   └── health.py             # Health checks
│   └── services/                 # Business logic
│       ├── preprocess.py         # OpenCV image preprocessing
│       ├── ocr_service.py        # Ensemble OCR engine
│       ├── extraction_service.py # Regex field extraction
│       ├── validator.py          # Post-extraction validation
│       ├── llm_extractor.py      # LLM fallback (Ollama/Mistral)
│       └── bank_extractor.py     # Bank statement extraction
├── chatbot/                      # Streamlit RAG chatbot
│   ├── rag_engine.py             # RAG pipeline
│   └── app.py                    # Streamlit UI
├── tests/                        # Pytest test suite (30+ tests)
├── scripts/benchmark_ocr.py      # OCR engine benchmarking
├── requirements.txt
├── setup.md                      # Detailed setup guide
└── README.md
```

## How It Compares

| Feature | Basic Approach | This Project |
|---------|---------------|-------------|
| OCR | Single engine | Ensemble (Paddle + Easy) merged by IoU |
| Extraction | Regex only | Regex + LLM fallback |
| Validation | None | GSTIN checksum + tax math + total cross-check |
| Platforms | 1-2 | 9 platforms |
| Bank statements | No | Yes, with analytics |
| Batch processing | No | Yes, ZIP upload |
| Export | No | CSV download |
| Tests | Manual curl | 30+ pytest cases |
| Chatbot | No | RAG with ChromaDB + Mistral |
