# Setup Guide — Financial Document AI Platform v2.0

## Prerequisites

| Software | Version | Download |
|----------|---------|----------|
| Python | 3.10 or 3.11 | python.org |
| Git | Latest | git-scm.com |
| VS Code (recommended) | Latest | code.visualstudio.com |
| Ollama (for LLM features) | Latest | ollama.com |

---

## Step 1: Clone or Create the Project

```bash
# Option A: Clone from GitHub
git clone https://github.com/YOUR_USERNAME/financial-ai-platform.git
cd financial-ai-platform

# Option B: If you have the files already
cd financial-ai-platform
```

## Step 2: Create Virtual Environment

```bash
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This takes 10-15 minutes (downloads OCR models, ML libraries, etc.).

**If PaddleOCR fails on Windows:**
```bash
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
pip install paddleocr
```

**If you get "No module named pydantic_settings":**
```bash
pip install pydantic-settings
```

## Step 4: Setup Environment File

```bash
cp .env.example .env
# Edit .env if needed (defaults work fine)
```

## Step 5: Install Ollama + Mistral (Optional but Recommended)

1. Download Ollama from ollama.com
2. Install and run:
```bash
ollama pull mistral
```
3. Verify it works:
```bash
ollama run mistral "Hello, what is GST?"
# Should get a response. Type /bye to exit.
```

If you don't install Ollama, the system still works — it just skips the LLM
fallback and uses regex-only extraction. Set `USE_LLM_FALLBACK=false` in `.env`.

## Step 6: Get Test Invoices

Put invoice images/PDFs in `data/test_invoices/`:

- **Your own orders:** Amazon.in → Orders → View Invoice → Download PDF
- **Flipkart:** My Orders → Invoice → Download
- **Swiggy/Zomato:** Past Orders → View Bill → Screenshot
- **Online samples:** Google "Amazon India invoice sample PDF"

## Step 7: Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

First run takes 30-60 seconds (OCR models download automatically).

Open browser: **http://localhost:8000/docs**

You should see the Swagger UI with all 6 endpoints.

## Step 8: Run the Chatbot (Optional)

Open a **new terminal** (keep the API running):

```bash
cd financial-ai-platform
source venv/bin/activate  # Windows: venv\Scripts\activate
streamlit run chatbot/app.py
```

Opens at: **http://localhost:8501**

## Step 9: Run Tests

```bash
pytest tests/ -v
```

Should show all tests passing.

## Step 10: Run OCR Benchmark (Optional)

```bash
python scripts/benchmark_ocr.py
```

Requires test images in `data/test_invoices/`.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named paddleocr` | `pip install paddleocr paddlepaddle` |
| `No module named pydantic_settings` | `pip install pydantic-settings` |
| `No module named cv2` | `pip install opencv-python-headless` |
| `No module named fitz` | `pip install pymupdf` |
| `Cannot connect to Ollama` | Run `ollama serve` in another terminal |
| `Port 8000 already in use` | Kill it: `lsof -ti:8000 \| xargs kill -9` |
| PaddleOCR returns empty | Image too small, try a PDF instead |
| Import errors between files | Check all `__init__.py` files exist |

---

## Project Structure After Setup

```
financial-ai-platform/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── invoice.py
│   │   ├── bank_statement.py
│   │   ├── classify.py
│   │   ├── batch.py
│   │   ├── export.py
│   │   └── health.py
│   └── services/
│       ├── __init__.py
│       ├── preprocess.py
│       ├── ocr_service.py
│       ├── extraction_service.py
│       ├── validator.py
│       ├── llm_extractor.py
│       └── bank_extractor.py
├── chatbot/
│   ├── rag_engine.py
│   └── app.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_extraction.py
│   ├── test_validation.py
│   ├── test_bank.py
│   └── test_api.py
├── scripts/
│   └── benchmark_ocr.py
├── data/
│   └── test_invoices/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── setup.md
└── README.md
```
