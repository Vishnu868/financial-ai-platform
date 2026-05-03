"""
Financial Document AI Platform v2.0 — Main Application

Start with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then open:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from app.routers import yolo_router

from app.config import settings
from app.routers import (
    invoice,
    bank_statement,
    classify,
    batch,
    export,
    health,
    yolo_router,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy library loggers
for lib in ["paddleocr", "ppocr", "paddle", "urllib3", "httpcore"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logging.info(f"Starting {settings.app_name} v{settings.app_version}")
    logging.info(
        f"Config: LLM_fallback={settings.use_llm_fallback}, "
        f"GPU={settings.ocr_use_gpu}, "
        f"Model={settings.ollama_model}"
    )
    yield
    logging.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="""
## Financial Document AI Platform v2.0

Production-grade automated extraction from Indian financial documents.

### Architecture
Image/PDF → Preprocess (OpenCV) → Ensemble OCR (Paddle+Easy)
→ Classify → Extract (Regex + LLM fallback) → Validate → JSON

### Key Differentiators
- **Ensemble OCR**: PaddleOCR + EasyOCR merged by bounding box IoU
- **LLM Fallback**: Local Mistral (via Ollama) for messy/unusual invoices
- **GSTIN Validation**: Format + state code + checksum verification
- **Tax Math Check**: CGST + SGST = Total Tax, CGST == SGST intra-state
- **Total Verification**: Subtotal + Tax - Discount + Delivery = Total
- **Batch Processing**: ZIP upload for multiple invoices
- **CSV Export**: Download extracted data as spreadsheet

### Supported Platforms
Amazon, Flipkart, Meesho, Myntra, Swiggy, Zomato, BigBasket, Blinkit, JioMart

### No External APIs
Everything runs locally: OCR, LLM, embeddings, vector store.
No API keys needed. No data leaves your machine.
    """,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow React/Streamlit frontends to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(
    health.router,
    tags=["System"],
)
app.include_router(
    invoice.router,
    prefix="/api/v2/invoice",
    tags=["Invoice Extraction"],
)
app.include_router(
    bank_statement.router,
    prefix="/api/v2/bank",
    tags=["Bank Statement"],
)
app.include_router(
    classify.router,
    prefix="/api/v2/classify",
    tags=["Classification"],
)
app.include_router(
    batch.router,
    prefix="/api/v2/batch",
    tags=["Batch Processing"],
)
app.include_router(
    export.router,
    prefix="/api/v2/export",
    tags=["Export"],
)
app.include_router(
    yolo_router.router,
    tags=["YOLO Detection"],
)
app.include_router(yolo_router.router) 

# Serve production frontend UI
# Access at: http://localhost:8000/ui
from fastapi.responses import FileResponse
from pathlib import Path

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/ui", include_in_schema=False)
async def serve_ui():
    """Serve the production frontend UI."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return {"error": "Frontend not found. Expected at frontend/index.html"}