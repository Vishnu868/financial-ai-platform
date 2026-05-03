# INTERNSHIP TASKS — Complete Status & File Map

## ALL 15 TASKS

| # | Task | Status | Files |
|---|------|--------|-------|
| 1 | Amazon/Flipkart extraction | ✅ Done | `app/services/extraction_service.py` |
| 2 | Manual extraction 5-9 brands | ✅ Done | Same file — 9 platforms (Amazon, Flipkart, Swiggy, Zomato, Meesho, Myntra, BigBasket, Blinkit, JioMart) |
| 3 | Universal extractor | ✅ Done | `extraction_service.py` — 22 fields for any invoice format |
| 4 | OCRs (multiple engines) | ✅ Done | `task4-ocr-upgrade/ocr_service.py` — RapidOCR + EasyOCR with confidence fallback + PyMuPDF for PDFs |
| 5 | Bank statement | ✅ Done | `app/services/bank_extractor.py` + `app/routers/bank_statement.py` |
| 6 | Image processing erosion/dilation | ✅ Done | `app/services/preprocess.py` — OpenCV pipeline (grayscale, upscale, denoise, CLAHE, deskew) |
| 7 | YOLO | ✅ Done | `task7-yolo/document_detector.py` — YOLOv8 document region detection |
| 8 | GenAI chatbot (Streamlit + LangChain + FastAPI) | ✅ Done | `chatbot/app.py` — Streamlit UI with LangChain + Ollama |
| 9 | RAG chatbot improvement | ✅ Done | `chatbot/rag_engine.py` — ChromaDB + sentence-transformers |
| 10 | Document classification ML/DL | ✅ Done | `app/routers/classify.py` — keyword-based + score-based classification |
| 11 | Combined FastAPI app | ✅ Done | `app/main.py` — all routers (invoice, bank, batch, export, classify, health) |
| 12 | Transformer from scratch | ✅ Done | `task12-transformer/transformer_from_scratch.py` — complete implementation |
| 13 | Architecture of Mistral | ✅ Done | `task13-mistral-arch/MISTRAL_ARCHITECTURE.md` — SWA, GQA, RoPE, SwiGLU |
| 14 | Architecture of LLaMA | ✅ Done | `task14-llama-arch/LLAMA_ARCHITECTURE.md` — Pre-RMSNorm, SwiGLU, RoPE, GQA |
| 15 | Fine-tuning GPT-2 for finance | ✅ Done | `task15-gpt2-finetune/finetune_gpt2_finance.py` — 20 Indian financial texts |

## HOW TO RUN EACH TASK

### Task 4 — RapidOCR (replace ocr_service.py in your project)
```bash
pip install rapidocr-onnxruntime
# Copy task4-ocr-upgrade/ocr_service.py → app/services/ocr_service.py
```

### Task 7 — YOLO
```bash
pip install ultralytics
python task7-yolo/document_detector.py <image_path>
```

### Task 12 — Transformer
```bash
pip install torch
python task12-transformer/transformer_from_scratch.py
```

### Task 15 — GPT-2 Fine-tuning
```bash
pip install transformers datasets accelerate torch
python task15-gpt2-finetune/finetune_gpt2_finance.py
```

### Tasks 13 & 14 — Read the markdown files
Open `MISTRAL_ARCHITECTURE.md` and `LLAMA_ARCHITECTURE.md`
