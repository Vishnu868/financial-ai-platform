"""
Centralized configuration for the entire application.
All settings in one place. Override via .env file or environment variables.

Usage:
    from app.config import settings
    print(settings.ollama_model)
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── API ───────────────────────────────────────────────────────────────
    app_name: str = "Financial Document AI Platform"
    app_version: str = "2.0.0"
    debug: bool = False

    # ── OCR ───────────────────────────────────────────────────────────────
    ocr_fallback_threshold: float = 0.6
    ocr_use_gpu: bool = False
    ocr_language: str = "en"

    # ── Preprocessing ─────────────────────────────────────────────────────
    min_image_width: int = 1200
    deskew_threshold: float = 0.5

    # ── Extraction ────────────────────────────────────────────────────────
    use_llm_fallback: bool = True
    min_fields_before_llm: int = 4
    confidence_threshold: float = 0.4

    # ── LLM (Ollama — local, free) ────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000

    # ── RAG ───────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./chroma_store"
    chunk_size: int = 400
    chunk_overlap: int = 60
    retrieval_k: int = 3

    # ── Batch Processing ──────────────────────────────────────────────────
    max_batch_size: int = 20

    # ── Fine-tuned Model (LoRA) ──────────────────────────────────────────
    use_finetuned_model: bool = False
    finetuned_model_path: str = "finetuning/models/invoice-lora"
    finetuned_base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./extraction_history.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
