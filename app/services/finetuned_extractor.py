"""
Fine-tuned LLM inference service.

Loads the LoRA adapter on top of the base model for invoice extraction.
Uses 4-bit quantization to fit in RTX 3050 (4GB VRAM).

Falls back to Ollama if the fine-tuned model is not available or fails.
"""

import json
import re
import logging
import torch
from typing import Optional
from app.config import settings
from app.models.schemas import (
    InvoiceData, DocumentPlatform,
)

logger = logging.getLogger(__name__)


class FinetunedLLMExtractor:
    """
    Invoice extraction using a locally fine-tuned LoRA model.

    Model loading is lazy — only loads when first called.
    This avoids GPU memory usage if the feature is disabled.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load_failed = False

    def _load_model(self):
        """Load the fine-tuned model with LoRA adapter (lazy, one-time)."""
        if self._loaded or self._load_failed:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            base_model_name = settings.finetuned_base_model
            adapter_path = settings.finetuned_model_path

            logger.info(
                f"Loading fine-tuned model: {base_model_name} + "
                f"adapter from {adapter_path}"
            )

            # 4-bit quantization for inference
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

            # Load LoRA adapter on top
            self._model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=torch.float16,
            )
            self._model.eval()

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                adapter_path, trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._loaded = True
            logger.info("Fine-tuned model loaded successfully")

        except FileNotFoundError:
            logger.warning(
                f"LoRA adapter not found at {settings.finetuned_model_path}. "
                f"Run: python finetuning/finetune_lora.py"
            )
            self._load_failed = True
        except ImportError as e:
            logger.warning(f"Missing dependency for fine-tuned model: {e}")
            self._load_failed = True
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            self._load_failed = True

    def extract(
        self, raw_text: str, platform: DocumentPlatform
    ) -> Optional[InvoiceData]:
        """
        Run inference with the fine-tuned model.

        Args:
            raw_text: OCR text from document
            platform: Detected platform

        Returns:
            InvoiceData or None if model unavailable
        """
        self._load_model()

        if not self._loaded:
            return None

        try:
            # Build prompt in same format used during training
            prompt = (
                f"### Instruction:\n"
                f"Extract structured invoice data from the following OCR text. "
                f"Return only valid JSON.\n\n"
                f"### Input:\n{raw_text[:2500]}\n\n"
                f"### Response:\n"
            )

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode only the generated part (skip the prompt)
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            response = self._tokenizer.decode(generated, skip_special_tokens=True)

            logger.info(f"Fine-tuned model response: {len(response)} chars")
            return self._parse_response(response, platform)

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory during fine-tuned inference")
            self._cleanup_gpu()
            return None
        except Exception as e:
            logger.error(f"Fine-tuned extraction failed: {e}")
            return None

    def _parse_response(
        self, text: str, platform: DocumentPlatform
    ) -> Optional[InvoiceData]:
        """Parse JSON from model response into flat InvoiceData."""
        try:
            text = text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

            json_match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in fine-tuned model response")
                return None

            d = json.loads(json_match.group())

            return InvoiceData(
                platform=platform,
                billing_address=d.get("billing_address"),
                shipping_address=d.get("shipping_address"),
                invoice_type=d.get("invoice_type"),
                order_number=d.get("order_number"),
                invoice_number=d.get("invoice_number"),
                order_date=d.get("order_date"),
                invoice_details=d.get("invoice_details"),
                invoice_date=d.get("invoice_date"),
                seller_info=d.get("seller_info"),
                seller_pan=d.get("seller_pan"),
                seller_gst=d.get("seller_gst") or d.get("seller_gstin"),
                fssai_license=d.get("fssai_license"),
                billing_state_code=d.get("billing_state_code"),
                shipping_state_code=d.get("shipping_state_code"),
                place_of_supply=d.get("place_of_supply"),
                place_of_delivery=d.get("place_of_delivery"),
                reverse_charge=d.get("reverse_charge"),
                amount_in_words=d.get("amount_in_words"),
                seller_name=d.get("seller_name"),
                seller_address=d.get("seller_address"),
                total_tax=self._safe_float(d.get("total_tax")),
                total_amount=self._safe_float(d.get("total_amount")),
                buyer_name=d.get("buyer_name"),
                subtotal=self._safe_float(d.get("subtotal")),
                cgst_amount=self._safe_float(d.get("cgst") or d.get("cgst_amount")),
                sgst_amount=self._safe_float(d.get("sgst") or d.get("sgst_amount")),
                igst_amount=self._safe_float(d.get("igst") or d.get("igst_amount")),
                discount=self._safe_float(d.get("discount")),
                delivery_charge=self._safe_float(d.get("delivery_charge")),
                payment_method=d.get("payment_method"),
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse fine-tuned model output: {e}")
            return None

    def _safe_float(self, value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _cleanup_gpu(self):
        """Free GPU memory after OOM error."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    def is_available(self) -> bool:
        """Check if the fine-tuned model can be loaded."""
        self._load_model()
        return self._loaded
