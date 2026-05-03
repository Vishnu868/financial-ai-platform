"""
LLM-based extraction fallback — updated for mentor's 22-field template.

Sends raw OCR text to local Mistral (Ollama) with a prompt that asks
for all 22 fields from the mentor's Output Template.
"""

import json
import re
import requests
import logging
from typing import Optional
from app.config import settings
from app.models.schemas import InvoiceData, DocumentPlatform

logger = logging.getLogger(__name__)


class LLMExtractor:
    """Extract invoice fields using local LLM when regex fails."""

    PROMPT_TEMPLATE = """You are a financial document data extractor specialized in Indian invoices.
Extract ALL the following fields from the invoice text below.

RULES:
- Return ONLY valid JSON. No explanation. No markdown. No preamble.
- Use null for any field you cannot find.
- For amounts, return numbers only (no currency symbols, no commas).
- For GSTIN, return the exact 15-character string.
- For PAN, return the exact 10-character string.
- For dates, return in the format found in the document.
- For addresses, include full text including city, state, pincode.

REQUIRED JSON FORMAT (22 fields):
{{"billing_address": "string or null", "shipping_address": "string or null", "invoice_type": "Tax Invoice or Bill of Supply or null", "order_number": "string or null", "invoice_number": "string or null", "order_date": "string or null", "invoice_details": "string or null", "invoice_date": "string or null", "seller_info": "full seller block text or null", "seller_pan": "10-char PAN or null", "seller_gst": "15-char GSTIN or null", "fssai_license": "14-digit number or null", "billing_state_code": "2-digit code or null", "shipping_state_code": "2-digit code or null", "place_of_supply": "string or null", "place_of_delivery": "string or null", "reverse_charge": "Yes or No or null", "amount_in_words": "string or null", "seller_name": "string or null", "seller_address": "string or null", "total_tax": number or null, "total_amount": number or null, "buyer_name": "string or null", "subtotal": number or null, "cgst_amount": number or null, "sgst_amount": number or null, "igst_amount": number or null, "discount": number or null, "delivery_charge": number or null, "payment_method": "string or null"}}

INVOICE TEXT:
{text}

JSON:"""

    def extract(
        self, raw_text: str, platform: DocumentPlatform
    ) -> Optional[InvoiceData]:
        """Send raw OCR text to local LLM for structured extraction."""
        try:
            trimmed = raw_text[:3000]
            prompt = self.PROMPT_TEMPLATE.format(text=trimmed)

            logger.info(f"Calling Ollama ({settings.ollama_model}) for LLM extraction...")

            response = requests.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": settings.llm_temperature,
                        "num_predict": settings.llm_max_tokens,
                    },
                },
                timeout=60,
            )

            if response.status_code != 200:
                logger.warning(f"Ollama returned HTTP {response.status_code}")
                return None

            result_text = response.json().get("response", "")
            logger.info(f"LLM response: {len(result_text)} chars")

            return self._parse_response(result_text, platform)

        except requests.ConnectionError:
            logger.warning(
                f"Cannot connect to Ollama at {settings.ollama_base_url}. "
                "Run: ollama serve"
            )
            return None
        except requests.Timeout:
            logger.warning("Ollama timed out after 60s")
            return None
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return None

    def _parse_response(
        self, text: str, platform: DocumentPlatform
    ) -> Optional[InvoiceData]:
        """Parse JSON from LLM response into InvoiceData."""
        try:
            text = text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

            json_match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return None

            d = json.loads(json_match.group())

            return InvoiceData(
                platform=platform,
                # Mentor's 22 fields
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
                seller_gst=d.get("seller_gst"),
                fssai_license=d.get("fssai_license"),
                billing_state_code=d.get("billing_state_code"),
                shipping_state_code=d.get("shipping_state_code"),
                place_of_supply=d.get("place_of_supply"),
                place_of_delivery=d.get("place_of_delivery"),
                reverse_charge=d.get("reverse_charge"),
                amount_in_words=d.get("amount_in_words"),
                seller_name=d.get("seller_name"),
                seller_address=d.get("seller_address"),
                total_tax=self._sf(d.get("total_tax")),
                total_amount=self._sf(d.get("total_amount")),
                # Bonus fields
                buyer_name=d.get("buyer_name"),
                subtotal=self._sf(d.get("subtotal")),
                cgst_amount=self._sf(d.get("cgst_amount")),
                sgst_amount=self._sf(d.get("sgst_amount")),
                igst_amount=self._sf(d.get("igst_amount")),
                discount=self._sf(d.get("discount")),
                delivery_charge=self._sf(d.get("delivery_charge")),
                payment_method=d.get("payment_method"),
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse LLM output: {e}")
            return None

    def _sf(self, value) -> Optional[float]:
        """Safe float conversion."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
