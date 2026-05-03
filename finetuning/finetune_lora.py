"""
LoRA Fine-Tuning Script for Invoice Data Extraction.

Trains a small LLM (TinyLlama 1.1B or Phi-2 2.7B) with LoRA adapters
to extract structured JSON from Indian invoice OCR text.

Hardware: RTX 3050 (4GB VRAM)
Strategy: 4-bit quantization (QLoRA) + LoRA rank 16 + gradient checkpointing

Usage:
    python finetuning/finetune_lora.py

    # With custom options:
    python finetuning/finetune_lora.py \
        --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --data_path finetuning/data/invoice_extraction_train.jsonl \
        --output_dir finetuning/models/invoice-lora \
        --epochs 3 \
        --batch_size 1 \
        --learning_rate 2e-4

Output: LoRA adapter weights saved to finetuning/models/invoice-lora/
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_dataset(data_path: str) -> list:
    """
    Load training data from JSONL file.
    Each line has: instruction, input (OCR text), output (JSON)
    """
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                samples.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num}: {e}")

    logger.info(f"Loaded {len(samples)} training samples from {data_path}")
    return samples


def format_prompt(sample: dict) -> str:
    """
    Format a single training sample into the chat template.
    Uses the Alpaca/Llama instruction format.
    """
    return (
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Input:\n{sample['input']}\n\n"
        f"### Response:\n{sample['output']}"
    )


def create_hf_dataset(samples: list, tokenizer):
    """Convert raw samples into a tokenized HuggingFace Dataset."""
    from datasets import Dataset

    formatted = []
    for s in samples:
        text = format_prompt(s) + tokenizer.eos_token
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)

    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(tokenize, remove_columns=["text"])
    logger.info(f"Tokenized {len(tokenized)} samples (max_length=1024)")
    return tokenized


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_model_and_tokenizer(base_model: str):
    """
    Load base model with 4-bit quantization (QLoRA) for RTX 3050 compatibility.

    Memory usage with 4-bit:
    - TinyLlama 1.1B: ~1.2 GB VRAM
    - Phi-2 2.7B: ~2.5 GB VRAM
    - Mistral 7B: Too large for 4GB, use Ollama instead
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    logger.info(f"Loading base model: {base_model}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()

    logger.info(
        f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params"
    )

    return model, tokenizer


def apply_lora(model):
    """
    Apply LoRA adapters to the model.

    LoRA config optimized for RTX 3050:
    - rank=16 (small but effective for structured extraction)
    - alpha=32 (2x rank is standard)
    - dropout=0.05 (light regularization)
    - Target: query + value projection layers
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                      # LoRA rank
        lora_alpha=32,             # Scaling factor
        lora_dropout=0.05,         # Dropout
        target_modules=[           # Which layers to adapt
            "q_proj", "v_proj",    # Attention projections
            "k_proj", "o_proj",    # Additional attention
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA applied: {trainable / 1e6:.2f}M trainable / "
        f"{total / 1e6:.0f}M total ({100 * trainable / total:.2f}%)"
    )

    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(
    model,
    tokenizer,
    dataset,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    warmup_steps: int = 10,
):
    """
    Train the LoRA model using HuggingFace Trainer.

    Training settings optimized for RTX 3050 4GB:
    - batch_size=1 (minimum to fit in VRAM)
    - gradient_accumulation=4 (effective batch size = 4)
    - fp16=True (half precision)
    - gradient_checkpointing=True (trade compute for memory)
    """
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,      # Effective batch = 4
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=True,                          # Half precision
        optim="paged_adamw_8bit",           # Memory-efficient optimizer
        lr_scheduler_type="cosine",
        report_to="none",                   # No wandb/tensorboard
        gradient_checkpointing=True,        # Critical for 4GB VRAM
        max_grad_norm=0.3,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    logger.info(f"Starting training: {epochs} epochs, lr={learning_rate}")
    trainer.train()

    # Save LoRA adapter weights only (small — ~10-50MB)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"LoRA adapter saved to {output_dir}")

    # Show saved size
    total_size = sum(
        f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
    )
    logger.info(f"Saved adapter size: {total_size / 1e6:.1f} MB")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for invoice extraction")
    parser.add_argument(
        "--base_model", type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model from HuggingFace (default: TinyLlama 1.1B)",
    )
    parser.add_argument(
        "--data_path", type=str,
        default="finetuning/data/invoice_extraction_train.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="finetuning/models/invoice-lora",
        help="Where to save LoRA adapter weights",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.data_path):
        logger.error(f"Training data not found: {args.data_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    samples = load_dataset(args.data_path)
    if len(samples) < 5:
        logger.warning(f"Only {len(samples)} samples — consider adding more for better results")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.base_model)

    # Apply LoRA
    model = apply_lora(model)

    # Tokenize dataset
    dataset = create_hf_dataset(samples, tokenizer)

    # Train
    train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    logger.info("Training complete!")
    logger.info(f"To use the fine-tuned model, set these in your .env:")
    logger.info(f"  USE_FINETUNED_MODEL=true")
    logger.info(f"  FINETUNED_MODEL_PATH={args.output_dir}")
    logger.info(f"  FINETUNED_BASE_MODEL={args.base_model}")


if __name__ == "__main__":
    main()
