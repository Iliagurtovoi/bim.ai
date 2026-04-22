#!/usr/bin/env python3
"""
BIM.AI — QLoRA Fine-tuning Script
====================================
Дообучает CodeLlama-7B (или Mistral-7B) на датасете MEP-семейств.
Использует Unsloth для 4-bit QLoRA — работает на одной RTX 4090 (24GB).

Запуск на Vast.ai:
    pip install unsloth
    python train_qlora.py --data datasets/mep_chatml.jsonl

Запуск с кастомными параметрами:
    python train_qlora.py \
        --model codellama/CodeLlama-7b-Instruct-hf \
        --data datasets/mep_chatml.jsonl \
        --epochs 5 \
        --batch-size 4 \
        --lr 2e-4 \
        --lora-r 16 \
        --output ./checkpoints/bim_ai_v1
"""

import argparse
import json
import os
import sys
import math
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# Constants & Defaults
# ─────────────────────────────────────────────

DEFAULT_MODEL = "codellama/CodeLlama-7b-Instruct-hf"
FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

MAX_SEQ_LENGTH = 2048        # достаточно для max ~1200 токенов в наших парах
DTYPE = None                  # auto-detect: float16 on Ampere, bfloat16 on Hopper

SYSTEM_PROMPT = (
    "You are BIM.AI, an expert assistant for generating Revit MEP family specifications. "
    "Given a description of an MEP component, output a complete JSON specification "
    "following the MEPBIMFamily schema. Include connectors with proper flow directions, "
    "system types, and dimensions. Include calculation parameters with formulas where applicable. "
    "Output only valid JSON, no explanations."
)


# ─────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    """Load ChatML JSONL or Alpaca JSON dataset."""
    path = Path(path)
    if not path.exists():
        print(f"❌ Dataset not found: {path}")
        sys.exit(1)

    data = []
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(path) as f:
            raw = json.load(f)
        # Convert Alpaca → ChatML
        for item in raw:
            instruction = item.get("instruction", "")
            # Strip [SYSTEM] prefix if present (we add it via model's system prompt)
            if "[USER]" in instruction:
                user_part = instruction.split("[USER]", 1)[1].strip()
            else:
                user_part = instruction
            data.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_part},
                    {"role": "assistant", "content": item.get("output", "")},
                ]
            })

    print(f"✅ Loaded {len(data)} samples from {path.name}")
    return data


def split_dataset(data: list[dict], val_ratio: float = 0.05):
    """Split into train/val sets."""
    import random
    random.seed(42)
    shuffled = data.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


# ─────────────────────────────────────────────
# Formatting for Unsloth
# ─────────────────────────────────────────────

def format_prompt(sample: dict, tokenizer) -> str:
    """
    Format a ChatML sample into the model's chat template.
    Unsloth's apply_chat_template handles the actual formatting.
    """
    messages = sample.get("messages", [])
    # Ensure system message is present
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ─────────────────────────────────────────────
# Validation Callback
# ─────────────────────────────────────────────

def run_validation_sample(model, tokenizer, val_data: list[dict], n_samples: int = 3) -> dict:
    """
    Run inference on a few validation samples, check JSON validity.
    Returns dict with metrics.
    """
    import json as json_mod
    import torch
    from mep_validator import MEPValidator

    validator = MEPValidator()
    model.eval()

    results = {"json_valid": 0, "mep_valid": 0, "total": 0}
    samples = val_data[:n_samples]

    for sample in samples:
        messages = sample["messages"]
        # Use only system + user for inference
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1500,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        results["total"] += 1

        # Check JSON validity
        try:
            parsed = json_mod.loads(generated.strip())
            results["json_valid"] += 1

            # Check MEP validity
            report = validator.validate(parsed)
            if report.is_valid:
                results["mep_valid"] += 1
        except (json_mod.JSONDecodeError, Exception):
            pass

    model.train()
    return results


# ─────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────

def train(args):
    print("\n" + "="*60)
    print("BIM.AI — QLoRA Fine-tuning")
    print("="*60)
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {args.data}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch_size} (grad_accum={args.grad_accum})")
    print(f"  LR:       {args.lr}")
    print(f"  LoRA r:   {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"  Output:   {args.output}")
    print("="*60 + "\n")

    # ── Import Unsloth ────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
        DTYPE_USE = "bfloat16" if is_bfloat16_supported() else "float16"
        print(f"✅ Unsloth loaded, dtype={DTYPE_USE}")
    except ImportError:
        print("❌ Unsloth not installed. Run: pip install unsloth")
        print("   For Vast.ai: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
        sys.exit(1)

    import torch
    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from datasets import Dataset

    # ── Load model ───────────────────────────────────────────────
    print(f"📥 Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,          # auto
        load_in_4bit=True,   # QLoRA 4-bit
    )

    # ── Apply LoRA ───────────────────────────────────────────────
    print(f"🔧 Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Load & prepare dataset ────────────────────────────────────
    raw_data = load_dataset(args.data)
    train_data, val_data = split_dataset(raw_data, val_ratio=0.05)
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}")

    def format_sample(sample):
        return {"text": format_prompt(sample, tokenizer)}

    train_dataset = Dataset.from_list(train_data).map(format_sample)
    val_dataset = Dataset.from_list(val_data).map(format_sample)

    # ── Training arguments ────────────────────────────────────────
    steps_per_epoch = math.ceil(len(train_data) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(100, total_steps // 10)
    eval_steps = max(50, steps_per_epoch // 2)
    save_steps = steps_per_epoch

    print(f"\n📊 Training plan:")
    print(f"   Steps/epoch: {steps_per_epoch}, Total: {total_steps}")
    print(f"   Warmup: {warmup_steps}, Eval every: {eval_steps} steps")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=warmup_steps,
        learning_rate=args.lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",               # set to "wandb" if you want tracking
        dataloader_num_workers=0,
    )

    # ── Trainer ──────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=True,                   # pack short sequences for efficiency
        args=training_args,
    )

    # ── Pre-training validation ───────────────────────────────────
    if args.validate:
        print("\n🔍 Pre-training validation (baseline)...")
        FastLanguageModel.for_inference(model)
        pre_metrics = run_validation_sample(model, tokenizer, val_data, n_samples=3)
        print(f"   JSON valid: {pre_metrics['json_valid']}/{pre_metrics['total']}")
        print(f"   MEP valid:  {pre_metrics['mep_valid']}/{pre_metrics['total']}")
        FastLanguageModel.for_training(model)

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n🚀 Starting training at {datetime.now().strftime('%H:%M:%S')}...")
    trainer_stats = trainer.train()

    elapsed = trainer_stats.metrics.get("train_runtime", 0)
    print(f"\n✅ Training complete in {elapsed/60:.1f} min")
    print(f"   Final loss: {trainer_stats.metrics.get('train_loss', 'N/A'):.4f}")

    # ── Post-training validation ──────────────────────────────────
    if args.validate:
        print("\n🔍 Post-training validation...")
        FastLanguageModel.for_inference(model)
        post_metrics = run_validation_sample(model, tokenizer, val_data, n_samples=5)
        print(f"   JSON valid: {post_metrics['json_valid']}/{post_metrics['total']}")
        print(f"   MEP valid:  {post_metrics['mep_valid']}/{post_metrics['total']}")

    # ── Save model ────────────────────────────────────────────────
    print(f"\n💾 Saving LoRA adapter to {output_dir}/lora_adapter...")
    model.save_pretrained(str(output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(output_dir / "lora_adapter"))

    if args.save_merged:
        print(f"💾 Saving merged 16-bit model to {output_dir}/merged_16bit...")
        model.save_pretrained_merged(
            str(output_dir / "merged_16bit"),
            tokenizer,
            save_method="merged_16bit",
        )

    if args.save_gguf:
        print(f"💾 Saving GGUF (Q4_K_M) to {output_dir}/gguf/...")
        model.save_pretrained_gguf(
            str(output_dir / "gguf"),
            tokenizer,
            quantization_method="q4_k_m",
        )

    # ── Save training config ──────────────────────────────────────
    config = {
        "model": args.model,
        "dataset": args.data,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_runtime_min": round(elapsed / 60, 1),
        "final_loss": trainer_stats.metrics.get("train_loss"),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ All done! Adapter saved to: {output_dir}/lora_adapter")
    print(f"   To run inference: python inference.py --adapter {output_dir}/lora_adapter")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="BIM.AI QLoRA Training")

    # Model
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Base model (default: {DEFAULT_MODEL})")

    # Dataset
    parser.add_argument("--data", default="datasets/mep_chatml.jsonl",
                        help="Path to ChatML JSONL or Alpaca JSON dataset")

    # Training
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of training epochs (default: 4)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device batch size (default: 2)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4, effective batch=8)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank (default: 16; use 32 for better quality)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (default: 32; keep 2x lora-r)")

    # Output
    parser.add_argument("--output", default="./checkpoints/bim_ai_v1",
                        help="Output directory for checkpoints and adapter")
    parser.add_argument("--save-merged", action="store_true",
                        help="Also save merged 16-bit model (needs ~14GB disk)")
    parser.add_argument("--save-gguf", action="store_true",
                        help="Also save GGUF Q4_K_M for llama.cpp inference")

    # Validation
    parser.add_argument("--validate", action="store_true",
                        help="Run MEP JSON validation before and after training")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
