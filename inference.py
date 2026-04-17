#!/usr/bin/env python3
"""
BIM.AI — FastAPI Inference Server (Звено 3)
============================================
Запускает обученную модель как REST API.
Принимает текстовое описание MEP-элемента, возвращает JSON-спецификацию.

Запуск:
    python inference.py --adapter ./checkpoints/bim_ai_v1/lora_adapter
    python inference.py --adapter ./checkpoints/bim_ai_v1/lora_adapter --port 8080

API:
    POST /generate   {"prompt": "Диффузор Ø200мм, 150 л/с"}
                     → {"family": {...}, "valid": true, "warnings": [...]}

    POST /validate   {"family": {...}}
                     → {"valid": true, "errors": [...], "warnings": [...]}

    GET  /health     → {"status": "ok", "model": "..."}
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

# ── FastAPI & validation imports ──────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("❌ FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

MAX_NEW_TOKENS = 1800
MAX_RETRIES = 3          # If JSON invalid, retry with correction prompt
TEMPERATURE = 0.15       # Low temp for structured output

SYSTEM_PROMPT = (
    "You are BIM.AI, an expert assistant for generating Revit MEP family specifications. "
    "Given a description of an MEP component, output a complete JSON specification "
    "following the MEPBIMFamily schema. Include connectors with proper flow directions, "
    "system types, and dimensions. Include calculation parameters with formulas where applicable. "
    "Output only valid JSON, no explanations, no markdown."
)

CORRECTION_PROMPT_SUFFIX = (
    "\n\nYour previous response contained invalid JSON. "
    "Output ONLY the corrected JSON object, nothing else."
)


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

model = None
tokenizer = None
model_name_loaded = ""


def load_model(adapter_path: str):
    """Load fine-tuned model with LoRA adapter."""
    global model, tokenizer, model_name_loaded

    print(f"📥 Loading model from: {adapter_path}")

    try:
        from unsloth import FastLanguageModel
        adapter_path = Path(adapter_path)

        # Try to find the base model name from training config
        config_path = adapter_path.parent / "training_config.json"
        base_model = "codellama/CodeLlama-7b-Instruct-hf"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            base_model = cfg.get("model", base_model)
            print(f"   Base model: {base_model}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        model_name_loaded = str(adapter_path)
        print(f"✅ Model loaded successfully")

    except ImportError:
        # Fallback: use transformers directly (without unsloth acceleration)
        print("⚠️  Unsloth not available, using transformers (slower)")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base = AutoModelForCausalLM.from_pretrained(
            adapter_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = base
        model_name_loaded = str(adapter_path)
        print(f"✅ Model loaded (transformers fallback)")


# ─────────────────────────────────────────────
# JSON Extraction & Correction
# ─────────────────────────────────────────────

def extract_json(text: str) -> Optional[dict]:
    """Extract JSON object from model output, handling common issues."""
    text = text.strip()

    # Remove markdown code blocks if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block
    start = text.find("{")
    if start == -1:
        return None

    # Walk to find matching closing brace
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    break

    # Last resort: try to fix common issues
    candidate = text[start:] if start >= 0 else text
    # Remove trailing commas before } or ]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────

def generate_family(prompt: str, retries: int = MAX_RETRIES) -> tuple[Optional[dict], list[str]]:
    """
    Generate MEP family JSON from text prompt.
    Returns (family_dict, list_of_warnings).
    Retries if output is not valid JSON.
    """
    import torch
    from mep_validator import MEPValidator

    validator = MEPValidator()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(retries):
        # Format prompt
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE if attempt == 0 else 0.3,
                do_sample=attempt > 0,   # deterministic on first try
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        elapsed = time.time() - start_time

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        family = extract_json(generated)

        if family is not None:
            # Validate MEP constraints
            report = validator.validate(family)
            warnings = [f"[{w.rule}] {w.message}" for w in report.warnings]
            return family, warnings

        # JSON parse failed — add correction to conversation
        print(f"⚠️  Attempt {attempt+1}: invalid JSON, retrying...")
        messages.append({"role": "assistant", "content": generated})
        messages.append({"role": "user", "content": CORRECTION_PROMPT_SUFFIX})

    return None, ["Failed to generate valid JSON after {retries} attempts"]


# ─────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────

app = FastAPI(
    title="BIM.AI MEP Generator",
    description="Generate Revit MEP family specifications from text descriptions",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    retries: int = MAX_RETRIES


class ValidateRequest(BaseModel):
    family: dict


class GenerateResponse(BaseModel):
    family: Optional[dict]
    valid: bool
    warnings: list[str]
    generation_time_ms: Optional[float] = None


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[dict]
    warnings: list[dict]


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "no_model",
        "model": model_name_loaded,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")

    start = time.time()
    family, warnings = generate_family(req.prompt, retries=req.retries)
    elapsed_ms = (time.time() - start) * 1000

    return GenerateResponse(
        family=family,
        valid=family is not None,
        warnings=warnings,
        generation_time_ms=round(elapsed_ms, 1),
    )


@app.post("/validate", response_model=ValidateResponse)
def validate(req: ValidateRequest):
    from mep_validator import MEPValidator
    validator = MEPValidator()
    report = validator.validate(req.family)

    return ValidateResponse(
        valid=report.is_valid,
        errors=[
            {"rule": e.rule, "message": e.message, "path": e.field_path}
            for e in report.errors
        ],
        warnings=[
            {"rule": w.rule, "message": w.message, "suggestion": w.suggestion}
            for w in report.warnings
        ],
    )


# ─────────────────────────────────────────────
# CLI Test Mode (no server, just test prompts)
# ─────────────────────────────────────────────

TEST_PROMPTS = [
    "Создай потолочный диффузор круглый Ø200мм, расход 150 л/с, Systemair",
    "Circulation pump 8 m³/h, head 12m, DN50 connections, Grundfos",
    "LED ceiling fixture 36W, 4200lm, 4000K, DALI dimming",
    "Pendant sprinkler K80, 68°C, coverage 12m², wet pipe system",
    "VAV box Ø250mm, max airflow 300 L/s, DDC control",
]


def test_mode(n_prompts: int = 3):
    """Run test prompts and print results."""
    from mep_validator import MEPValidator
    validator = MEPValidator()

    print("\n" + "="*60)
    print("BIM.AI — Test Mode")
    print("="*60)

    for i, prompt in enumerate(TEST_PROMPTS[:n_prompts]):
        print(f"\n[{i+1}] PROMPT: {prompt}")
        print("-" * 40)

        start = time.time()
        family, warnings = generate_family(prompt)
        elapsed = time.time() - start

        if family:
            report = validator.validate(family)
            print(f"✅ Generated in {elapsed:.1f}s")
            print(f"   Name: {family.get('family_name')}")
            print(f"   Category: {family.get('category')}")
            print(f"   Domain: {family.get('mep_domain')}")
            print(f"   Connectors: {len(family.get('connectors', []))}")
            print(f"   Types: {len(family.get('family_types', []))}")
            print(f"   MEP valid: {'✅' if report.is_valid else '❌'}")
            if warnings:
                print(f"   Warnings: {warnings[:2]}")
        else:
            print(f"❌ Failed to generate valid JSON after {elapsed:.1f}s")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="BIM.AI Inference Server")
    parser.add_argument("--adapter", required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--test", action="store_true",
                        help="Run test prompts instead of starting server")
    parser.add_argument("--test-n", type=int, default=3,
                        help="Number of test prompts to run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model(args.adapter)

    if args.test:
        test_mode(n_prompts=args.test_n)
    else:
        print(f"\n🚀 Starting server on http://{args.host}:{args.port}")
        print(f"   Docs: http://localhost:{args.port}/docs")
        uvicorn.run(app, host=args.host, port=args.port)
