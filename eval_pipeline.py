#!/usr/bin/env python3
"""
BIM.AI — Evaluation Pipeline (Звено 2→3)
==========================================
Автоматически оценивает качество обученной модели по 4 метрикам:

  1. JSON validity rate     — % ответов с валидным JSON
  2. MEP validity rate      — % ответов, прошедших mep_validator
  3. Schema completeness    — среднее % заполненных полей схемы
  4. Connector accuracy     — % правильных domain/system type пар

Запуск:
    python eval_pipeline.py --adapter ./checkpoints/bim_ai_v1/lora_adapter
    python eval_pipeline.py --adapter ./checkpoints/bim_ai_v1/lora_adapter --n 50
    python eval_pipeline.py --baseline   # оценить без модели (baseline = 0%)
"""

import argparse
import json
import sys
import time
import statistics
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# Eval Prompts (held-out, не в train датасете)
# ─────────────────────────────────────────────

EVAL_PROMPTS = [
    # HVAC — диффузоры
    {"prompt": "Потолочный круглый диффузор Ø160мм, расход 80 л/с",
     "expected_domain": "HVAC", "expected_category": "Air Terminals"},
    {"prompt": "Supply ceiling diffuser 315mm, airflow 300 L/s, Trox",
     "expected_domain": "HVAC", "expected_category": "Air Terminals"},
    {"prompt": "Приточная решётка настенная 400×200мм, расход 120 л/с",
     "expected_domain": "HVAC", "expected_category": "Air Terminals"},

    # HVAC — FCU
    {"prompt": "Фанкойл 4-трубный потолочный, холодопроизводительность 5кВт",
     "expected_domain": "HVAC", "expected_category": "Mechanical Equipment"},
    {"prompt": "4-pipe fan coil unit 7kW cooling, ceiling, Daikin",
     "expected_domain": "HVAC", "expected_category": "Mechanical Equipment"},

    # HVAC — VAV
    {"prompt": "VAV-бокс Ø200мм, максимальный расход 200 л/с, без догрева",
     "expected_domain": "HVAC", "expected_category": "Duct Accessories"},
    {"prompt": "Variable air volume box 250mm inlet, 250 L/s max, DDC control",
     "expected_domain": "HVAC", "expected_category": "Duct Accessories"},

    # HVAC — AHU
    {"prompt": "Приточно-вытяжная установка 5000 м³/ч, с роторным рекуператором 80%",
     "expected_domain": "HVAC", "expected_category": "Mechanical Equipment"},

    # Piping — насосы
    {"prompt": "Циркуляционный насос 5 м³/ч, напор 8 м, Grundfos",
     "expected_domain": "Piping", "expected_category": "Plumbing Equipment"},
    {"prompt": "Inline circulation pump 20 m³/h, 15m head, DN80",
     "expected_domain": "Piping", "expected_category": "Plumbing Equipment"},

    # Piping — сантехника
    {"prompt": "Умывальник настенный 500×400мм, подключения ХВС DN15, ГВС DN15",
     "expected_domain": "Piping", "expected_category": "Plumbing Fixtures"},
    {"prompt": "Wall-mounted washbasin 600mm wide, accessible design",
     "expected_domain": "Piping", "expected_category": "Plumbing Fixtures"},

    # Electrical — щиты
    {"prompt": "Распределительный щит 100А, 24 модуля, трёхфазный",
     "expected_domain": "Electrical", "expected_category": "Electrical Equipment"},
    {"prompt": "Distribution panel 160A, 36 ways, 400V, IP54",
     "expected_domain": "Electrical", "expected_category": "Electrical Equipment"},

    # Electrical — освещение
    {"prompt": "Светодиодный потолочный светильник 36Вт, 4200лм, 4000K",
     "expected_domain": "Electrical", "expected_category": "Lighting Fixtures"},
    {"prompt": "LED panel 45W, 5400lm, 3000K, DALI dimmable, recessed ceiling",
     "expected_domain": "Electrical", "expected_category": "Lighting Fixtures"},

    # Fire Protection — спринклеры
    {"prompt": "Спринклер пендентный K80, температура активации 68°C, DN15",
     "expected_domain": "FireProtection", "expected_category": "Sprinklers"},
    {"prompt": "Concealed pendant sprinkler K115, 79°C, quick response",
     "expected_domain": "FireProtection", "expected_category": "Sprinklers"},

    # Fire Protection — извещатели
    {"prompt": "Оптический дымовой извещатель, покрытие 60 м², адресный",
     "expected_domain": "FireProtection", "expected_category": "Fire Alarm Devices"},
    {"prompt": "Multi-sensor fire detector, 80m² coverage, 12V DC",
     "expected_domain": "FireProtection", "expected_category": "Fire Alarm Devices"},
]


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

REQUIRED_FIELDS = [
    "family_name", "category", "mep_domain", "subcategory",
    "connectors", "calculation_params", "family_types", "classification",
    "host_type", "template_file",
]

REQUIRED_CONNECTOR_FIELDS = ["id", "domain", "flow_direction", "system_type", "shape", "dimensions"]


def score_completeness(family: dict) -> float:
    """Score schema completeness: fraction of required fields present and non-empty."""
    score = 0
    for f in REQUIRED_FIELDS:
        val = family.get(f)
        if val is not None and val != "" and val != [] and val != {}:
            score += 1

    # Bonus: connectors have required sub-fields
    connectors = family.get("connectors", [])
    if connectors:
        conn_score = 0
        for conn in connectors:
            for cf in REQUIRED_CONNECTOR_FIELDS:
                if conn.get(cf):
                    conn_score += 1
        conn_ratio = conn_score / (len(connectors) * len(REQUIRED_CONNECTOR_FIELDS))
        score += conn_ratio  # up to 1 extra point

    return score / (len(REQUIRED_FIELDS) + 1)


def score_connector_accuracy(family: dict) -> float:
    """Score connector domain/system type compatibility."""
    from mep_validator import DOMAIN_SYSTEM_COMPATIBILITY

    connectors = family.get("connectors", [])
    if not connectors:
        return 0.0

    correct = 0
    for conn in connectors:
        domain = conn.get("domain", "")
        system = conn.get("system_type", "")
        allowed = DOMAIN_SYSTEM_COMPATIBILITY.get(domain, set())
        if system in allowed:
            correct += 1

    return correct / len(connectors)


def score_domain_match(family: dict, expected_domain: str) -> bool:
    return family.get("mep_domain", "") == expected_domain


def score_category_match(family: dict, expected_category: str) -> bool:
    return family.get("category", "") == expected_category


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────

class EvalResult:
    def __init__(self, prompt: str, expected_domain: str, expected_category: str):
        self.prompt = prompt
        self.expected_domain = expected_domain
        self.expected_category = expected_category
        self.generated_text: str = ""
        self.family: Optional[dict] = None
        self.json_valid: bool = False
        self.mep_valid: bool = False
        self.completeness: float = 0.0
        self.connector_accuracy: float = 0.0
        self.domain_match: bool = False
        self.category_match: bool = False
        self.generation_time_s: float = 0.0
        self.error: Optional[str] = None


def run_eval(model, tokenizer, prompts: list[dict], verbose: bool = False) -> list[EvalResult]:
    """Run evaluation on all prompts."""
    import torch
    import re
    from mep_validator import MEPValidator

    validator = MEPValidator()
    results = []

    SYSTEM_PROMPT = (
        "You are BIM.AI, an expert assistant for generating Revit MEP family specifications. "
        "Given a description of an MEP component, output a complete JSON specification "
        "following the MEPBIMFamily schema. Include connectors with proper flow directions, "
        "system types, and dimensions. Include calculation parameters with formulas where applicable. "
        "Output only valid JSON, no explanations, no markdown."
    )

    for i, item in enumerate(prompts):
        result = EvalResult(
            prompt=item["prompt"],
            expected_domain=item["expected_domain"],
            expected_category=item["expected_category"],
        )

        if verbose:
            print(f"\n[{i+1}/{len(prompts)}] {item['prompt'][:60]}...")

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1800,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            result.generation_time_s = time.time() - start

            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            result.generated_text = generated

            # Parse JSON
            text = generated.strip()
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*$", "", text).strip()

            try:
                family = json.loads(text)
            except json.JSONDecodeError:
                # Try to extract JSON block
                start_idx = text.find("{")
                family = None
                if start_idx >= 0:
                    try:
                        family = json.loads(text[start_idx:])
                    except Exception:
                        pass

            if family is not None:
                result.json_valid = True
                result.family = family

                # MEP validation
                report = validator.validate(family)
                result.mep_valid = report.is_valid

                # Scores
                result.completeness = score_completeness(family)
                result.connector_accuracy = score_connector_accuracy(family)
                result.domain_match = score_domain_match(family, item["expected_domain"])
                result.category_match = score_category_match(family, item["expected_category"])

                if verbose:
                    status = "✅" if result.mep_valid else "⚠️ "
                    print(f"   {status} JSON valid, MEP={'✅' if result.mep_valid else '❌'}, "
                          f"domain={'✅' if result.domain_match else '❌'}, "
                          f"complete={result.completeness:.0%}, "
                          f"connectors={result.connector_accuracy:.0%} "
                          f"({result.generation_time_s:.1f}s)")
            else:
                result.error = "Invalid JSON"
                if verbose:
                    print(f"   ❌ JSON parse failed")

        except Exception as e:
            result.error = str(e)
            if verbose:
                print(f"   ❌ Error: {e}")

        results.append(result)

    return results


def print_report(results: list[EvalResult]):
    """Print evaluation summary."""
    n = len(results)
    json_valid = sum(1 for r in results if r.json_valid)
    mep_valid = sum(1 for r in results if r.mep_valid)
    domain_match = sum(1 for r in results if r.domain_match)
    cat_match = sum(1 for r in results if r.category_match)

    completeness_scores = [r.completeness for r in results if r.json_valid]
    connector_scores = [r.connector_accuracy for r in results if r.json_valid]
    gen_times = [r.generation_time_s for r in results]

    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"Samples evaluated: {n}")
    print()
    print(f"  JSON validity:        {json_valid}/{n} = {json_valid/n:.0%}")
    print(f"  MEP validity:         {mep_valid}/{n} = {mep_valid/n:.0%}")
    print(f"  Domain match:         {domain_match}/{n} = {domain_match/n:.0%}")
    print(f"  Category match:       {cat_match}/{n} = {cat_match/n:.0%}")
    if completeness_scores:
        print(f"  Schema completeness:  {statistics.mean(completeness_scores):.0%} avg")
    if connector_scores:
        print(f"  Connector accuracy:   {statistics.mean(connector_scores):.0%} avg")
    if gen_times:
        print(f"  Generation time:      {statistics.mean(gen_times):.1f}s avg, {max(gen_times):.1f}s max")

    # Domain breakdown
    print("\nBy domain:")
    for domain in ["HVAC", "Piping", "Electrical", "FireProtection"]:
        domain_results = [r for r in results if r.expected_domain == domain]
        if domain_results:
            valid = sum(1 for r in domain_results if r.mep_valid)
            print(f"  {domain:20s}: {valid}/{len(domain_results)} MEP valid")

    # Failures
    failures = [r for r in results if not r.json_valid]
    if failures:
        print(f"\nFailed JSON parsing ({len(failures)}):")
        for r in failures[:5]:
            print(f"  - {r.prompt[:60]}")
            print(f"    Output: {r.generated_text[:100]}...")

    print("="*60)

    # Return summary dict
    return {
        "n": n,
        "json_validity_rate": json_valid / n,
        "mep_validity_rate": mep_valid / n,
        "domain_match_rate": domain_match / n,
        "category_match_rate": cat_match / n,
        "schema_completeness": statistics.mean(completeness_scores) if completeness_scores else 0,
        "connector_accuracy": statistics.mean(connector_scores) if connector_scores else 0,
        "avg_generation_time_s": statistics.mean(gen_times) if gen_times else 0,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BIM.AI Evaluation Pipeline")
    parser.add_argument("--adapter", help="Path to LoRA adapter (required unless --baseline)")
    parser.add_argument("--n", type=int, default=len(EVAL_PROMPTS),
                        help=f"Number of eval prompts (max {len(EVAL_PROMPTS)})")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline evaluation (random/empty outputs)")
    parser.add_argument("--verbose", action="store_true", help="Show per-sample results")
    parser.add_argument("--output", help="Save results JSON to file")
    args = parser.parse_args()

    if args.baseline:
        print("📊 Baseline evaluation (no model — expected ~0% on all metrics)")
        results = []
        for item in EVAL_PROMPTS[:args.n]:
            r = EvalResult(item["prompt"], item["expected_domain"], item["expected_category"])
            results.append(r)
        summary = print_report(results)
        return

    if not args.adapter:
        print("❌ --adapter required (or use --baseline)")
        sys.exit(1)

    # Load model
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        print("❌ Unsloth not installed. Run: pip install unsloth")
        sys.exit(1)

    print(f"✅ Model loaded: {args.adapter}")
    print(f"📊 Running eval on {args.n} prompts...\n")

    prompts = EVAL_PROMPTS[:args.n]
    results = run_eval(model, tokenizer, prompts, verbose=args.verbose)
    summary = print_report(results)

    if args.output:
        out = {
            "adapter": args.adapter,
            "summary": summary,
            "results": [
                {
                    "prompt": r.prompt,
                    "expected_domain": r.expected_domain,
                    "expected_category": r.expected_category,
                    "json_valid": r.json_valid,
                    "mep_valid": r.mep_valid,
                    "domain_match": r.domain_match,
                    "category_match": r.category_match,
                    "completeness": round(r.completeness, 3),
                    "connector_accuracy": round(r.connector_accuracy, 3),
                    "generation_time_s": round(r.generation_time_s, 2),
                    "error": r.error,
                }
                for r in results
            ]
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
