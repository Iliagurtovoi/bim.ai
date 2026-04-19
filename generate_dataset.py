#!/usr/bin/env python3
"""
"""

import argparse
import json
from pathlib import Path

from mep_generator import MEPFamilyGenerator, ALL_TEMPLATES
from mep_prompts import MEPPromptBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=40,
                        help="families per template (40 → ~3120 pairs)")
    parser.add_argument("--prompts-per-family", type=int, default=3,
                        help="prompt variants per family")
    parser.add_argument("--output-dir", type=str, default="datasets")
    parser.add_argument("--lang", type=str, default="ru", choices=["ru", "en"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"→ Generating families: {args.scale} per template × {len(ALL_TEMPLATES)} templates")
    gen = MEPFamilyGenerator(seed=args.seed)
    families = gen.generate_dataset(n_per_template=args.scale)
    print(f"  produced {len(families)} families")

    print(f"→ Building prompt pairs (lang={args.lang})")
    builder = MEPPromptBuilder(lang=args.lang, seed=args.seed)
    dataset = builder.build_dataset(families, prompts_per_family=args.prompts_per_family)
    print(f"  produced {len(dataset)} (prompt, completion) pairs")

    # Export ChatML
    chatml = builder.export_for_chat(dataset)
    chatml_path = out_dir / "mep_chatml.jsonl"
    with chatml_path.open("w", encoding="utf-8") as f:
        for item in chatml:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ Wrote {chatml_path} ({chatml_path.stat().st_size // 1024} KB)")

    # Export Alpaca
    alpaca = builder.export_for_unsloth(dataset)
    alpaca_path = out_dir / "mep_alpaca.jsonl"
    with alpaca_path.open("w", encoding="utf-8") as f:
        for item in alpaca:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ Wrote {alpaca_path} ({alpaca_path.stat().st_size // 1024} KB)")

    # Summary
    domains = {}
    for f in families:
        domains[f.get("mep_domain", "?")] = domains.get(f.get("mep_domain", "?"), 0) + 1
    print("\nBreakdown by domain:")
    for d, n in sorted(domains.items()):
        print(f"  {d}: {n} families")


if __name__ == "__main__":
    main()
