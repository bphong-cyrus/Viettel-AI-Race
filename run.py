# -*- coding: utf-8 -*-
"""
Unified runner: orchestrate rule-based, LLM, and hybrid pipelines.

Examples:
    # Rules only (baseline, fast)
    python run.py rules

    # LLM only (Qwen2.5-7B)
    python run.py llm --model qwen2.5-7b-instruct --vram 16

    # Hybrid (rules + LLM)
    python run.py hybrid --model qwen2.5-7b-instruct --vram 16

    # LoRA-fine-tuned model
    python run.py llm --model qwen2.5-7b-instruct --vram 16 \\
        --lora_path ./lora_adapter

    # Evaluate on bootstrap_gt
    python run.py eval --pred_dir output_v20
    python run.py eval --pred_dir output_hybrid

    # Compare runs
    python run.py compare
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def cmd_rules(args):
    """Run v20 rule-based pipeline."""
    import v20_pipeline
    print("Running v20 rule-based pipeline...")
    v20_pipeline.run_batch(args.input_dir, args.output_dir, args.zip_path)
    print(f"Done. Output: {args.output_dir}, ZIP: {args.zip_path}")


def cmd_llm(args):
    """Run LLM-only pipeline (or hybrid with rules if --with_rules)."""
    from llm_inference import LLMExtractor

    print(f"Running LLM pipeline with model={args.model}, vram={args.vram}GB")

    extractor = LLMExtractor(
        model_key=args.model,
        vram_gb=args.vram,
        use_few_shot=not args.no_few_shot,
    )

    # If LoRA adapter provided, load on top
    if args.lora_path and os.path.exists(args.lora_path):
        import torch
        from peft import PeftModel
        print(f"Loading LoRA adapter from {args.lora_path}")
        extractor.load()
        extractor.model = PeftModel.from_pretrained(extractor.model, args.lora_path)
        extractor.model.eval()
        print("LoRA adapter loaded.")

    if args.with_rules:
        from hybrid_pipeline import HybridNER
        runner = HybridNER(llm_extractor=extractor, use_llm=True).extract
    else:
        runner = extractor.extract

    os.makedirs(args.output_dir, exist_ok=True)
    input_files = sorted(Path(args.input_dir).glob("*.txt"),
                         key=lambda p: int(p.stem) if p.stem.isdigit() else 999)
    total_entities = 0
    type_counts = {"THUỐC": 0, "TRIỆU_CHỨNG": 0, "CHẨN_ĐOÁN": 0}
    t0 = time.time()
    for i, in_file in enumerate(input_files, 1):
        text = in_file.read_text(encoding="utf-8")
        ents = runner(text)
        with open(f"{args.output_dir}/{in_file.stem}.json", "w", encoding="utf-8") as f:
            json.dump(ents, f, ensure_ascii=False, indent=2)
        total_entities += len(ents)
        for e in ents:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
        if i % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(input_files) - i)
            print(f"  [{i}/{len(input_files)}] {elapsed:.1f}s, ETA {eta:.1f}s")

    print(f"\nTotal: {total_entities}, types: {type_counts}")
    print(f"Time: {time.time()-t0:.1f}s")


def cmd_eval(args):
    """Evaluate predictions against ground truth."""
    from evaluate import evaluate
    evaluate(args.gt_dir, args.pred_dir, args.max_samples, args.per_sample, args.worst)


def cmd_compare(args):
    """Compare multiple prediction directories."""
    from evaluate import compare_runs
    runs = []
    for p in args.runs:
        name = Path(p).name
        runs.append((name, p))
    compare_runs(runs, args.gt_dir)


def cmd_smoke(args):
    """Quick sanity check: load model, extract from 1 sample, print result."""
    from llm_inference import LLMExtractor
    extractor = LLMExtractor(args.model, vram_gb=args.vram)
    extractor.load()
    text = Path("input/input/1.txt").read_text(encoding="utf-8")[:1500]
    ents = extractor.extract(text)
    print(f"\n=== Extracted {len(ents)} entities ===")
    for e in ents:
        print(f"  [{e['type']}] '{e['text']}' assert={e['assertions']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # rules
    p_rules = sub.add_parser("rules", help="Run v20 rule-based pipeline")
    p_rules.add_argument("--input_dir", default="input/input")
    p_rules.add_argument("--output_dir", default="output_v20")
    p_rules.add_argument("--zip_path", default="output_v20.zip")

    # llm
    p_llm = sub.add_parser("llm", help="Run LLM-based pipeline")
    p_llm.add_argument("--model", default=None)
    p_llm.add_argument("--vram", type=float, default=24.0)
    p_llm.add_argument("--input_dir", default="input/input")
    p_llm.add_argument("--output_dir", default="output_llm")
    p_llm.add_argument("--with_rules", action="store_true",
                       help="Combine with rule-based (hybrid mode)")
    p_llm.add_argument("--lora_path", default=None, help="Path to LoRA adapter")
    p_llm.add_argument("--no_few_shot", action="store_true")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate predictions against GT")
    p_eval.add_argument("--gt_dir", default="bootstrap_gt")
    p_eval.add_argument("--pred_dir", required=True)
    p_eval.add_argument("--max_samples", type=int, default=None)
    p_eval.add_argument("--per_sample", action="store_true")
    p_eval.add_argument("--worst", type=int, default=5)

    # compare
    p_cmp = sub.add_parser("compare", help="Compare multiple pred dirs")
    p_cmp.add_argument("--gt_dir", default="bootstrap_gt")
    p_cmp.add_argument("--runs", nargs="+", required=True,
                       help="List of pred dirs (name derived from basename)")

    # smoke
    p_smoke = sub.add_parser("smoke", help="Quick smoke test")
    p_smoke.add_argument("--model", default=None)
    p_smoke.add_argument("--vram", type=float, default=24.0)

    args = ap.parse_args()
    {
        "rules": cmd_rules,
        "llm": cmd_llm,
        "eval": cmd_eval,
        "compare": cmd_compare,
        "smoke": cmd_smoke,
    }[args.cmd](args)


if __name__ == "__main__":
    main()