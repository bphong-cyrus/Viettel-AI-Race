# -*- coding: utf-8 -*-
"""
Evaluation harness: scores predicted JSONs against ground truth JSONs.

Usage:
    python evaluate.py --gt_dir bootstrap_gt --pred_dir output_v20
    python evaluate.py --gt_dir bootstrap_gt --pred_dir output_hybrid

Reports per-sample and aggregate scores (text WER, assertions Jaccard, candidates Jaccard).
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from scorer import NERScorer, word_error_rate, jaccard_for_field


def load_json(path: Path) -> list:
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []


def evaluate(gt_dir: str, pred_dir: str, max_samples: int = None,
             per_sample: bool = False, worst: int = 5):
    gt_path = Path(gt_dir)
    pr_path = Path(pred_dir)

    scorer = NERScorer()

    gt_dict = {}
    pr_dict = {}
    for gt_file in sorted(gt_path.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 999):
        sid = int(gt_file.stem)
        gt_dict[sid] = load_json(gt_file)
        pr_file = pr_path / f"{sid}.json"
        if pr_file.exists():
            pr_dict[sid] = load_json(pr_file)
        else:
            pr_dict[sid] = []

    if max_samples:
        gt_dict = dict(list(gt_dict.items())[:max_samples])

    sample_scores = []
    total_text = 0.0
    total_assert = 0.0
    total_cand = 0.0

    for sid in sorted(gt_dict.keys()):
        gt = gt_dict[sid]
        pred = pr_dict.get(sid, [])
        scores = scorer.score_sample(gt, pred)
        scores["sample_id"] = sid
        sample_scores.append(scores)
        total_text += scores["text_score"]
        total_assert += scores["assertions_score"]
        total_cand += scores["candidates_score"]

    n = len(sample_scores)
    if n == 0:
        print("No samples")
        return

    avg_text = total_text / n
    avg_assert = total_assert / n
    avg_cand = total_cand / n
    final = 0.3 * avg_text + 0.3 * avg_assert + 0.4 * avg_cand

    print("=" * 70)
    print(f"EVALUATION  GT={gt_dir}  PRED={pred_dir}")
    print(f"Samples: {n}")
    print("=" * 70)
    print(f"  text_score      : {avg_text:.4f}  (1 - WER)")
    print(f"  assertions_score: {avg_assert:.4f}  (Jaccard)")
    print(f"  candidates_score: {avg_cand:.4f}  (weighted Jaccard)")
    print(f"  FINAL SCORE     : {final:.4f}")
    print("=" * 70)

    if per_sample:
        print("\nPer-sample scores:")
        for s in sample_scores:
            print(f"  #{s['sample_id']:3d} text={s['text_score']:.3f} "
                  f"assert={s['assertions_score']:.3f} cand={s['candidates_score']:.3f} "
                  f"final={s['final_score']:.3f}")

    if worst > 0:
        worst_list = sorted(sample_scores, key=lambda s: s["final_score"])[:worst]
        print(f"\nWorst {worst} samples:")
        for s in worst_list:
            print(f"  #{s['sample_id']:3d} final={s['final_score']:.3f} "
                  f"(text={s['text_score']:.3f} assert={s['assertions_score']:.3f} cand={s['candidates_score']:.3f})")

    return {
        "text_score": avg_text,
        "assertions_score": avg_assert,
        "candidates_score": avg_cand,
        "final_score": final,
        "n_samples": n,
    }


def compare_runs(runs: list, gt_dir: str):
    """Compare multiple prediction dirs side by side."""
    print("=" * 90)
    print(f"COMPARISON  GT={gt_dir}")
    print("=" * 90)
    print(f"{'Run':<30} {'Text':>8} {'Assert':>8} {'Cand':>8} {'Final':>8}")
    print("-" * 90)
    for name, pred_dir in runs:
        gt_path = Path(gt_dir)
        pr_path = Path(pred_dir)
        scorer = NERScorer()

        gt_dict = {int(p.stem): load_json(p) for p in sorted(gt_path.glob("*.json"),
                  key=lambda x: int(x.stem) if x.stem.isdigit() else 999)}

        tt, ta, tc = 0, 0, 0
        for sid, gt in gt_dict.items():
            pr_file = pr_path / f"{sid}.json"
            pred = load_json(pr_file) if pr_file.exists() else []
            s = scorer.score_sample(gt, pred)
            tt += s["text_score"]
            ta += s["assertions_score"]
            tc += s["candidates_score"]

        n = len(gt_dict)
        avg_t, avg_a, avg_c = tt/n, ta/n, tc/n
        final = 0.3*avg_t + 0.3*avg_a + 0.4*avg_c
        print(f"{name:<30} {avg_t:>8.4f} {avg_a:>8.4f} {avg_c:>8.4f} {final:>8.4f}")
    print("=" * 90)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", default="bootstrap_gt")
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--per_sample", action="store_true")
    ap.add_argument("--worst", type=int, default=5)
    args = ap.parse_args()

    evaluate(args.gt_dir, args.pred_dir, args.max_samples,
             args.per_sample, args.worst)