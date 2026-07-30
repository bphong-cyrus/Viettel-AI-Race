# -*- coding: utf-8 -*-
"""
Score submission on local_gt using scorer.py logic.
"""
import sys, json
from pathlib import Path
from typing import List, Dict
from scorer import jaccard_for_field, word_error_rate

def score_file(gt_path: Path, pred_path: Path) -> Dict:
    """Score a single file pair."""
    gt = json.load(gt_path.open(encoding='utf-8')) if gt_path.exists() else []
    pred = json.load(pred_path.open(encoding='utf-8')) if pred_path.exists() else []
    # Group by type
    def by_type(ents):
        d = {}
        for e in ents:
            d.setdefault(e['type'], []).append(e)
        return d

    gt_bt = by_type(gt)
    pr_bt = by_type(pred)

    text_total = 0.0
    text_count = 0
    assertions_total = 0.0
    assertions_count = 0
    candidates_total = 0.0
    candidates_count = 0

    # Score per type
    all_types = set(gt_bt.keys()) | set(pr_bt.keys())
    for t in all_types:
        gt_e = gt_bt.get(t, [])
        pr_e = pr_bt.get(t, [])
        if not gt_e and not pr_e:
            continue
        # Text: simple WER pairwise
        if gt_e or pr_e:
            # match each pred to closest GT
            used_gt = set()
            t_total = 0.0
            n_match = 0
            for p in pr_e:
                best_wer = 1.0
                best_idx = -1
                for j, g in enumerate(gt_e):
                    if j in used_gt:
                        continue
                    w = word_error_rate(g['text'].lower(), p['text'].lower())
                    if w < best_wer:
                        best_wer = w
                        best_idx = j
                if best_idx >= 0 and best_wer < 0.5:
                    used_gt.add(best_idx)
                    t_total += best_wer
                    n_match += 1
                else:
                    t_total += 1.0  # penalty for unmatched pred
            # Also penalize unmatched GT
            for j, g in enumerate(gt_e):
                if j not in used_gt:
                    t_total += 1.0
                    n_match += 1
            text_total += t_total
            text_count += max(len(gt_e), len(pr_e))

        # Assertions: jaccard on flat sets
        gt_a = [tuple(sorted(a)) for e in gt_e for a in (e.get('assertions') or [])]
        pr_a = [tuple(sorted(a)) for e in pr_e for a in (e.get('assertions') or [])]
        if gt_a or pr_a:
            j = jaccard_for_field(gt_a, pr_a)
            assertions_total += j
            assertions_count += 1

        # Candidates: jaccard on flat lists weighted
        gt_c = []
        for e in gt_e:
            for c in (e.get('candidates') or []):
                gt_c.append(c)
        pr_c = []
        for e in pr_e:
            for c in (e.get('candidates') or []):
                pr_c.append(c)
        if gt_c or pr_c:
            j = jaccard_for_field(gt_c, pr_c)
            candidates_total += j
            candidates_count += 1

    return {
        'text': text_total / text_count if text_count else 0,
        'assertions': assertions_total / assertions_count if assertions_count else 0,
        'candidates': candidates_total / candidates_count if candidates_count else 0,
        'n': text_count,
    }


def score_dir(gt_dir: Path, pred_dir: Path, max_files: int = 100) -> Dict:
    """Score directory pairs and average."""
    text_sum, ass_sum, cand_sum = 0.0, 0.0, 0.0
    wt_sum, wa_sum, wc_sum = 0.0, 0.0, 0.0
    n = 0
    for i in range(1, max_files + 1):
        gt_p = gt_dir / f'{i}.json'
        pr_p = pred_dir / f'{i}.json'
        if not gt_p.exists() or not pr_p.exists():
            continue
        r = score_file(gt_p, pr_p)
        text_sum += r['text']
        ass_sum += r['assertions']
        cand_sum += r['candidates']
        wt_sum += r['n'] if r['n'] else 0
        n += 1

    text_score = (text_sum / n) if n else 0
    ass_score = (ass_sum / n) if n else 0
    cand_score = (cand_sum / n) if n else 0
    final = 0.3 * text_score + 0.3 * ass_score + 0.4 * cand_score
    return {
        'text': text_score * 100,
        'assertions': ass_score * 100,
        'candidates': cand_score * 100,
        'final': final * 100,
    }


if __name__ == '__main__':
    gt_dir = Path('local_gt')
    results = {}
    for name in ['output_v11r3_max_controlled_recall', 'output_v2', 'output_v3']:
        d = Path(name)
        if not d.exists():
            continue
        r = score_dir(gt_dir, d, 100)
        results[name] = r
        print(f'{name}: text={r["text"]:.2f} assertions={r["assertions"]:.2f} candidates={r["candidates"]:.2f} FINAL={r["final"]:.2f}')
    print()
    print('Final ranking:')
    for name in sorted(results, key=lambda x: -results[x]['final']):
        r = results[name]
        print(f'  {name}: {r["final"]:.2f}')
