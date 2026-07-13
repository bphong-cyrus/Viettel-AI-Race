# -*- coding: utf-8 -*-
"""
Hybrid pipeline: rules-first (v20) + LLM-fill-missing.

Strategy:
1. Run v20 rule-based pipeline (proven to extract 70%+ of THUỐC and many vocab terms)
2. Identify "missed" entities — entities the rules did NOT catch:
   - LLM extracts independently
   - We compare positions: keep rule-based where they overlap, ADD LLM-only
3. Merge with LLM's superior type/assertion judgement for ambiguous cases
4. Re-rank by confidence

This is more reliable than pure LLM (avoids hallucination on drugs) and more
complete than pure rules (catches complex symptoms/diagnoses).
"""
import os
import sys
import json
import zipfile
import time
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Import base pipeline
from v20_pipeline import extract_entities_v20


# ============================================================================
# POSITION OVERLAP — for merging rule + LLM extractions
# ============================================================================
def positions_overlap(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
    """True if two spans overlap at all (including containment)."""
    return not (p1[1] <= p2[0] or p2[1] <= p1[0])


def position_match_score(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """0..1 similarity: 1.0 = exact match, 0.5 = containment, 0 = no overlap."""
    s1, e1 = p1
    s2, e2 = p2
    if e1 <= s2 or e2 <= s1:
        return 0.0
    inter = min(e1, e2) - max(s1, s2)
    union = max(e1, e2) - min(s1, s2)
    return inter / max(union, 1)


# ============================================================================
# HYBRID MERGE
# ============================================================================
def merge_rule_and_llm(rule_ents: List[Dict], llm_ents: List[Dict],
                       rule_weight: float = 0.6,
                       match_threshold: float = 0.5) -> List[Dict]:
    """Merge two entity lists.

    - If rule + LLM extractions overlap at >= threshold, prefer the rule one
      (rules have more reliable drug CUI candidates)
    - If LLM extraction has no rule match, add it (rules miss complex symptoms)
    - Use LLM's assertion/type judgement for cases where both exist but disagree
      on type

    Returns: merged entity list, sorted by position.
    """
    if not rule_ents and not llm_ents:
        return []
    if not rule_ents:
        return sorted(llm_ents, key=lambda e: e["position"][0])
    if not llm_ents:
        return sorted(rule_ents, key=lambda e: e["position"][0])

    matched_llm = set()
    merged = []

    for r in rule_ents:
        rp = tuple(r["position"])
        best_match = None
        best_score = 0.0
        for i, l in enumerate(llm_ents):
            if i in matched_llm:
                continue
            score = position_match_score(rp, tuple(l["position"]))
            if score > best_score:
                best_score = score
                best_match = (i, l)

        if best_match and best_score >= match_threshold:
            i, l = best_match
            matched_llm.add(i)
            # Merge — prefer rule's position, but enrich with LLM
            merged_ent = dict(r)
            # If LLM has more/better assertions, take them
            if len(l.get("assertions", [])) > len(r.get("assertions", [])):
                merged_ent["assertions"] = l["assertions"]
            # If LLM has candidates and rule doesn't (for THUỐC), take LLM
            if r.get("type") == "THUỐC" and not r.get("candidates") and l.get("candidates"):
                merged_ent["candidates"] = l["candidates"]
            # If types disagree, prefer the more confident one
            # Default: trust rule for THUỐC, LLM for TRIỆU_CHỨNG/CHẨN_ĐOÁN
            if l.get("type") != r.get("type"):
                # Rule is generally more reliable for drugs
                if r.get("type") == "THUỐC":
                    pass  # keep rule
                else:
                    # LLM might know better for symptom/diagnosis
                    # but only if it has assertions or different text
                    if l.get("assertions") and not r.get("assertions"):
                        merged_ent["type"] = l["type"]
            merged.append(merged_ent)
        else:
            merged.append(r)

    # Add unmatched LLM entities
    for i, l in enumerate(llm_ents):
        if i not in matched_llm:
            merged.append(l)

    # Sort by position
    merged.sort(key=lambda e: (e["position"][0], e["position"][1]))

    # Final dedupe (same text + position)
    seen = set()
    final = []
    for e in merged:
        key = (e["text"].lower().strip(), e["position"][0], e["position"][1], e["type"])
        if key not in seen:
            seen.add(key)
            final.append(e)
    return final


# ============================================================================
# POST-PROCESS: trim LLM hallucinations
# ============================================================================
def clean_llm_entities(entities: List[Dict], original_text: str) -> List[Dict]:
    """Remove or fix likely LLM hallucinations."""
    cleaned = []
    for e in entities:
        text = e.get("text", "").strip()
        etype = e.get("type", "")

        # Skip empty or invalid types
        if not text or etype not in ("THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN"):
            continue

        # Skip too-short generic terms
        if len(text) < 2:
            continue

        # Skip "stop words" extracted as entities
        if text.lower() in {"bệnh", "triệu chứng", "thuốc", "bệnh nhân", "có", "không", "chưa"}:
            continue

        # Skip candidates for non-drug types
        if etype != "THUỐC" and e.get("candidates"):
            e["candidates"] = []

        # Verify position
        pos = e.get("position", [0, 0])
        if not (isinstance(pos, list) and len(pos) == 2):
            continue
        s, end = pos
        if s < 0 or end > len(original_text):
            continue
        actual = original_text[s:end]
        if actual.strip() != text:
            # Try to relocate
            idx = original_text.find(text)
            if idx >= 0:
                e["position"] = [idx, idx + len(text)]
            else:
                # Skip if we can't place it
                continue
        cleaned.append(e)
    return cleaned


# ============================================================================
# MAIN HYBRID PIPELINE
# ============================================================================
class HybridNER:
    """Combines rule-based v20 with LLM extraction."""

    def __init__(self, llm_extractor=None, use_llm: bool = True,
                 rule_weight: float = 0.6):
        self.llm = llm_extractor
        self.use_llm = use_llm and llm_extractor is not None
        self.rule_weight = rule_weight

    def extract(self, text: str) -> List[Dict]:
        # Step 1: rule-based
        rule_ents = extract_entities_v20(text)

        # Step 2: LLM (if available)
        if self.use_llm:
            try:
                llm_ents = self.llm.extract(text)
                llm_ents = clean_llm_entities(llm_ents, text)
            except Exception as e:
                print(f"LLM error: {e}")
                llm_ents = []
        else:
            llm_ents = []

        # Step 3: merge
        merged = merge_rule_and_llm(rule_ents, llm_ents, self.rule_weight)
        return merged


def process_file_hybrid(input_path: str, output_path: str, hybrid: HybridNER):
    text = open(input_path, encoding="utf-8").read()
    entities = hybrid.extract(text)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    return entities


def run_batch_hybrid(input_dir: str, output_dir: str, zip_path: str,
                     hybrid: HybridNER, total: int = 100):
    os.makedirs(output_dir, exist_ok=True)
    total_entities = 0
    type_counts = {"THUỐC": 0, "TRIỆU_CHỨNG": 0, "CHẨN_ĐOÁN": 0}
    t0 = time.time()
    for i in range(1, total + 1):
        in_file = os.path.join(input_dir, f"{i}.txt")
        out_file = os.path.join(output_dir, f"{i}.json")
        if os.path.exists(in_file):
            ents = process_file_hybrid(in_file, out_file, hybrid)
            total_entities += len(ents)
            for e in ents:
                type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
            if i % 10 == 0:
                print(f"  [{i}/{total}] {time.time()-t0:.1f}s")

    print(f"\nHybrid total: {total_entities}, types: {type_counts}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, total + 1):
            jf = os.path.join(output_dir, f"{i}.json")
            if os.path.exists(jf):
                zf.write(jf, f"output/{i}.json")
    print(f"Created: {zip_path}")


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="LLM model key (None = rules only)")
    ap.add_argument("--vram", type=float, default=24.0)
    ap.add_argument("--input_dir", default="input/input")
    ap.add_argument("--output_dir", default="output_hybrid")
    ap.add_argument("--zip_path", default="output_hybrid.zip")
    ap.add_argument("--no_llm", action="store_true", help="Skip LLM, rules only")
    ap.add_argument("--rule_weight", type=float, default=0.6)
    args = ap.parse_args()

    llm_extractor = None
    if not args.no_llm:
        from llm_inference import LLMExtractor
        llm_extractor = LLMExtractor(args.model, vram_gb=args.vram)
        llm_extractor.load()

    hybrid = HybridNER(llm_extractor=llm_extractor, use_llm=not args.no_llm,
                       rule_weight=args.rule_weight)
    run_batch_hybrid(args.input_dir, args.output_dir, args.zip_path, hybrid)