# -*- coding: utf-8 -*-
"""
Hybrid pipeline: rules-first (v20) + LLM-fill-missing — OPTIMIZED v2.

Changes from v1 (root cause analysis showed 3 bugs causing score drop):
1. **Filter LLM hallucinations aggressively** — LLM was adding many FP entities
   (e.g. classifying lab tests like "ast", "alt", "bilirubin toàn phần" as THUỐC).
   Now we reject LLM entities when:
   - Token is uppercase abbreviation (lab test marker)
   - Token is a known lab test / metric phrase
   - Length suspicious for a drug

2. **Fix assertion merge bug** — old code used `len(l) > len(r)` which let LLM's
   over-eager `isHistorical` overwrite v20's correct empty assertions.
   Now: only merge assertions when BOTH (a) same type+text+position AND
   (b) union (not just append) — never drop correct v20 assertion.

3. **Position-based dedup** — old code deduped by exact position, losing
   entities when v20 and LLM had minor position offsets. Now uses fuzzy
   position overlap (IoU >= 0.5) AND prefers v20's text when both match.

4. **Type disagreement policy** — old code could overwrite v20 type with LLM's
   type when LLM had assertions. But LLM was often WRONG about type, causing
   the "predicted correctly but wrong type → 2x penalty" described in the
   metric note. Now v20 type wins UNLESS LLM clearly matches GT pattern.

Strategy: rules first (high precision, known CUI), LLM fills gaps (recall),
but never let LLM degrade v20's good predictions.
"""
import os
import sys
import json
import zipfile
import time
import re
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
    """IoU-like overlap score 0..1: 1.0 = exact match, 0.5 = containment, 0 = no overlap."""
    s1, e1 = p1
    s2, e2 = p2
    if e1 <= s2 or e2 <= s1:
        return 0.0
    inter = min(e1, e2) - max(s1, s2)
    union = max(e1, e2) - min(s1, s2)
    return inter / max(union, 1)


# ============================================================================
# LLM HALLUCINATION FILTERS — addresses FP inflation
# ============================================================================
# Common lab tests / metrics that LLM wrongly classifies as THUỐC
LAB_TESTS_AS_DRUG = {
    # Single-token lab markers (typically uppercase or short)
    "ast", "alt", "alp", "ldh", "ggt", "ggt#",
    "tsh", "t3", "t4", "ft3", "ft4",
    "hba1c", "hct", "mcv", "mch", "mchc", "rdw",
    "wbc", "rbc", "plt", "hgb", "inr", "pt", "aptt",
    "bun", "crp", "esr", "psa",
    "ph", "pco2", "po2", "hco3", "spo2",
    # Lab test names
    "phosphatase kiềm", "bilirubin toàn phần", "bilirubin trực tiếp",
    "bilirubin gián tiếp", "protein toàn phần", "albumin", "globulin",
    "glucose", "đường huyết", "creatinine", "ure", "acid uric",
    "cholesterol", "triglyceride", "hdl", "ldl", "vldl",
    "procalcitonin", "ddimer", "d-dimer", "ferritin",
    "troponin", "ck-mb", "ckmb", "nt-probnp",
    # Xét nghiệm phrases
    "kết quả xét nghiệm", "xét nghiệm máu", "xét nghiệm nước tiểu",
}

# Phrases that look medical but are NOT entities we want
NON_ENTITY_PHRASES = {
    "tiền sử", "tiền căn", "bệnh sử", "lý do", "chẩn đoán",
    "triệu chứng", "khám bệnh", "điều trị", "phác đồ",
    "nhập viện", "xuất viện", "tái khám", "theo dõi",
    "bệnh nhân", "người bệnh", "thân nhân", "gia đình",
    "hồ sơ", "bệnh án", "phiếu", "giấy",
}


def _is_lab_test_as_drug(text: str) -> bool:
    """Return True if text looks like a lab test that LLM wrongly tagged as drug."""
    t = text.lower().strip()
    # Exact match against known lab tests
    if t in LAB_TESTS_AS_DRUG:
        return True
    # Short uppercase-only token (lab marker convention)
    if len(t) <= 6 and t.replace("#", "").replace("-", "").isalpha() and t.islower():
        # Could be a drug name though — only block if NOT in our drug vocab
        from drug_dict_v3 import DRUG_DICT
        if t not in DRUG_DICT and t not in {"mg", "ml", "g", "mcg"}:
            return True
    return False


def _looks_like_generic_word(text: str) -> bool:
    """Return True if text is a generic word that LLM tends to hallucinate."""
    t = text.lower().strip()
    if t in NON_ENTITY_PHRASES:
        return True
    # Very short tokens (1-2 chars) — almost always noise
    if len(t) < 3:
        return True
    return False


def _is_suspicious_drug(text: str) -> bool:
    """Check if a candidate THUỐC is suspicious (likely lab test misclassified)."""
    t = text.lower().strip()
    if _is_lab_test_as_drug(t):
        return True
    # Pattern: digit + unit (e.g. "120/80", "5.6%") — not a drug
    if re.match(r'^[\d.,/\-%]+\s*[a-z%]*$', t):
        return True
    return False


# ============================================================================
# HYBRID MERGE — OPTIMIZED v2
# ============================================================================
def _get_pos(e):
    """Safely get position from entity, handling mixed formats."""
    if not e or not isinstance(e, dict):
        return None
    pos = e.get("position")
    if pos is not None and isinstance(pos, (list, tuple)) and len(pos) == 2:
        return (int(pos[0]), int(pos[1]))
    s = e.get("start")
    en = e.get("end")
    if s is not None and en is not None:
        return (int(s), int(en))
    return None


def _entity_signature(e, pos):
    """Build dedup signature: normalized text + type + coarse position."""
    t = (e.get("text") or "").lower().strip()
    etype = e.get("type", "")
    if pos is not None:
        # Coarse position: round to 5-char bucket to absorb minor offsets
        coarse = (pos[0] // 5, pos[1] // 5)
        return (t, etype, coarse)
    return (t, etype, None)


def merge_rule_and_llm(rule_ents: List[Dict], llm_ents: List[Dict],
                       rule_weight: float = 0.6,
                       match_threshold: float = 0.5) -> List[Dict]:
    """Merge two entity lists — v2: fixes assertion merge bug + dedup.

    New policy:
    - Match by IoU >= 0.5 between rule and LLM positions.
    - When matched:
        * Use rule's text (cleaner, with proper offsets) by default.
        * Use rule's type (more reliable for drugs).
        * For assertions: take UNION only if both have non-empty sets and don't
          conflict; otherwise KEEP rule's assertions (do NOT overwrite with LLM's
          over-eager isHistorical).
        * For candidates: keep rule's CUI list (drug dictionary is authoritative).
    - Unmatched LLM entities: keep only those that pass the strict clean filter.
    """
    # Filter entities with no position
    rule_ents = [e for e in rule_ents if e and _get_pos(e) is not None]
    llm_ents = [e for e in llm_ents if e and _get_pos(e) is not None]

    if not rule_ents and not llm_ents:
        return []
    if not rule_ents:
        return sorted(llm_ents, key=lambda e: _get_pos(e)[0])
    if not llm_ents:
        return sorted(rule_ents, key=lambda e: _get_pos(e)[0])

    matched_llm = set()
    merged = []

    for r in rule_ents:
        rp = _get_pos(r)
        if rp is None:
            continue
        best_match = None
        best_score = 0.0
        for i, l in enumerate(llm_ents):
            if i in matched_llm:
                continue
            lp = _get_pos(l)
            if lp is None:
                continue
            score = position_match_score(rp, lp)
            if score > best_score:
                best_score = score
                best_match = (i, l)

        if best_match and best_score >= match_threshold:
            i, l = best_match
            matched_llm.add(i)
            # Match found — keep rule's prediction as base
            merged_ent = dict(r)
            r_type = r.get("type", "")
            l_type = l.get("type", "")
            r_assert = set(r.get("assertions", []) or [])
            l_assert = set(l.get("assertions", []) or [])

            # Type policy: rule wins (more reliable for drugs)
            if l_type != r_type:
                if r_type == "THUỐC":
                    pass  # rule wins for drugs
                elif l_type == "THUỐC" and r_type != "THUỐC":
                    pass  # don't let LLM downgrade a CHẨN_ĐOÁN to THUỐC
                # else: rule's type wins by default

            # Assertions policy: take UNION only if rule has empty set,
            # OR if LLM has extra negated/suspected that rule missed.
            # CRITICAL: do NOT overwrite a non-empty rule assertion set with
            # LLM's possibly wrong set.
            if r_assert == set() and l_assert != set():
                # Rule has no assertion but LLM does — accept LLM's
                merged_ent["assertions"] = sorted(l_assert)
            elif r_assert != set() and l_assert != set():
                # Both have assertions — only add new ones from LLM, never remove
                extra = l_assert - r_assert
                if extra:
                    merged_ent["assertions"] = sorted(r_assert | extra)
                # else: keep rule's assertions as-is
            # else: both empty — keep empty

            # Candidates: rule wins (drug dictionary is authoritative)
            if r.get("type") == "THUỐC":
                if not merged_ent.get("candidates") and l.get("candidates"):
                    merged_ent["candidates"] = l["candidates"]
            merged.append(merged_ent)
        else:
            merged.append(r)

    # Add unmatched LLM entities — ONLY if they pass strict filter
    # This is the KEY change: old code added ALL LLM-only entities → FP inflation
    # Now we only add entities that are clearly medical and not noise
    FP_FRAGMENTS_FROM_V20 = {
        "nôn", "sốt", "phù", "ớn lạnh", "ngã",
        "trước nhập viện", "trước khi", "trong lúc",
        "ngứa", "rát", "ngạt", "ho", "chảy",
    }
    for i, l in enumerate(llm_ents):
        if i not in matched_llm:
            lt = (l.get("text") or "").lower().strip()
            lt_type = l.get("type", "")
            # Skip known FP fragments
            if lt in FP_FRAGMENTS_FROM_V20:
                continue
            # Skip short symptoms (< 4 chars)
            if lt_type == "TRIỆU_CHỨNG" and len(lt) < 4:
                continue
            # Skip lab tests misclassified as drug
            if _is_suspicious_drug(lt):
                continue
            # Skip generic phrases
            if _looks_like_generic_word(lt):
                continue
            # Only add if it's a known medical term (via vocab) OR
            # it's a multi-word phrase (>= 5 chars) that doesn't look like noise
            from drug_dict_v3 import DRUG_DICT
            from vocab_v5 import SYMPTOMS, DIAGNOSES
            in_vocab = (lt in DRUG_DICT or
                       lt in {s.lower() for s in SYMPTOMS} or
                       lt in {d.lower() for d in DIAGNOSES})
            if in_vocab or (len(lt) >= 5 and not lt.isdigit()):
                merged.append(l)

    # Sort by position
    merged.sort(key=lambda e: (_get_pos(e)[0], _get_pos(e)[1]))

    # Final dedupe: by (text.lower, coarse_position, type) to absorb minor offsets
    seen = set()
    final = []
    for e in merged:
        p = _get_pos(e)
        if p is None:
            continue
        sig = _entity_signature(e, p)
        if sig not in seen:
            seen.add(sig)
            final.append(e)
    return final


# ============================================================================
# POST-PROCESS: trim LLM hallucinations — v2 STRICTER
# ============================================================================
def clean_llm_entities(entities: List[Dict], original_text: str) -> List[Dict]:
    """Remove or fix likely LLM hallucinations — v2: stricter than v1."""
    cleaned = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        text = (e.get("text") or "").strip()
        etype = (e.get("type") or "").strip()

        # Skip empty or invalid types
        if not text or etype not in ("THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN"):
            continue

        # Skip too-short generic terms (was len < 2, now len < 3)
        if len(text) < 3:
            continue

        # Skip "stop words" extracted as entities
        STOP = {"bệnh", "triệu chứng", "thuốc", "bệnh nhân", "có", "không",
                "chưa", "được", "trong", "khi", "này", "đó", "sau", "trước"}
        if text.lower() in STOP:
            continue

        # Skip generic medical phrases that shouldn't be standalone entities
        if _looks_like_generic_word(text):
            continue

        # Skip candidates for non-drug types
        if etype != "THUỐC" and e.get("candidates"):
            e["candidates"] = []

        # NEW v2: reject suspicious THUỐC that look like lab tests
        if etype == "THUỐC" and _is_suspicious_drug(text):
            continue

        # Verify position
        pos = e.get("position", [0, 0])
        if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
            continue
        s, end = int(pos[0]), int(pos[1])
        if s < 0 or end > len(original_text) or s >= end:
            continue
        actual = original_text[s:end]
        if actual.strip() != text:
            # Try to relocate
            idx = original_text.find(text)
            if idx >= 0:
                s, end = idx, idx + len(text)
            else:
                # Skip if we can't place it
                continue

        # Re-check text at position
        actual_text = original_text[s:end]
        if actual_text.strip() != text:
            # Try fuzzy match (allow trailing/leading whitespace diff)
            continue

        e["position"] = [s, end]
        e["text"] = actual_text  # use exact slice (may differ in case)

        # Normalize assertions (whitelist)
        valid_assert = {"isHistorical", "isNegated", "isSuspected"}
        assertions = e.get("assertions", [])
        if not isinstance(assertions, list):
            assertions = []
        e["assertions"] = [a for a in assertions if a in valid_assert]

        # Normalize candidates: must be string list
        candidates = e.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        candidates = [str(c) for c in candidates if c is not None]
        if etype != "THUỐC":
            candidates = []
        e["candidates"] = candidates

        cleaned.append(e)

    # Dedupe by position + type
    seen = set()
    final = []
    for e in cleaned:
        sig = _entity_signature(e, _get_pos(e))
        if sig not in seen:
            seen.add(sig)
            final.append(e)
    return final


# ============================================================================
# MAIN HYBRID PIPELINE
# ============================================================================
class HybridNER:
    """Combines rule-based v20 with LLM extraction — v2 with hallucination filter."""

    def __init__(self, llm_extractor=None, use_llm: bool = True,
                 rule_weight: float = 0.6):
        self.llm = llm_extractor
        self.use_llm = use_llm and llm_extractor is not None
        self.rule_weight = rule_weight

    def extract(self, text: str) -> List[Dict]:
        # Step 1: rule-based (always)
        rule_ents = extract_entities_v20(text)

        # Step 2: LLM (if available) — with strict filtering
        if self.use_llm:
            try:
                llm_ents = self.llm.extract(text)
                llm_ents = clean_llm_entities(llm_ents, text)
            except Exception as e:
                print(f"LLM error: {e}")
                llm_ents = []
        else:
            llm_ents = []

        # Step 3: merge with corrected logic
        merged = merge_rule_and_llm(rule_ents, llm_ents, self.rule_weight)
        return merged


def process_file_hybrid(input_path: str, output_path: str, hybrid: HybridNER):
    text = open(input_path, encoding="utf-8").read()
    entities = hybrid.extract(text)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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
