# -*- coding: utf-8 -*-
"""
Ensemble pipeline: combines rule-based + LLM predictions.
Strategy:
1. Run rule-based to get high-precision base entities.
2. Add LLM predictions that are NEW (not already in rule-based).
3. Use CUI mapping for all drugs.
4. Clean up using word boundaries.
"""
import os, re, json, zipfile, sys
from typing import List, Dict, Tuple, Optional, Set

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from drug_dict_v3 import DRUG_DICT
from pipeline import (
    extract_entities as rule_extract,
    KEY_PHRASES, COMPILED_KEY_PHRASES, SYMPTOMS_SORTED, DIAGNOSES_SORTED,
    find_drugs, lookup_drug_cuis_for_text, classify_assertion,
    detect_section, build_line_offsets, get_section_for_pos,
    ULTRA_SHORT_BLACKLIST, DRUG_BLACKLIST
)


def find_in_text(text: str, snippet: str) -> Optional[Tuple[int, int]]:
    """Find a snippet in text using word boundaries. Returns (start, end) or None."""
    if not snippet or not text:
        return None
    tl = text.lower()
    sl = snippet.lower().strip()
    if len(sl) < 2 or len(sl) > 80:
        return None
    start = 0
    while True:
        idx = tl.find(sl, start)
        if idx == -1:
            return None
        before_ok = (idx == 0) or (not tl[idx-1].isalnum())
        after_idx = idx + len(sl)
        after_ok = (after_idx >= len(tl)) or (not tl[after_idx].isalnum())
        if before_ok and after_ok:
            return (idx, after_idx)
        start = idx + 1


def ensemble_extract(text: str, llm_entities: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Combine rule-based + LLM predictions.
    Returns deduplicated, scored entities.
    """
    # Get rule-based entities
    rule_ents = rule_extract(text)

    if not llm_entities:
        return rule_ents

    # Build used spans from rule-based
    used_spans = set()
    seen_keys = set()
    final_entities = []

    for e in rule_ents:
        s, en = e['position']
        used_spans.add((s, en))
        # Key by text+type+approximate position (loose match)
        seen_keys.add((e['text'].lower().strip(), e['type']))
        final_entities.append(e)

    # Add LLM entities that aren't already in rule-based
    for le in llm_entities:
        t = le.get('text', '').strip()
        etype = le.get('type', '').strip()
        if not t or not etype:
            continue
        if etype not in ('THUỐC', 'TRIỆU_CHỨNG', 'CHẨN_ĐOÁN'):
            continue
        if len(t) < 3 or len(t) > 80:
            continue
        # Filter out LLM hallucinations: phrases starting with context words
        tl = t.lower()
        bad_starts = [
            'tiền sử', 'sử dụng', 'thuốc lá', 'trụ niệu', 'thính lực',
            'da dầu', 'cấy que', 'tai nạn', 'cách đây', 'cần dùng',
            'có thể', 'nếu', 'khi', 'đã', 'đang', 'sẽ', 'vì', 'do ',
            'bệnh nhân', 'người bệnh', 'ông', 'bà', 'anh', 'chị',
            'hôm nay', 'hôm qua', 'tối qua', 'sáng nay',
            'kết quả', 'xét nghiệm', 'chẩn đoán xác định', 'chẩn đoán phân biệt',
            'theo dõi', 'tái khám', 'nhập viện', 'xuất viện', 'khám bệnh',
            'có dấu hiệu', 'có biểu hiện', 'có triệu chứng',
            'ghi nhận', 'phát hiện', 'quan sát',
        ]
        if any(tl.startswith(b) for b in bad_starts):
            continue
        # Filter out entities with special chars (commas, slashes, etc.)
        if any(c in t for c in [',', '/', '(', ')', ':', ';']):
            # But allow parens if surrounded by Vietnamese
            if not re.match(r'^[\w\s\-]+$', t) and '(' in t and ')' in t:
                # Could be like "nấm bẹn" - still bad
                pass
            # If contains comma or slash, skip
            if ',' in t or '/' in t:
                continue
        # Find position in text
        pos = find_in_text(text, t)
        if not pos:
            continue
        s, en = pos
        # Check overlap with existing
        overlap = any(not (en <= us or s >= ue) for us, ue in used_spans)
        if overlap:
            continue
        # Check duplicate by text+type
        if (t.lower(), etype) in seen_keys:
            continue

        # Get CUI if drug
        candidates = []
        if etype == 'THUỐC':
            cuis = lookup_drug_cuis_for_text(t)
            if not cuis:
                # Try fuzzy match in DRUG_DICT
                for k in DRUG_DICT:
                    if k in t.lower() or t.lower() in k:
                        cuis.append(DRUG_DICT[k])
                        break
            # CRITICAL: Skip LLM drugs without any drug dictionary match
            # This filters out hallucinations like "tiền sử giai đoạn..."
            if not cuis:
                continue
            candidates = list(set(cuis))

        # Get assertions
        line_offsets = build_line_offsets(text)
        sections = []
        cur = None
        for line in text.split('\n'):
            d = detect_section(line)
            if d: cur = d
            sections.append(cur)
        section_type = get_section_for_pos(s, line_offsets, sections)
        assertions = classify_assertion(s, en, text, section_type, etype)

        # Merge with LLM assertions if provided
        llm_assertions = le.get('assertions', [])
        for a in llm_assertions:
            if a in ('isHistorical', 'isNegated', 'isSuspected') and a not in assertions:
                assertions.append(a)

        used_spans.add((s, en))
        seen_keys.add((t.lower(), etype))
        final_entities.append({
            'text': t,
            'type': etype,
            'candidates': candidates,
            'assertions': assertions,
            'position': [s, en],
        })

    # Sort by position
    final_entities.sort(key=lambda x: x['position'][0])

    # Final dedup by exact position
    seen_pos = set()
    dedup = []
    for e in final_entities:
        k = (e['position'][0], e['position'][1], e['type'])
        if k in seen_pos:
            continue
        seen_pos.add(k)
        dedup.append(e)

    return dedup


def process_file_with_llm(input_path: str, output_path: str, llm_path: Optional[str] = None):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    llm_ents = None
    if llm_path and os.path.exists(llm_path):
        with open(llm_path, 'r', encoding='utf-8') as f:
            llm_ents = json.load(f)

    entities = ensemble_extract(text, llm_ents)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    return entities


def run_ensemble(input_dir: str, output_dir: str, llm_dir: Optional[str] = None,
                 zip_path: Optional[str] = None, total: int = 100):
    os.makedirs(output_dir, exist_ok=True)
    total_entities = 0
    type_counts = {'THUỐC': 0, 'TRIỆU_CHỨNG': 0, 'CHẨN_ĐOÁN': 0}
    for i in range(1, total + 1):
        ip = os.path.join(input_dir, f'{i}.txt')
        op = os.path.join(output_dir, f'{i}.json')
        lp = os.path.join(llm_dir, f'{i}.json') if llm_dir else None
        if os.path.exists(ip):
            entities = process_file_with_llm(ip, op, lp)
            total_entities += len(entities)
            for e in entities:
                type_counts[e['type']] = type_counts.get(e['type'], 0) + 1
    print(f'Total: {total_entities}, Types: {type_counts}')
    if zip_path:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i in range(1, total + 1):
                jf = os.path.join(output_dir, f'{i}.json')
                if os.path.exists(jf):
                    zf.write(jf, f'output/{i}.json')
        print(f'Created: {zip_path}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', default=r'D:\projects\Viettel AI race\input_turn2_vong1\input')
    ap.add_argument('--output_dir', default=r'D:\projects\Viettel AI race\ensemble_output')
    ap.add_argument('--llm_dir', default=r'D:\projects\Viettel AI race\llm_gt')
    ap.add_argument('--zip_path', default=r'D:\projects\Viettel AI race\output.zip')
    ap.add_argument('--total', type=int, default=100)
    args = ap.parse_args()
    run_ensemble(args.input_dir, args.output_dir, args.llm_dir, args.zip_path, args.total)