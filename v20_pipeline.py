# -*- coding: utf-8 -*-
"""
Vietnamese Medical NER Pipeline v20 — Exact v11 replica
All 6 steps: KEY_PHRASES → drugs → verb-drugs → section vocab → lido → viêm reclassify
"""
import os, re, json, zipfile, sys
from typing import List, Dict, Tuple, Optional, Set

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from drug_dict_v3 import find_drugs_in_text, lookup_drug_cui, lookup_drug_cuis, DRUG_DICT
from vocab_v5 import SYMPTOMS, DIAGNOSES


# ============================================================================
# SECTION DETECTION — same as v11
# ============================================================================
SECTION_PATTERNS = [
    (r'thuốc\s+trước', 'THUỐC'),
    (r'thuốc\s+(?:đang|hiện|dùng|tại|nhà)', 'THUỐC'),
    (r'phác\s+đồ\s+điều\s+trị', 'THUỐC'),
    (r'điều\s+trị\s+(?:hiện\s+tại|tại\s+nhà)', 'THUỐC'),
    (r'tiền\s+sử\s+bệnh\s*(?:nội\s+khoa|lý)?', 'CHẨN_ĐOÁN'),
    (r'tiền\s+căn', 'CHẨN_ĐOÁN'),
    (r'bệnh\s+lý\s+(?:mãn\s+tính|nền|kèm)', 'CHẨN_ĐOÁN'),
    (r'chẩn\s+đoán\s+(?:xác|sơ|cuối|cùng|trước|sau|ra|phân|biệt)?', 'CHẨN_ĐOÁN'),
    (r'kết\s+quả\s+(?:xét\s+nghiệm|chẩn\s+đoán)', 'CHẨN_ĐOÁN'),
    (r'tiền\s+sử\s+phẫu\s+thuật', 'CHẨN_ĐOÁN'),
    (r'phẫu\s+thuật\s*/\s*thủ\s+thuật', 'CHẨN_ĐOÁN'),
    (r'triệu\s+chứng(?:\s+cơ\s+năng|\s+thực\s+thể|\s+hiện\s+tại|\s+khi\s+nhập)?', 'TRIỆU_CHỨNG'),
    (r'bệnh\s+sử(?:\s+hiện\s+tại|\s*\s+ngoại\s+khoa|\s*\s+nội\s+khoa)?', 'TRIỆU_CHỨNG'),
    (r'lịch\s+sử\s+bệnh(?:\s+hiện\s+tại)?', 'TRIỆU_CHỨNG'),
    (r'lý\s+do\s+(?:nhập|vào|khám|ra)', 'TRIỆU_CHỨNG'),
    (r'đặc\s+điểm\s+triệu\s+chứng', 'TRIỆU_CHỨNG'),
    (r'tình\s+trạng\s+(?:ngay|trước|khi|lúc)', 'TRIỆU_CHỨNG'),
    (r'diễn\s+biến', 'TRIỆU_CHỨNG'),
    (r'các\s+yếu\s+tố\s+nguy\s+cơ', 'TRIỆU_CHỨNG'),
    (r'lúc\s+vào\s+viện', 'TRIỆU_CHỨNG'),
    (r'khám\s+tại\s+bệnh\s+viện', 'TRIỆU_CHỨNG'),
    (r'đánh\s+giá\s+tại\s+bệnh\s+viện', 'CHẨN_ĐOÁN'),
    (r'cận\s+lâm\s+sàng', 'CHẨN_ĐOÁN'),
]
COMPILED_SECTION = [(re.compile(p, re.IGNORECASE), t) for p, t in SECTION_PATTERNS]


def detect_section_type(line_text: str) -> Optional[str]:
    line_clean = re.sub(r'^\s*[\d]+\.?\s*', '', line_text).strip()
    for pattern, etype in COMPILED_SECTION:
        if pattern.search(line_clean.lower()):
            return etype
    return None


def build_line_offsets(text: str) -> List[int]:
    offsets, pos = [], 0
    for line in text.split('\n'):
        offsets.append(pos)
        pos += len(line) + 1
    return offsets


# ============================================================================
# ASSERTIONS — same as v11
# ============================================================================
NEG_RE = [
    re.compile(r'\bkhông\s+(?:có|còn|thấy|ghi\s+nhận|xuất\s+hiện|đau|thở|sốt|buồn|sợ|ngứa|ra|cảm|ớn|phù|ho|chảy|rát|ngạt)', re.IGNORECASE),
    re.compile(r'\bchưa\s+(?:có|từng|bảo\s+giờ)', re.IGNORECASE),
    re.compile(r'\bko\s+(?:có|còn)', re.IGNORECASE),
    re.compile(r'\bkhông\s+?\-\s+?(?:có|còn)', re.IGNORECASE),
    re.compile(r'\b(vắng|mất)\s+(?:mặt|tên)', re.IGNORECASE),
    re.compile(r'\b(?:no|not|absent|negative)\b', re.IGNORECASE),
    re.compile(r'\b(?:-|–)\s*(?:không|chưa)', re.IGNORECASE),
]
HIST_RE = [
    re.compile(r'\btiền\s+sử\b', re.IGNORECASE),
    re.compile(r'\bđã\s+(?:dùng|sử|mắc|bị|có|tiêm|uống|phẫu\s+thuật)', re.IGNORECASE),
    re.compile(r'\btrước\s+(?:khi|đây|kia)', re.IGNORECASE),
    re.compile(r'\b(cũ|lâu|năm|tháng|ngày)\s+(?:trước|nay|đây)', re.IGNORECASE),
    re.compile(r'\bđiều\s+trị\s+(?:trước|trước\s+đây)', re.IGNORECASE),
    re.compile(r'\bcó\s+bệnh\s+từ', re.IGNORECASE),
    re.compile(r'\btừ\s+(?:trước|năm|tháng)', re.IGNORECASE),
    re.compile(r'\bnăm\s+(?:ngoái|ày)', re.IGNORECASE),
    re.compile(r'\bnhiều\s+năm\s+trước', re.IGNORECASE),
    re.compile(r'\btừng\s+(?:bị|mắc|dùng)', re.IGNORECASE),
    re.compile(r'\b(cắt|bỏ)\s+(?:đại\s+tràng|dạ\s+dày|thận|phổi)', re.IGNORECASE),
]
SUSP_RE = [
    re.compile(r'\bcó\s+thể\b', re.IGNORECASE),
    re.compile(r'\bnghi\s*ngờ\b', re.IGNORECASE),
    re.compile(r'\bnghĩ\s+đến\b', re.IGNORECASE),
    re.compile(r'\bchưa\s+rõ\b', re.IGNORECASE),
    re.compile(r'\bđang\s+theo\s+dõi\b', re.IGNORECASE),
    re.compile(r'\bđang\s+xem\s+xét\b', re.IGNORECASE),
    re.compile(r'\bchưa\s+xác\s+định\b', re.IGNORECASE),
    re.compile(r'\bđang\s+chờ\b', re.IGNORECASE),
    re.compile(r'\bchưa\s+loại\s+trừ\b', re.IGNORECASE),
]


def classify_assertion(start: int, end: int, text: str,
                       section_type: Optional[str], entity_type: str) -> List[str]:
    result = []
    ctx_before = text[max(0, start-50):start].lower()
    ctx_after = text[end:min(len(text), end+20)].lower()
    ctx_window = ctx_before + ' ' + ctx_after

    negated = False
    for pat in NEG_RE:
        if pat.search(ctx_window):
            result.append('isNegated')
            negated = True
            break
    if not negated:
        for pat in SUSP_RE:
            if pat.search(ctx_window):
                result.append('isSuspected')
                break

    is_hist = False
    if entity_type == 'THUỐC':
        if section_type == 'THUỐC':
            is_hist = True
        elif any(p.search(ctx_before[:70]) for p in HIST_RE):
            is_hist = True
    elif entity_type == 'CHẨN_ĐOÁN':
        if section_type == 'CHẨN_ĐOÁN':
            if 'kết quả' in ctx_before.lower() or 'chẩn đoán' in ctx_before.lower():
                is_hist = False
            else:
                is_hist = True
        elif any(p.search(ctx_before) for p in HIST_RE):
            is_hist = True
    else:
        if section_type not in ('TRIỆU_CHỨNG',) and any(p.search(ctx_before) for p in HIST_RE):
            is_hist = True
    if is_hist:
        result.append('isHistorical')
    return result


# ============================================================================
# DRUG BLACKLIST — same as v11
# ============================================================================
DRUG_BLACKLIST = {
    'thuốc trước', 'thuốc sau', 'thuốc nào', 'thuốc đó', 'thuốc này',
    'thuốc đang', 'thuốc được', 'thuốc hiện', 'thuốc cũ', 'thuốc mới',
    'thuốc trị', 'thuốc điều', 'thuốc bệnh', 'thuốc nam', 'thuốc bắc',
    'thuốc tây', 'ống hôm', 'ống hàng', 'siêu âm', 'x-quang',
    'bình thường', 'việc dùng', 'việc uống', 'cơm', 'thức ăn',
    'đồ uống', 'nước uống', 'túi thuốc', 'thuốc theo',
    'bệnh nhân', 'bệnh viện', 'bác sĩ',
}

# Fragment blacklist — same as v11 (small, only most noisy)
FRAGMENT_BLACKLIST = {
    'yếu', 'mệt', 'tê',
    'buồn', 'lo', 'chóng', 'váng', 'ù', 'ngất',
    'ợ', 'táo', 'tiêu', 'chảy', 'đầy',
}


def is_false_drug(text: str) -> bool:
    t = text.lower().strip()
    if t in DRUG_BLACKLIST:
        return True
    if len(t) < 4:
        return True
    if re.match(r'^[\d\.\/\s]+$', t):
        return True
    return False


def is_fragment_symptom(term: str) -> bool:
    t = term.lower().strip()
    if len(t) < 3:
        return True
    if t in FRAGMENT_BLACKLIST:
        return True
    return False


# ============================================================================
# KEY PHRASES — v11's
# ============================================================================
KEY_PHRASES = [
    (r'nhịp\s+xoang\s+chiếm\s+ưu\s+thế', 'CHẨN_ĐOÁN'),
    (r'ngoại\s+tâm\s+thu\s+nhĩ', 'CHẨN_ĐOÁN'),
    (r'ngoại\s+tâm\s+thu\s+thất', 'CHẨN_ĐOÁN'),
    (r'nghẽn\s+tắc\s+và\s+hẹp\s+động\s+mạch\s+cảnh', 'CHẨN_ĐOÁN'),
    (r'chấn\s+thương\s+gãy\s+xương\s+sườn(?:\s+(?:trái|phải))?', 'CHẨN_ĐOÁN'),
    (r'hội\s+chứng\s+mạch\s+vành\s+cấp', 'CHẨN_ĐOÁN'),
    (r'nhồi\s+máu\s+cơ\s+tim', 'CHẨN_ĐOÁN'),
    (r'bệnh\s+(?:mạch\s+vành|thiếu\s+máu\s+cơ\s+tim)', 'CHẨN_ĐOÁN'),
    (r'nhịp\s+xoang(?!\s+chiếm)', 'CHẨN_ĐOÁN'),
    (r'hẹp\s+động\s+mạch\s+cảnh', 'CHẨN_ĐOÁN'),
    (r'đau\s+thắt\s+ngực(?:\s+(?:ổn\s+định|không\s+ổn\s+định|ổn định))?', 'CHẨN_ĐOÁN'),
    (r'suy\s+tim', 'CHẨN_ĐOÁN'),
    (r'vết\s+thương\s+thấu\s+bụng', 'CHẨN_ĐOÁN'),
    (r'đái\s+tháo\s+đường(?:\s+(?:type\s*[12]|týp\s*[12]))?', 'CHẨN_ĐOÁN'),
    (r'tăng\s+huyết\s+áp', 'CHẨN_ĐOÁN'),
    (r'tiểu\s+tiện\s+không\s+tự\s+chủ', 'TRIỆU_CHỨNG'),
    (r'sa\s+âm\s+đạo', 'TRIỆU_CHỨNG'),
    (r'giọng\s+khàn', 'TRIỆU_CHỨNG'),
    (r'bàng\s+quang\s+căng', 'TRIỆU_CHỨNG'),
    (r'bí\s+tiếu\s+liên\s+tục', 'TRIỆU_CHỨNG'),
    (r'tổn\s+thương\s+dây\s+thanh\s+quản', 'TRIỆU_CHỨNG'),
    (r'giảm\s+dung\s+nạp\s+gắng\s+sức', 'TRIỆU_CHỨNG'),
    (r'đau\s+quanh\s+vết\s+mổ', 'TRIỆU_CHỨNG'),
    (r'đau\s+hạ\s+sườn\s+phải', 'TRIỆU_CHỨNG'),
    (r'khò\s+khè', 'TRIỆU_CHỨNG'),
    (r'triệu\s+chứng\s+đường\s+hô\s+hấp\s+trên', 'TRIỆU_CHỨNG'),
    (r'\bsốt\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\b', 'TRIỆU_CHỨNG'),
    (r'\bho\b', 'TRIỆU_CHỨNG'),
    (r'\bnôn\b', 'TRIỆU_CHỨNG'),
    (r'\bphù\b', 'TRIỆU_CHỨNG'),
    (r'\bngứa\b', 'TRIỆU_CHỨNG'),
    (r'\bvàng\s+da\b', 'TRIỆU_CHỨNG'),
    (r'\bchóng\s+mặt\b', 'TRIỆU_CHỨNG'),
    (r'\bmất\s+ngủ\b', 'TRIỆU_CHỨNG'),
    (r'béo\s+phì', 'CHẨN_ĐOÁN'),
]
COMPILED_KEY_PHRASES = [(re.compile(p, re.IGNORECASE), t) for p, t in KEY_PHRASES]


def find_terms_in_line(line: str, line_offset: int, terms: List[str], etype: str,
                       used_spans: List[Tuple[int, int]], seen: Set[str],
                       min_len: int = 3) -> List[Dict]:
    if not line.strip():
        return []
    found = []
    text_lower = line.lower()
    sorted_terms = sorted(terms, key=lambda x: -len(x))
    for term in sorted_terms:
        term_lower = term.lower()
        tlen = len(term_lower)
        if tlen < min_len:
            continue
        if is_fragment_symptom(term):
            continue
        start = 0
        while True:
            idx = text_lower.find(term_lower, start)
            if idx == -1:
                break
            before_ok = (idx == 0) or (not text_lower[idx-1].isalnum())
            after_idx = idx + tlen
            after_ok = (after_idx >= len(text_lower)) or (not text_lower[after_idx].isalnum())
            if before_ok and after_ok:
                abs_start = line_offset + idx
                abs_end = line_offset + after_idx
                has_overlap = any(abs_start < ue and abs_end > us for us, ue in used_spans)
                if not has_overlap:
                    key = term_lower
                    seen.add(key)
                    used_spans.append((abs_start, abs_end))
                    found.append({
                        'text': line[idx:after_idx],
                        'type': etype,
                        'position': [abs_start, abs_end],
                    })
            start = idx + 1
    return found


def _get_section_for_pos(pos: int, line_offsets: List[int], sections: List) -> Optional[str]:
    for i, offset in enumerate(line_offsets):
        if pos < offset or i == len(line_offsets) - 1:
            return sections[i - 1] if i > 0 else sections[0]
    return sections[-1] if sections else None


# ============================================================================
# VERB-PATTERN DRUG EXTRACTION — v11 Step 2
# ============================================================================
def _extract_verb_drugs(text: str, used_spans, seen) -> List[Dict]:
    """Extract drugs via verb pattern."""
    found = []
    patterns = [
        r'\b(?:uống|tiêm|truyền|dùng|sử\s*dụng|đang\s*dùng|điều\s*trị|chỉ\s*định|đã\s*dùng)\s+'
        r'([a-zà-ỹ][\w\-]{3,30}(?:\s+[a-zà-ỹ][\w\-]{2,15}){0,2})',
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            drug = m.group(1).strip()
            dl = drug.lower()
            if len(dl) < 4 or dl in DRUG_BLACKLIST:
                continue
            if any(dl.startswith(s) for s in ['thuốc', 'bệnh', 'việc', 'ngày', 'bác']):
                continue
            m_start = m.start(1)
            end = m.end(1)
            if any(not (end <= us or m_start >= ue) for us, ue in used_spans):
                continue
            cui = lookup_drug_cui(drug)
            if not cui:
                continue
            key = drug.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append({'text': drug, 'start': m_start, 'end': end, 'cui': cui})
    return found


# ============================================================================
# LÝ DO NHẬP VIỆN EXTRACTION — v11 Step 5
# ============================================================================
def _extract_lido_lines(text: str, lines: List, line_offsets: List,
                        sections: List, used_spans, seen, entities):
    """Extract from 'Lý do nhập viện:' lines."""
    ly_do_re = re.compile(r'lý\s+do\s+(?:nhập|vào|khám)(?:\s*viện)?\s*:?\s*([^\n]+)', re.IGNORECASE)
    for m in ly_do_re.finditer(text):
        content = m.group(1).strip()
        if not content or len(content) < 3:
            continue
        line_no = text[:m.start()].count('\n')
        section_type = sections[line_no] if line_no < len(sections) else 'TRIỆU_CHỨNG'
        content_lower = content.lower()
        for terms, target_type in [(DIAGNOSES, 'CHẨN_ĐOÁN'), (SYMPTOMS, 'TRIỆU_CHỨNG')]:
            for term in sorted(terms, key=lambda x: -len(x)):
                if is_fragment_symptom(term):
                    continue
                if term in content_lower:
                    abs_start = m.start(1) + content_lower.find(term)
                    abs_end = abs_start + len(term)
                    key = term
                    if key not in seen:
                        if not any(not (abs_end <= us or abs_start >= ue) for us, ue in used_spans):
                            seen.add(key)
                            used_spans.append((abs_start, abs_end))
                            assertions = classify_assertion(abs_start, abs_end, text, section_type, target_type)
                            entities.append({
                                'text': term,
                                'type': target_type,
                                'candidates': [],
                                'assertions': assertions,
                                'position': [abs_start, abs_end],
                            })
                            break
            break


# ============================================================================
# MAIN — v11's exact 6-step pipeline
# ============================================================================
def extract_entities_v20(text: str) -> List[Dict]:
    if not text or not text.strip():
        return []

    entities = []
    used_spans: List[Tuple[int, int]] = []
    seen: Set[str] = set()
    lines = text.split('\n')

    sections = []
    current_section = None
    for line in lines:
        detected = detect_section_type(line)
        if detected:
            current_section = detected
        sections.append(current_section)

    line_offsets = build_line_offsets(text)

    # --- Step 0: KEY_PHRASES ---
    for pat, etype in COMPILED_KEY_PHRASES:
        for m in pat.finditer(text):
            abs_start = m.start()
            abs_end = m.end()
            phrase = m.group().strip()
            key = phrase.lower()
            if len(key) < 4:
                continue
            if key in seen:
                continue
            if any(not (abs_end <= us or abs_start >= ue) for us, ue in used_spans):
                continue
            seen.add(key)
            used_spans.append((abs_start, abs_end))
            section_type = _get_section_for_pos(abs_start, line_offsets, sections)
            assertions = classify_assertion(abs_start, abs_end, text, section_type, etype)
            entities.append({
                'text': phrase, 'type': etype, 'candidates': [],
                'assertions': assertions, 'position': [abs_start, abs_end],
            })

    # --- Step 1: Drug extraction ---
    drug_hits = find_drugs_in_text(text)
    for d in drug_hits:
        drug_text = d['text'].strip()
        if is_false_drug(drug_text):
            continue
        drug_clean = ' '.join(drug_text.split())
        drug_clean = re.sub(r'[\.,;:]+$', '', drug_clean).strip()
        if drug_clean and re.match(r'^[\d\.\/\s]+$', drug_clean):
            continue
        if not drug_clean or len(drug_clean) < 4:
            continue
        start, end = d['start'], d['end']
        if any(not (end <= us or start >= ue) for us, ue in used_spans):
            continue
        used_spans.append((start, end))
        section_type = _get_section_for_pos(start, line_offsets, sections)
        assertions = classify_assertion(start, end, text, section_type, 'THUỐC')
        cui_list = lookup_drug_cuis(drug_clean)
        if not cui_list:
            cui_list = [d['cui']] if d.get('cui') else []
        entities.append({
            'text': drug_clean, 'type': 'THUỐC', 'candidates': cui_list,
            'assertions': assertions, 'position': [start, end],
        })

    # --- Step 2: Verb-pattern drug extraction ---
    verb_drugs = _extract_verb_drugs(text, used_spans, seen)
    for dr in verb_drugs:
        start, end = dr['start'], dr['end']
        section_type = _get_section_for_pos(start, line_offsets, sections)
        assertions = classify_assertion(start, end, text, section_type, 'THUỐC')
        entities.append({
            'text': dr['text'], 'type': 'THUỐC',
            'candidates': dr.get('cui', []),
            'assertions': assertions, 'position': [start, end],
        })

    # --- Step 3-4: Section-aware vocab extraction ---
    for i, line in enumerate(lines):
        if i >= len(line_offsets):
            continue
        line_offset = line_offsets[i]
        section_type = sections[i] if i < len(sections) else None
        if not line.strip():
            continue
        effective_section = section_type if section_type else 'TRIỆU_CHỨNG'

        if effective_section == 'THUỐC':
            for etype, terms_to_use in [('TRIỆU_CHỨNG', SYMPTOMS), ('CHẨN_ĐOÁN', DIAGNOSES)]:
                found = find_terms_in_line(line, line_offset, terms_to_use, etype,
                                          used_spans, seen, min_len=3)
                for f in found:
                    s, e = f['position']
                    assertions = classify_assertion(s, e, text, effective_section, etype)
                    entities.append({
                        'text': f['text'], 'type': etype, 'candidates': [],
                        'assertions': assertions, 'position': f['position'],
                    })
        elif effective_section == 'TRIỆU_CHỨNG':
            for etype, terms_to_use in [('TRIỆU_CHỨNG', SYMPTOMS), ('CHẨN_ĐOÁN', DIAGNOSES)]:
                found = find_terms_in_line(line, line_offset, terms_to_use, etype,
                                          used_spans, seen, min_len=3)
                for f in found:
                    s, e = f['position']
                    assertions = classify_assertion(s, e, text, effective_section, etype)
                    entities.append({
                        'text': f['text'], 'type': etype, 'candidates': [],
                        'assertions': assertions, 'position': f['position'],
                    })
        elif effective_section == 'CHẨN_ĐOÁN':
            for etype, terms_to_use in [('CHẨN_ĐOÁN', DIAGNOSES), ('TRIỆU_CHỨNG', SYMPTOMS)]:
                found = find_terms_in_line(line, line_offset, terms_to_use, etype,
                                          used_spans, seen, min_len=3)
                for f in found:
                    s, e = f['position']
                    assertions = classify_assertion(s, e, text, effective_section, etype)
                    entities.append({
                        'text': f['text'], 'type': etype, 'candidates': [],
                        'assertions': assertions, 'position': f['position'],
                    })

    # --- Step 5: Lý do nhập viện ---
    _extract_lido_lines(text, lines, line_offsets, sections, used_spans, seen, entities)

    # --- Step 6: Reclassify 'viêm X' that are actually symptoms ---
    for e in entities:
        if e['type'] == 'CHẨN_ĐOÁN' and e['text'].lower().startswith('viêm '):
            start, end = e['position']
            ctx = text[max(0, start-30):start].lower()
            if re.search(r'\b(cho|vì|do|biểu\s+hiện)\s*$', ctx):
                e['type'] = 'TRIỆU_CHỨNG'

    # Sort + dedupe
    entities.sort(key=lambda x: x['position'][0])
    seen_pos = set()
    deduped = []
    for e in entities:
        pos_key = (e['text'].lower(), e['position'][0], e['position'][1])
        if pos_key not in seen_pos:
            seen_pos.add(pos_key)
            deduped.append(e)

    # ---- POST-PROCESSING: remove FP fragments ----
    # These patterns are known FP in bootstrap that GT doesn't include
    # They appear standalone but GT uses the full phrase
    _FP_FRAGMENTS = {
        # Standalone fragments that should be part of larger phrases
        "nôn", "sốt", "phù", "ớn lạnh", "ngã",
        # Temporal/spatial phrases that aren't medical entities
        "trước nhập viện", "trước khi", "trong lúc",
        # Generic symptoms GT doesn't track as standalone
        "ngứa", "rát", "ngạt", "ho", "chảy",
    }
    # Entities that are fragments of larger phrases (checked by looking for longer match)
    _FRAGMENT_BY_LONGER = {
        "ngã": None,  # remove if not preceded by disease name
    }

    filtered = []
    for e in deduped:
        txt = e['text'].lower().strip()
        etype = e['type']

        # Remove known FP fragments
        if txt in _FP_FRAGMENTS:
            # Check if this is part of a larger known phrase in the text
            # by looking for longer symptom/diagnosis matches nearby
            start, end = e['position']
            # Check 20 chars before and after
            ctx = text[max(0,start-20):end+20].lower()
            longer_found = False
            for sym in SYMPTOMS:
                if len(sym) > len(txt) + 3 and sym.lower() in ctx:
                    longer_found = True
                    break
            for diag in DIAGNOSES:
                if len(diag) > len(txt) + 3 and diag.lower() in ctx:
                    longer_found = True
                    break
            if not longer_found:
                continue  # Skip this FP

        # Remove very short symptoms that are likely noise
        if etype == 'TRIỆU_CHỨNG' and len(txt) < 4:
            continue

        filtered.append(e)

    return [{
        'text': e['text'],
        'type': e['type'],
        'candidates': e.get('candidates', []),
        'assertions': e.get('assertions', []),
        'position': e['position'],
    } for e in filtered]


def process_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    entities = extract_entities_v20(text)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    return entities


def run_batch(input_dir: str, output_dir: str, zip_path: str, total: int = 100):
    os.makedirs(output_dir, exist_ok=True)
    total_entities = 0
    type_counts = {'THUỐC': 0, 'TRIỆU_CHỨNG': 0, 'CHẨN_ĐOÁN': 0}
    for i in range(1, total + 1):
        input_file = os.path.join(input_dir, f'{i}.txt')
        output_file = os.path.join(output_dir, f'{i}.json')
        if os.path.exists(input_file):
            entities = process_file(input_file, output_file)
            total_entities += len(entities)
            for e in entities:
                type_counts[e['type']] = type_counts.get(e['type'], 0) + 1
    print(f'v20 Total: {total_entities}, Types: {type_counts}')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, total + 1):
            json_file = os.path.join(output_dir, f'{i}.json')
            if os.path.exists(json_file):
                zf.write(json_file, f'output/{i}.json')
    print(f'Created: {zip_path}')


if __name__ == '__main__':
    INPUT_DIR = r'D:\projects\Viettel AI race\input\input'
    OUTPUT_DIR = r'D:\projects\Viettel AI race\output_v20'
    ZIP_PATH = r'D:\projects\Viettel AI race\output_v20.zip'
    print(f'v20: Exact v11 replica (6 steps)')
    run_batch(INPUT_DIR, OUTPUT_DIR, ZIP_PATH)