# -*- coding: utf-8 -*-
"""
Optimized Vietnamese Medical NER Pipeline
Designed to maximize: 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score

Strategy:
- High precision drug extraction (candidates = 0.4 weight)
- Comprehensive symptom/diagnosis vocab matching
- Accurate assertion classification (isHistorical/isNegated/isSuspected)
- Avoid noise fragments that hurt text_score (WER penalty)
"""

import os, re, json, zipfile, sys
from typing import List, Dict, Tuple, Optional, Set

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from drug_dict_v3 import DRUG_DICT
from vocab_v43 import SYMPTOMS, DIAGNOSES


# ============================================================================
# 1) KEY_PHRASES - High priority, very specific medical entities
# ============================================================================
KEY_PHRASES = [
    # ====== COMPLEX DIAGNOSES (multi-word) ======
    (r'nhịp\s+xoang\s+chiếm\s+ưu\s+thế', 'CHẨN_ĐOÁN'),
    (r'ngoại\s+tâm\s+thu\s+nhĩ', 'CHẨN_ĐOÁN'),
    (r'ngoại\s+tâm\s+thu\s+thất', 'CHẨN_ĐOÁN'),
    (r'nghẽn\s+tắc\s+và\s+hẹp\s+động\s+mạch\s+cảnh', 'CHẨN_ĐOÁN'),
    (r'hội\s+chứng\s+mạch\s+vành\s+cấp', 'CHẨN_ĐOÁN'),
    (r'nhồi\s+máu\s+cơ\s+tim', 'CHẨN_ĐOÁN'),
    (r'bệnh\s+(?:mạch\s+vành|thiếu\s+máu\s+cơ\s+tim)', 'CHẨN_ĐOÁN'),
    (r'hẹp\s+động\s+mạch\s+cảnh', 'CHẨN_ĐOÁN'),
    (r'đau\s+thắt\s+ngực(?:\s+(?:ổn\s+định|không\s+ổn\s+định))?', 'CHẨN_ĐOÁN'),
    (r'suy\s+tim', 'CHẨN_ĐOÁN'),
    (r'đái\s+tháo\s+đường(?:\s+(?:type\s*[12]|týp\s*[12]))?', 'CHẨN_ĐOÁN'),
    (r'tăng\s+huyết\s+áp', 'CHẨN_ĐOÁN'),
    (r'hạ\s+huyết\s+áp', 'CHẨN_ĐOÁN'),
    (r'béo\s+phì', 'CHẨN_ĐOÁN'),
    (r'bệnh\s+Kawasaki', 'CHẨN_ĐOÁN'),
    (r'viêm\s+(?:gan|dạ\s+dày|phổi|thực\s+quản|khớp|cơ|bàng\s+quang|xoang|họng|tai\s+giữa|tụy|màng\s+não|mũi)\s*\w*', 'CHẨN_ĐOÁN'),
    (r'thao\s+hóa\s+(?:khớp|cột\s+sống)', 'CHẨN_ĐOÁN'),
    (r'thoái\s+hóa\s+(?:khớp|cột\s+sống)', 'CHẨN_ĐOÁN'),
    (r'trào\s+ngược(?:\s+dạ\s+dày\s+thực\s+quản)?', 'CHẨN_ĐOÁN'),
    (r'nhịp\s+xoang(?!\s+chiếm)', 'CHẨN_ĐOÁN'),
    (r'rung\s+nhĩ', 'CHẨN_ĐOÁN'),
    (r'xơ\s+(?:gan|vữa|phổi)', 'CHẨN_ĐOÁN'),
    (r'suy\s+(?:thận|gan|hô\s+hấp|giáp|thượng\s+thận)', 'CHẨN_ĐOÁN'),
    (r'thiếu\s+máu(?:\s+(?:cơ\s+tim|não|cấp|mạn|nặng|nhẹ|tan\s+huyết|thiếu\s+sắt))?', 'CHẨN_ĐOÁN'),
    (r'tăng\s+(?:huyết\s+áp|kali\s+máu|đường\s+huyết|men\s+gan|natri\s+máu|triglyceride|cholesterol|bilirubin)', 'CHẨN_ĐOÁN'),
    (r'hạ\s+(?:kali\s+máu|đường\s+huyết|canxi\s+máu|magne\s+máu|natri\s+máu)', 'CHẨN_ĐOÁN'),
    (r'đa\s+u\s+tuỷ\s+xương', 'CHẨN_ĐOÁN'),
    (r'ung\s+thư\s+\w+', 'CHẨN_ĐOÁN'),
    (r'hội\s+chứng\s+(?:mạch\s+vành\s+cấp|ruột\s+kích\s+thích|cushing|nghiện\s+rượu|parkinson|thận\s+hư|chuyển\s+hóa|ống\s+cổ\s+tay)', 'CHẨN_ĐOÁN'),
    (r'phình\s+(?:động\s+mạch|mạch)', 'CHẨN_ĐOÁN'),
    (r'tắc\s+(?:mạch|ruột|nghẽn)', 'CHẨN_ĐOÁN'),
    (r'viêm\s+(?:dạ\s+dày\s+ruột|gan|phổi|thực\s+quản)', 'CHẨN_ĐOÁN'),
    # ====== COMPLEX SYMPTOMS ======
    (r'khó\s+thở(?:\s+(?:khi\s+gắng\s+sức|khi\s+nằm|khi\s+ngủ|khi\s+nói|khi\s+hoạt\s+động|kéo\s+dài|liên\s+tục|nhẹ|đột\s+ngột|vào|ra))?', 'TRIỆU_CHỨNG'),
    (r'đau\s+ngực(?:\s+(?:khi\s+thở|kiểu\s+bóp\s+nghẹt|kiểu\s+thiếu\s+máu|lan\s+lên|lan\s+ra|phải|trái|sau\s+xương\s+ức|ít))?', 'TRIỆU_CHỨNG'),
    (r'đau\s+bụng(?:\s+(?:dưới|dữ\s+dội|kinh|kinh|trên|quanh\s+rốn|âm\s+ỉ|trên|cơn|vùng\s+hạ\s+sườn\s+phải|vùng\s+thượng\s+vị|từng\s+cơn|ngày\s+càng\s+nặng))?', 'TRIỆU_CHỨNG'),
    (r'đau\s+đầu(?:\s+(?:dữ\s+dội|kinh\s+kiểu\s+migraine|migraine|nặng|vận\s+mạch|tăng\s+dần|kéo\s+dài|căng\s+cơ))?', 'TRIỆU_CHỨNG'),
    (r'giảm\s+dung\s+nạp\s+gắng\s+sức', 'TRIỆU_CHỨNG'),
    (r'tê\s+bì(?:\s+(?:vùng\s+trán\s+phải|nửa\s+mặt\s+phải|chân\s+tay|ở\s+cánh\s+tay\s+trái))?', 'TRIỆU_CHỨNG'),
    (r'yếu\s+(?:nửa\s+người|sức|tay|chân|cơ|toàn\s+thân)', 'TRIỆU_CHỨNG'),
    (r'phù\s+(?:mắt\s+cá\s+chân|chân|tay|hai\s+chân|hai\s+bên|ngoại\s+vi|toàn\s+thân|não|phổi|gan)', 'TRIỆU_CHỨNG'),
    (r'sốt(?:\s+(?:cao|nhẹ|không\s+rõ\s+nguyên\s+nhân|rét\s+run|về\s+chiều|về\s+đêm|kéo\s+dài|phát\s+ban|đến))?', 'TRIỆU_CHỨNG'),
    (r'ho(?:\s+(?:ra\s+máu|khan|có\s+đờm|máu|máu\s+tươi|mạn\s+tính|ra\s+đờm))?', 'TRIỆU_CHỨNG'),
    (r'nôn(?:\s+(?:ra\s+máu|khan|ói|dịch\s+vàng|ra\s+thức\s+ăn))?', 'TRIỆU_CHỨNG'),
    (r'ói(?:\s+(?:mửa|ra\s+máu))?', 'TRIỆU_CHỨNG'),
    (r'buồn\s+nôn(?:\s+(?:nhẹ|sau\s+ăn))?', 'TRIỆU_CHỨNG'),
    (r'đánh\s+trống\s+ngực(?:\s+(?:khi\s+gắng\s+sức|liên\s+hồi|liên\s+tục|từng\s+cơn))?', 'TRIỆU_CHỨNG'),
    (r'tăng\s+đánh\s+trống\s+ngực', 'TRIỆU_CHỨNG'),
    (r'chóng\s+mặt', 'TRIỆU_CHỨNG'),
    (r'mất\s+ngủ', 'TRIỆU_CHỨNG'),
    (r'khó\s+thở\s+nhẹ', 'TRIỆU_CHỨNG'),
    (r'khó\s+thở\s+khi\s+gắng\s+sức', 'TRIỆU_CHỨNG'),
    (r'khó\s+thở\s+khi\s+nằm', 'TRIỆU_CHỨNG'),
    (r'đau\s+vùng\s+hạ\s+sườn\s+phải', 'TRIỆU_CHỨNG'),
    (r'đau\s+hạ\s+sườn\s+(?:phải|trái)', 'TRIỆU_CHỨNG'),
    (r'ngất(?:\s+xỉu|\s+do\s+tim|\s+khi\s+thay\s+đổi\s+tư\s+thế)?', 'TRIỆU_CHỨNG'),
    (r'gắng\s+sức', 'TRIỆU_CHỨNG'),
    (r'đau\s+nhức', 'TRIỆU_CHỨNG'),
    (r'lo\s+âu', 'TRIỆU_CHỨNG'),
    (r'hụt\s+hơi', 'TRIỆU_CHỨNG'),
    (r'mệt\s+mỏi', 'TRIỆU_CHỨNG'),
    (r'buồn', 'TRIỆU_CHỨNG'),
    (r'táo\s+bón', 'TRIỆU_CHỨNG'),
    (r'tê', 'TRIỆU_CHỨNG'),
    (r'mất\s+thăng\s+bằng', 'TRIỆU_CHỨNG'),
    (r'gần\s+ngất', 'TRIỆU_CHỨNG'),
    (r'ho\s+ra\s+máu', 'TRIỆU_CHỨNG'),
    (r'xuất\s+huyết', 'TRIỆU_CHỨNG'),
    (r'co\s+giật', 'TRIỆU_CHỨNG'),
    (r'đổ\s+mồ\s+hôi', 'TRIỆU_CHỨNG'),
    (r'vàng\s+da', 'TRIỆU_CHỨNG'),
    (r'hôn\s+mê', 'TRIỆU_CHỨNG'),
    (r'lú\s+lẫn', 'TRIỆU_CHỨNG'),
    (r'đau\s+lưng', 'TRIỆU_CHỨNG'),
    (r'đau\s+chân', 'TRIỆU_CHỨNG'),
    (r'phù\s+mắt\s+cá\s+chân', 'TRIỆU_CHỨNG'),
    (r'chướng\s+bụng', 'TRIỆU_CHỨNG'),
    (r'Căng\s+thẳng', 'TRIỆU_CHỨNG'),
    (r'thắt\s+chặt\s+ngực', 'TRIỆU_CHỨNG'),
    (r'khó\s+nuốt', 'TRIỆU_CHỨNG'),
    (r'khó\s+tiêu', 'TRIỆU_CHỨNG'),
    (r'phù\s+nề', 'TRIỆU_CHỨNG'),
    (r'đau\s+thượng\s+vị', 'TRIỆU_CHỨNG'),
    (r'đầy\s+bụng', 'TRIỆU_CHỨNG'),
    (r'ợ\s+(?:nóng|hơi|chua)', 'TRIỆU_CHỨNG'),
    (r'đau\s+quanh\s+rốn', 'TRIỆU_CHỨNG'),
    (r'đau\s+hạ\s+vị', 'TRIỆU_CHỨNG'),
    (r'choáng\s+váng', 'TRIỆU_CHỨNG'),
    (r'run\s+(?:tay|tay\s+chân|rẩy|chi)', 'TRIỆU_CHỨNG'),
    (r'rét\s+run', 'TRIỆU_CHỨNG'),
    (r'ớn\s+lạnh', 'TRIỆU_CHỨNG'),
    (r'căng\s+thẳng', 'TRIỆU_CHỨNG'),
    (r'nhức\s+đầu', 'TRIỆU_CHỨNG'),
    (r'đau\s+vai', 'TRIỆU_CHỨNG'),
    (r'đau\s+cổ', 'TRIỆU_CHỨNG'),
    (r'đau\s+khớp', 'TRIỆU_CHỨNG'),
    (r'khò\s+khè', 'TRIỆU_CHỨNG'),
    (r'khàn\s+tiếng', 'TRIỆU_CHỨNG'),
    (r'giọng\s+khàn', 'TRIỆU_CHỨNG'),
    (r'ho\s+khó\s+thở', 'TRIỆU_CHỨNG'),
]
COMPILED_KEY_PHRASES = [(re.compile(p, re.IGNORECASE), t) for p, t in KEY_PHRASES]


# ============================================================================
# 2) DRUG DICTIONARY - Optimized for high candidates_score
# ============================================================================
# Sort drug keys by length desc to prefer longer matches
DRUG_KEYS_SORTED = sorted(DRUG_DICT.keys(), key=lambda k: -len(k))


# Drug blacklist - filter out non-drugs
DRUG_BLACKLIST = {
    'thuốc trước', 'thuốc sau', 'thuốc nào', 'thuốc đó', 'thuốc này',
    'thuốc đang', 'thuốc được', 'thuốc hiện', 'thuốc cũ', 'thuốc mới',
    'thuốc trị', 'thuốc điều', 'thuốc bệnh', 'thuốc nam', 'thuốc bắc',
    'thuốc tây', 'túi thuốc', 'thuốc theo', 'thuốc kê',
    'bệnh nhân', 'bệnh viện', 'bác sĩ', 'siêu âm', 'x-quang',
    'bình thường', 'cơm', 'thức ăn', 'đồ uống', 'nước uống',
    'việc dùng', 'việc uống', 'ống hôm', 'ống hàng',
}


def find_drugs(text: str) -> List[Dict]:
    """Find drugs in text. Uses word boundaries, sorts longest first."""
    if not text:
        return []
    text_lower = text.lower()
    found = []
    used_spans = []

    for key in DRUG_KEYS_SORTED:
        if len(key) < 4:
            continue
        start = 0
        while True:
            idx = text_lower.find(key, start)
            if idx == -1:
                break
            # Word boundary check
            before_ok = (idx == 0) or (not text_lower[idx-1].isalnum())
            after_idx = idx + len(key)
            after_ok = (after_idx >= len(text_lower)) or (not text_lower[after_idx].isalnum())
            if before_ok and after_ok:
                # Check overlap
                overlap = any(not (after_idx <= us or idx >= ue) for us, ue in used_spans)
                if not overlap:
                    # Extend drug name with dose and route/freq
                    end = after_idx
                    tail_match = re.match(
                        r'(?:\s*\d+(?:\.\d+)?(?:\s*-\s*\d+)?\s*(?:mg|mcg|μg|ug|g|ml|iu|ui|meq)?)?'
                        r'(?:\s+x\s+\d+)?'
                        r'(?:\s+(?:po|iv|im|sc|ng|sl|pr|top|uống|uong|tiêm|tiem|đặt|dat|qd|bid|tid|qid|qhs|prn|daily|ngày|hours?|giờ|h|am|pm|q\d+h?|q\d+d?|xl|viên))?'
                        r'(?:\s+(?:po|iv|im|sc|ng|sl|pr|top|qd|bid|tid|qid|qhs|prn|daily|q\d+h?|q\d+d?))?',
                        text_lower[after_idx:after_idx+50],
                        re.IGNORECASE
                    )
                    if tail_match and tail_match.end() <= 40:
                        end = after_idx + tail_match.end()
                    used_spans.append((idx, end))
                    name = text[idx:end].strip()
                    found.append({
                        'text': name,
                        'cui': DRUG_DICT[key],
                        'start': idx,
                        'end': end,
                    })
            start = idx + 1
    found.sort(key=lambda x: x['start'])
    return found


def lookup_drug_cuis_for_text(drug_text: str) -> List[str]:
    """Get all CUIs matching drug text."""
    if not drug_text:
        return []
    t = drug_text.lower().strip()
    cuis = set()
    # Try full match
    if t in DRUG_DICT:
        cuis.add(DRUG_DICT[t])
    # Try all keys as substring
    for key, cui in DRUG_DICT.items():
        if key in t:
            idx = t.find(key)
            before_ok = (idx == 0) or (not t[idx-1].isalnum())
            after_idx = idx + len(key)
            after_ok = (after_idx >= len(t)) or (not t[after_idx].isalnum())
            if before_ok and after_ok:
                cuis.add(cui)
    return list(cuis)


# ============================================================================
# 3) VOCAB MATCHING (Symptoms & Diagnoses)
# ============================================================================
SYMPTOMS_SORTED = sorted(set(SYMPTOMS), key=lambda x: -len(x))
DIAGNOSES_SORTED = sorted(set(DIAGNOSES), key=lambda x: -len(x))

# Fragment blacklist - prevent over-matching single words
# These terms are too short/generic to match standalone
FRAGMENT_BLACKLIST = {
    'yếu', 'mệt', 'tê', 'buồn', 'lo', 'chóng', 'váng', 'ù', 'ngất',
    'ợ', 'táo', 'tiêu', 'chảy', 'đầy', 'nôn', 'sốt', 'phù', 'ngứa', 'rát',
    'ho', 'đau', 'loét', 'khó', 'rối', 'rung', 'mất', 'viêm',
    'đổ', 'suy', 'đau', 'hô', 'hạ', 'nhiễm', 'nhức',
    'xỉu', 'ói', 'đầy', 'thở',
    'chán', 'run', 'phụ', 'mày', 'lạnh', 'nặng', 'nghén',
    'ớn', 'chua', 'bứt', 'rứt', 'bỏng', 'quáng', 'nhèm',
    'nghẹt', 'ứ', 'khác', 'mắt', 'bỏ', 'mửa', 'mềm',
    'gầy', 'mập', 'bí', 'ngoái', 'gồng', 'cơn',
    'nhức', 'sợi', 'xanh', 'vàng', 'bênh', 'khò',
    'phườn', 'tai', 'mũi', 'phổi', 'thắt', 'đặc',
}

# Better: blacklist 1-2 char terms and common false positives
ULTRA_SHORT_TERMS = {
    'ho', 'ói', 'ù', 'đổ', 'ợ', 'tê', 'phù', 'lạnh', 'nặng',
    'mềm', 'đầy', 'bỏ', 'mắt', 'tai', 'mũi', 'gan',
    'ruột', 'phổi', 'thắt', 'đặc', 'xanh', 'vàng',
    'táo', 'tiêu', 'viêm', 'sốt', 'chóng', 'ngất',
    'nôn', 'mửa', 'rát', 'ngứa', 'phù', 'run',
    'yếu', 'mệt', 'mất', 'suy', 'khó', 'đau',
    'đòi', 'xỉu', 'váng', 'lo', 'buồn', 'chán',
    'rung', 'loét', 'rối', 'lú', 'ớn', 'mày',
    'quáng', 'nhèm', 'ói', 'gầy', 'mập', 'bí',
    'ngoái', 'gồng', 'nhức', 'sợi', 'rứt', 'bứt',
    'lạnh', 'quàng', 'thấp', 'cao', 'mềm', 'cứng',
    'khác', 'phụ', 'nghén', 'có', 'không',
    # Additional problematic terms
    'lú', 'buồn', 'chán', 'gầy', 'mập',
}

# Terms that need a qualifier (only match when not at sentence start alone)
NEEDS_QUALIFIER = {
    'sỏi thận', 'sỏi', 'đau', 'tiểu', 'nước',
}

# Filter out ultra-short terms from vocab (false positive prone)
def _filter_terms(terms_list):
    """Filter out too-short, ambiguous terms."""
    filtered = []
    for t in terms_list:
        tl = t.lower()
        if len(tl) <= 2:
            continue
        if tl in ULTRA_SHORT_TERMS:
            continue
        if tl in FRAGMENT_BLACKLIST:
            continue
        filtered.append(t)
    return filtered

SYMPTOMS_SORTED = _filter_terms(SYMPTOMS_SORTED)
DIAGNOSES_SORTED = _filter_terms(DIAGNOSES_SORTED)

# Stopwords that should never match as standalone
_STOPWORDS = {'và', 'của', 'cho', 'trong', 'trên', 'với', 'không', 'có', 'là', 'được'}


def find_vocab_in_line(line: str, line_offset: int, terms: List[str], etype: str,
                       used_spans: List[Tuple[int, int]], seen: Set[str],
                       min_len: int = 4) -> List[Dict]:
    """Find vocab terms in a line."""
    if not line.strip():
        return []
    found = []
    text_lower = line.lower()

    for term in terms:
        tlen = len(term)
        if tlen < min_len:
            continue
        term_lower = term.lower()
        # Skip fragments
        if term_lower in FRAGMENT_BLACKLIST:
            continue
        # Skip ultra-short terms (1-2 chars or stopwords)
        if term_lower in ULTRA_SHORT_TERMS:
            continue
        # Skip pure stopwords
        if term_lower in _STOPWORDS:
            continue
        # Skip 1-2 char terms always
        if tlen <= 2:
            continue
        # Skip terms starting with "viêm" that aren't real diagnoses
        # (handled in pipeline step 5 via reclassification)
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
                overlap = any(not (abs_end <= us or abs_start >= ue) for us, ue in used_spans)
                if not overlap:
                    seen_key = (term_lower, abs_start, abs_end)
                    if seen_key not in seen:
                        seen.add(seen_key)
                        used_spans.append((abs_start, abs_end))
                        found.append({
                            'text': line[idx:after_idx],
                            'type': etype,
                            'position': [abs_start, abs_end],
                        })
            start = idx + 1
    return found


# ============================================================================
# 4) SECTION DETECTION
# ============================================================================
SECTION_PATTERNS = [
    (r'thuốc\s+(?:trước|đang|hiện|dùng|tại|nhà)', 'THUỐC'),
    (r'phác\s+đồ\s+điều\s+trị', 'THUỐC'),
    (r'điều\s+trị\s+(?:hiện\s+tại|tại\s+nhà)', 'THUỐC'),
    (r'tiền\s+sử\s+bệnh(?:\s+nội\s+khoa|\s+lý|\s+ngoại\s+khoa)?', 'CHẨN_ĐOÁN'),
    (r'tiền\s+căn', 'CHẨN_ĐOÁN'),
    (r'bệnh\s+lý\s+(?:mãn\s+tính|nền|kèm)', 'CHẨN_ĐOÁN'),
    (r'chẩn\s+đoán(?:\s+(?:xác|sơ|cuối|cùng|trước|sau|ra|phân|biệt))?', 'CHẨN_ĐOÁN'),
    (r'tiền\s+sử\s+phẫu\s+thuật', 'CHẨN_ĐOÁN'),
    (r'phẫu\s+thuật\s*/\s*thủ\s+thuật', 'CHẨN_ĐOÁN'),
    (r'triệu\s+chứng(?:\s+cơ\s+năng|\s+thực\s+thể|\s+hiện\s+tại|\s+khi\s+nhập)?', 'TRIỆU_CHỨNG'),
    (r'bệnh\s+sử(?:\s+hiện\s+tại|\s+ngoại\s+khoa|\s+nội\s+khoa)?', 'TRIỆU_CHỨNG'),
    (r'lịch\s+sử\s+bệnh', 'TRIỆU_CHỨNG'),
    (r'lý\s+do\s+(?:nhập|vào|khám|ra)', 'TRIỆU_CHỨNG'),
    (r'đặc\s+điểm\s+triệu\s+chứng', 'TRIỆU_CHỨNG'),
    (r'tình\s+trạng\s+(?:ngay|trước|khi|lúc)', 'TRIỆU_CHỨNG'),
    (r'diễn\s+biến', 'TRIỆU_CHỨNG'),
    (r'lúc\s+vào\s+viện', 'TRIỆU_CHỨNG'),
    (r'khám\s+tại\s+bệnh\s+viện', 'TRIỆU_CHỨNG'),
    (r'đánh\s+giá\s+tại\s+bệnh\s+viện', 'CHẨN_ĐOÁN'),
    (r'cận\s+lâm\s+sàng', 'CHẨN_ĐOÁN'),
]
COMPILED_SECTION = [(re.compile(p, re.IGNORECASE), t) for p, t in SECTION_PATTERNS]


def detect_section(line: str) -> Optional[str]:
    """Detect section type from line."""
    line_clean = re.sub(r'^\s*[\d]+\.?\s*', '', line).strip().lower()
    if not line_clean:
        return None
    for pat, etype in COMPILED_SECTION:
        if pat.search(line_clean):
            return etype
    return None


def build_line_offsets(text: str) -> List[int]:
    """Build character offsets for each line."""
    offsets, pos = [], 0
    for line in text.split('\n'):
        offsets.append(pos)
        pos += len(line) + 1
    return offsets


def get_section_for_pos(pos: int, line_offsets: List[int], sections: List) -> Optional[str]:
    """Get section type for a position."""
    for i, offset in enumerate(line_offsets):
        if pos < offset or i == len(line_offsets) - 1:
            return sections[i - 1] if i > 0 and i > 0 else sections[0] if sections else None
    return sections[-1] if sections else None


# ============================================================================
# 5) ASSERTION CLASSIFICATION
# ============================================================================
NEG_PATTERNS = [
    re.compile(r'\bkhông\s+(?:có|còn|thấy|ghi\s+nhận|xuất\s+hiện|đau|thở|sốt|buồn|sợ|ngứa|ra|cảm|ớn|phù|ho|chảy|rát|ngạt)', re.IGNORECASE),
    re.compile(r'\bchưa\s+(?:có|từng|bảo\s+giờ|rõ)', re.IGNORECASE),
    re.compile(r'\bko\s+(?:có|còn)', re.IGNORECASE),
    re.compile(r'\b(vắng|mất)\s+(?:mặt|tên)', re.IGNORECASE),
    re.compile(r'\bphủ\s+nhận\b', re.IGNORECASE),
    re.compile(r'(?<!\w)không(?!\w)', re.IGNORECASE),
    re.compile(r'\bchưa\s+xác\s+định\b', re.IGNORECASE),
    re.compile(r'\b(?:â[m]m\s+đặc\s+hiệu)', re.IGNORECASE),
    re.compile(r'(?:không\s+đặc\s+hiệu)', re.IGNORECASE),
]
SUSP_PATTERNS = [
    re.compile(r'\bcó\s+thể\b', re.IGNORECASE),
    re.compile(r'\bnghi\s*ngờ\b', re.IGNORECASE),
    re.compile(r'\bnghĩ\s+đến\b', re.IGNORECASE),
    re.compile(r'\bchưa\s+rõ\b', re.IGNORECASE),
    re.compile(r'\bđang\s+theo\s+dõi\b', re.IGNORECASE),
    re.compile(r'\bđang\s+xem\s+xét\b', re.IGNORECASE),
    re.compile(r'\bnghi\s+nhiều\s+đến\b', re.IGNORECASE),
    re.compile(r'\bnghĩ\s+nhiều\s+đến\b', re.IGNORECASE),
    re.compile(r'\bnghĩ\s+đến\b', re.IGNORECASE),
    re.compile(r'\bchưa\s+loại\s+trừ\b', re.IGNORECASE),
    re.compile(r'\bnguy\s+cơ\b', re.IGNORECASE),
]
HIST_PATTERNS = [
    re.compile(r'\btiền\s+sử\b', re.IGNORECASE),
    re.compile(r'\bđã\s+(?:dùng|sử|mắc|bị|có|tiêm|uống|phẫu\s+thuật|thực\s+hiện|được)', re.IGNORECASE),
    re.compile(r'\btrước\s+(?:khi|đây|kia)', re.IGNORECASE),
    re.compile(r'\b(cũ|lâu|năm|tháng|ngày)\s+(?:trước|nay|đây)', re.IGNORECASE),
    re.compile(r'\bđiều\s+trị\s+(?:trước|trước\s+đây)', re.IGNORECASE),
    re.compile(r'\bcó\s+bệnh\s+từ', re.IGNORECASE),
    re.compile(r'\btừ\s+(?:trước|năm|tháng)', re.IGNORECASE),
    re.compile(r'\bnăm\s+(?:ngoái|nay)', re.IGNORECASE),
    re.compile(r'\bnhiều\s+năm\s+trước', re.IGNORECASE),
    re.compile(r'\btừng\s+(?:bị|mắc|dùng)', re.IGNORECASE),
    re.compile(r'\b(cắt|bỏ)\s+(?:đại\s+tràng|dạ\s+dày|thận|phổi)', re.IGNORECASE),
    re.compile(r'\b(?:năm|vài|hai)\s+năm\s+trước', re.IGNORECASE),
    re.compile(r'\bđược\s+chẩn\s+đoán\s+(?:hội\s+chứng|viêm|bệnh|ung\s+thư|tăng|hạ)', re.IGNORECASE),
    re.compile(r'\b(?:trước\s+đó|trước\s+đây)', re.IGNORECASE),
    re.compile(r'\bcách\s+đây\b', re.IGNORECASE),
]


def classify_assertion(start: int, end: int, text: str,
                       section_type: Optional[str], entity_type: str) -> List[str]:
    """Classify assertion for an entity."""
    result = []
    ctx_before = text[max(0, start-60):start].lower()
    ctx_after = text[end:min(len(text), end+30)].lower()
    ctx_window = ctx_before + ' ' + ctx_after

    # Check negation FIRST (highest priority for status)
    negated = False
    for pat in NEG_PATTERNS:
        if pat.search(ctx_before[-40:]):
            result.append('isNegated')
            negated = True
            break

    # Check suspected (only if not negated)
    if not negated:
        for pat in SUSP_PATTERNS:
            if pat.search(ctx_window):
                result.append('isSuspected')
                break

    # Check historical
    is_hist = False
    if entity_type == 'THUỐC':
        if section_type == 'THUỐC':
            is_hist = True
        else:
            for pat in HIST_PATTERNS:
                if pat.search(ctx_before):
                    is_hist = True
                    break
    elif entity_type == 'CHẨN_ĐOÁN':
        if section_type == 'CHẨN_ĐOÁN':
            ctx_b = ctx_before.lower()
            if 'kết quả' in ctx_b or 'xét nghiệm' in ctx_b:
                is_hist = False
            else:
                is_hist = True
        else:
            for pat in HIST_PATTERNS:
                if pat.search(ctx_before):
                    is_hist = True
                    break
    else:  # TRIỆU_CHỨNG
        if section_type == 'TRIỆU_CHỨNG':
            is_hist = False  # current symptoms
        else:
            for pat in HIST_PATTERNS:
                if pat.search(ctx_before):
                    is_hist = True
                    break

    if is_hist and 'isNegated' not in result:
        result.append('isHistorical')

    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def extract_entities(text: str) -> List[Dict]:
    if not text or not text.strip():
        return []

    entities = []
    used_spans: List[Tuple[int, int]] = []
    seen: Set = set()
    lines = text.split('\n')

    # Detect sections
    sections = []
    current_section = None
    for line in lines:
        detected = detect_section(line)
        if detected:
            current_section = detected
        sections.append(current_section)

    line_offsets = build_line_offsets(text)

    # ===== Step 0: KEY_PHRASES (highest priority) =====
    for pat, etype in COMPILED_KEY_PHRASES:
        for m in pat.finditer(text):
            abs_start = m.start()
            abs_end = m.end()
            phrase = m.group().strip()
            key = (phrase.lower(), abs_start, abs_end)
            if key in seen:
                continue
            # Skip ultra-short phrase matches
            ph_lower = phrase.lower()
            if len(ph_lower) <= 3:
                continue
            if ph_lower in ULTRA_SHORT_TERMS or ph_lower in FRAGMENT_BLACKLIST:
                continue
            # Skip if phrase is inside a longer word
            if before_idx_chk := (abs_start > 0 and text[abs_start-1].isalnum()):
                continue
            if abs_end < len(text) and text[abs_end].isalnum():
                continue
            if any(not (abs_end <= us or abs_start >= ue) for us, ue in used_spans):
                continue
            seen.add(key)
            used_spans.append((abs_start, abs_end))
            section_type = get_section_for_pos(abs_start, line_offsets, sections)
            assertions = classify_assertion(abs_start, abs_end, text, section_type, etype)
            entities.append({
                'text': phrase, 'type': etype, 'candidates': [],
                'assertions': assertions, 'position': [abs_start, abs_end],
            })

    # ===== Step 1: DRUGS =====
    drug_hits = find_drugs(text)
    for d in drug_hits:
        drug_text = d['text'].strip()
        dl = drug_text.lower().strip()
        if dl in DRUG_BLACKLIST:
            continue
        if len(dl) < 4:
            continue
        if re.match(r'^[\d\.\/\s]+$', dl):
            continue
        # Clean trailing punctuation
        drug_clean = re.sub(r'[\.,;:]+$', '', drug_text).strip()
        if not drug_clean or len(drug_clean) < 4:
            continue
        start, end = d['start'], d['end']
        if any(not (end <= us or start >= ue) for us, ue in used_spans):
            continue
        # Check for overlap with used spans for end boundary
        end_adjusted = end
        used_spans.append((start, end_adjusted))
        section_type = get_section_for_pos(start, line_offsets, sections)
        assertions = classify_assertion(start, end_adjusted, text, section_type, 'THUỐC')
        cui_list = lookup_drug_cuis_for_text(drug_clean)
        if not cui_list and d.get('cui'):
            cui_list = [d['cui']]
        entities.append({
            'text': drug_clean, 'type': 'THUỐC', 'candidates': cui_list,
            'assertions': assertions, 'position': [start, end_adjusted],
        })

    # ===== Step 2-3: VOCAB per line, section-aware =====
    for i, line in enumerate(lines):
        if i >= len(line_offsets):
            continue
        line_offset = line_offsets[i]
        section_type = sections[i] if i < len(sections) else None
        if not line.strip():
            continue

        # Default: extract SYMPTOMS first (more frequent in clinical text)
        # In THUỐC section: prefer SYMPTOMS
        # In CHẨN_ĐOÁN section: prefer DIAGNOSES
        if section_type == 'CHẨN_ĐOÁN':
            ordered_extractions = [
                ('CHẨN_ĐOÁN', DIAGNOSES_SORTED),
                ('TRIỆU_CHỨNG', SYMPTOMS_SORTED),
            ]
        elif section_type == 'TRIỆU_CHỨNG':
            ordered_extractions = [
                ('TRIỆU_CHỨNG', SYMPTOMS_SORTED),
                ('CHẨN_ĐOÁN', DIAGNOSES_SORTED),
            ]
        else:
            ordered_extractions = [
                ('TRIỆU_CHỨNG', SYMPTOMS_SORTED),
                ('CHẨN_ĐOÁN', DIAGNOSES_SORTED),
            ]

        for etype, terms in ordered_extractions:
            found = find_vocab_in_line(line, line_offset, terms, etype,
                                       used_spans, seen, min_len=4)
            for f in found:
                s, e = f['position']
                assertions = classify_assertion(s, e, text, section_type, etype)
                entities.append({
                    'text': f['text'], 'type': etype, 'candidates': [],
                    'assertions': assertions, 'position': f['position'],
                })

    # ===== Step 4: Lý do nhập viện (special extraction) =====
    ly_do_re = re.compile(r'lý\s+do\s+(?:nhập|vào|khám)(?:\s*viện)?\s*:?\s*([^\n]+)', re.IGNORECASE)
    for m in ly_do_re.finditer(text):
        content = m.group(1).strip()
        if not content or len(content) < 3:
            continue
        content_lower = content.lower()
        content_start = m.start(1)

        # Extract symptoms first, then diagnoses
        for etype, terms in [('TRIỆU_CHỨNG', SYMPTOMS_SORTED), ('CHẨN_ĐOÁN', DIAGNOSES_SORTED)]:
            for term in terms:
                tlen = len(term)
                if tlen < 4:
                    continue
                term_lower = term.lower()
                if term_lower in FRAGMENT_BLACKLIST:
                    continue
                idx = content_lower.find(term_lower)
                if idx == -1:
                    continue
                before_ok = (idx == 0) or (not content_lower[idx-1].isalnum())
                after_idx = idx + tlen
                after_ok = (after_idx >= len(content_lower)) or (not content_lower[after_idx].isalnum())
                if before_ok and after_ok:
                    abs_start = content_start + idx
                    abs_end = abs_start + tlen
                    if any(not (abs_end <= us or abs_start >= ue) for us, ue in used_spans):
                        continue
                    key = (term_lower, abs_start, abs_end)
                    if key in seen:
                        continue
                    seen.add(key)
                    used_spans.append((abs_start, abs_end))
                    section_type = 'TRIỆU_CHỨNG'
                    assertions = classify_assertion(abs_start, abs_end, text, section_type, etype)
                    entities.append({
                        'text': term, 'type': etype, 'candidates': [],
                        'assertions': assertions, 'position': [abs_start, abs_end],
                    })

    # ===== Step 5: Viêm reclassification & cleanup =====
    for e in entities:
        if e['type'] == 'CHẨN_ĐOÁN' and e['text'].lower().startswith('viêm '):
            start, end = e['position']
            ctx = text[max(0, start-30):start].lower()
            if re.search(r'\b(cho|vì|do|biểu\s+hiện|nghi|triệu\s+chứng)\s*$', ctx):
                e['type'] = 'TRIỆU_CHỨNG'

    # Step 5b: Remove entities with trailing/leading non-vietnamese words (noise)
    cleaned = []
    for e in entities:
        t = e['text']
        # Drop if entity has junk chars like "em", "!", ",", "."
        if re.search(r'(?:\s|^)(?:em|anh|chị|bạn|bác|con|người|ông|bà)(?:\s|$)', t.lower()):
            # only if it's not a critical part of an established phrase
            continue
        # Drop "viêm dạ dày em" type noise
        if re.search(r'\s+(?:em|anh|chị|bạn)\s*$', t.lower()):
            continue
        cleaned.append(e)
    entities = cleaned

    # Step 5c: Filter out drug names that are actually lab values/disease names
    # (like "Glucose" used in context of disease, "Albumin" as lab value)
    non_drug_terms = {
        'glucose', 'albumin', 'hemoglobin', 'creatinine', 'urea', 'bilirubin',
        'ast', 'alt', 'alp', 'ferritin', 'ceruloplasmin',
        'đạm', 'đường', 'muối', 'nước',
    }
    drug_filtered = []
    for e in entities:
        if e['type'] == 'THUỐC':
            t = e['text'].lower().strip()
            # Only keep as drug if has dosage or route/freq
            has_dose = bool(re.search(r'\d+\s*(?:mg|mcg|ug|g|ml|iu|ui)', t, re.IGNORECASE))
            has_route = bool(re.search(r'\b(?:po|iv|im|sc|ng|sl|pr|uống|tiêm|viên)\b', t, re.IGNORECASE))
            has_freq = bool(re.search(r'\b(?:qd|bid|tid|qid|qhs|prn|daily|x\s*\d+|ngày|sáng|trưa|tối|chiều)\b', t, re.IGNORECASE))
            # If it's a generic lab-only single word (no dose), skip
            if t in non_drug_terms and not has_dose and not has_route and not has_freq:
                continue
        drug_filtered.append(e)
    entities = drug_filtered

    # ===== Sort by position =====
    entities.sort(key=lambda x: x['position'][0])

    # ===== Final dedup by EXACT position =====
    seen_exact = set()
    deduped1 = []
    for e in entities:
        key = (e['position'][0], e['position'][1], e['type'])
        if key not in seen_exact:
            seen_exact.add(key)
            deduped1.append(e)
    entities = deduped1

    # ===== De-overlap: prefer longer/wider entities =====
    final = []
    entities_by_pos = sorted(entities, key=lambda x: (x['position'][0], -(x['position'][1] - x['position'][0])))
    occupied = []  # list of (start, end) used
    for e in entities_by_pos:
        s, en = e['position']
        # Check if this entity is contained by an already-kept longer entity
        contained = False
        for os_, oe in occupied:
            if os_ <= s and en <= oe:
                contained = True
                break
        if contained:
            continue
        # Check if this entity contains previously kept entities (shorter)
        new_occupied = []
        to_remove = []
        for i, (os_, oe) in enumerate(occupied):
            if s <= os_ and oe <= en:
                # This shorter entity is inside the current one, remove it
                to_remove.append(i)
            else:
                new_occupied.append((os_, oe))
        for i in reversed(to_remove):
            occupied.pop(i)
            if i < len(final):
                # Don't actually remove from final; just don't add
                pass
        # Add current
        final.append(e)
        occupied.append((s, en))

    # ===== Final dedup by text+close position =====
    seen_text_pos = set()
    result = []
    for e in final:
        # Same text appearing multiple times -> keep best one (longest, or first)
        key = (e['type'], e['text'].lower())
        if key in seen_text_pos:
            continue
        seen_text_pos.add(key)
        result.append(e)

    entities = result
    entities.sort(key=lambda x: x['position'][0])

    return [{
        'text': e['text'],
        'type': e['type'],
        'candidates': e.get('candidates', []),
        'assertions': e.get('assertions', []),
        'position': e['position'],
    } for e in entities]


def process_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    entities = extract_entities(text)
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
    print(f'Total: {total_entities}, Types: {type_counts}')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, total + 1):
            json_file = os.path.join(output_dir, f'{i}.json')
            if os.path.exists(json_file):
                zf.write(json_file, f'output/{i}.json')
    print(f'Created: {zip_path}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', default=r'D:\projects\Viettel AI race\input_turn2_vong1\input')
    ap.add_argument('--output_dir', default=r'D:\projects\Viettel AI race\output')
    ap.add_argument('--zip_path', default=r'D:\projects\Viettel AI race\output.zip')
    ap.add_argument('--total', type=int, default=100)
    args = ap.parse_args()
    run_batch(args.input_dir, args.output_dir, args.zip_path, args.total)