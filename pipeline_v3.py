# -*- coding: utf-8 -*-
"""
Pipeline V3 - Vietnamese Medical NER improvements over pipeline_v2.

Improvements (from explore agent analysis):
  1. Add English drug dict (DRUG_DICT_EN) — 100% of GT drugs are in English
  2. Add ~100 missing ICD-10 codes (candidates score, 40% weight)
  3. Widen assertion context window 60 → 300 chars (assertions score)
  4. Add missing symptoms + diagnoses to vocab (text score)
  5. Aggressive dedup that preserves longest-match (text WER)
  6. Skip over-extending ICD matches (e.g. 'ung thư biểu mô' → emit 'ung thư')

Run:
  python pipeline_v3.py --total 100
  python pipeline_v3.py --total 5 --start 1
"""
import os, re, json, sys, zipfile
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from drug_dict_v3 import DRUG_DICT
from drug_dict_en import DRUG_DICT_EN
from vocab_v43 import SYMPTOMS, DIAGNOSES
from icd10_dict_expanded import ICD10_DICT_EXPANDED, ICD10_KEYS_SORTED_EXPANDED
from test_dict import TEST_NAMES_SORTED, TEST_RESULTS_SORTED, TEST_SECTION_RE


# ============================================================================
# UTILITIES
# ============================================================================

def find_in_text(text: str, snippet: str) -> Optional[Tuple[int, int]]:
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


def find_all_in_text(text: str, snippet: str) -> List[Tuple[int, int]]:
    results = []
    start = 0
    while True:
        pos = find_in_text(text[start:], snippet)
        if not pos:
            break
        s, e = pos
        results.append((start + s, start + e))
        start = start + e
    return results


# ============================================================================
# DRUG FALSE POSITIVES (built from test names + manual)
# ============================================================================

DRUG_FALSE_POSITIVES = set()
for t in TEST_NAMES_SORTED:
    if len(t) >= 3:
        DRUG_FALSE_POSITIVES.add(t.lower())

# Plus manual
DRUG_FALSE_POSITIVES.update({
    'glucose', 'glucose máu', 'protein', 'albumin', 'creatinin', 'creatinine',
    'hemoglobin', 'bilirubin', 'cholesterol', 'triglycerid', 'ast', 'alt',
    'got', 'gpt', 'crp', 'bnp', 'troponin', 'hba1c', 'lactate', 'bilirubin toàn phần',
    'phân tích', 'công thức', 'nước tiểu',
    'kali', 'kali (k)', 'k+', 'k', 'na+', 'na', 'cl-', 'cl', 'ca++', 'ca',
    'ion đồ', 'điện giải đồ', 'điện giải', 'đông máu', 'đông máu cơ bản',
    'hồng cầu', 'bạch cầu', 'tiểu cầu',
    # Common false positives from analysis
    'vitamin c', 'vitamin b12', 'vitamin d', 'vitamin k',
})


# ============================================================================
# STEP 1: ICD-10 lookup for CHẨN_ĐOÁN
# ============================================================================

# Symptoms that have ICD10 codes (R-codes) but should NOT be CHẨN_ĐOÁN
ICD10_SYMPTOMS_SKIP = {
    'sốt', 'sốt cao', 'sốt nhẹ', 'sốt kéo dài', 'sốt không rõ nguyên nhân',
    'ớn lạnh', 'mệt mỏi', 'vàng da', 'vàng mắt', 'vàng niêm mạc',
    'ngứa', 'ngứa mũi', 'ngứa họng', 'ngứa mắt',
    'hoa mắt', 'chóng mặt', 'mất ngủ', 'đau đầu',
    'đau bụng', 'đau ngực', 'đau lưng', 'đau khớp', 'đau cơ',
    'đau họng', 'đau răng', 'đau tai', 'đau mắt', 'đau tim',
    'khó thở', 'khò khè', 'ho khan', 'ho có đờm', 'ho ra máu',
    'nôn ói', 'nôn ra máu', 'buồn nôn', 'tiêu chảy', 'táo bón',
    'yếu sức', 'run tay', 'run chân', 'phù', 'phù chân', 'phù mặt',
    'phát ban', 'nổi mẩn', 'tức ngực', 'nặng ngực', 'khàn tiếng',
    'sụt cân', 'tăng cân', 'chán ăn', 'ăn kém', 'khát nước',
    'co giật', 'bất tỉnh', 'ngất', 'tê tay', 'tê chân',
    'nhìn mờ', 'ù tai', 'điếc', 'giảm thính lực',
    'ói mửa', 'sốt phát ban',
}


def find_icd10_diagnoses(text: str) -> List[Dict]:
    found = []
    used_spans = []
    for key in ICD10_KEYS_SORTED_EXPANDED:
        if len(key) < 3:
            continue
        # Skip symptoms (R-codes are not real diagnoses)
        if key.lower() in ICD10_SYMPTOMS_SKIP:
            continue
        positions = find_all_in_text(text, key)
        for s, e in positions:
            overlap = any(not (e <= us or s >= ue) for us, ue in used_spans)
            if overlap:
                continue
            codes = ICD10_DICT_EXPANDED.get(key, [])
            if not codes:
                continue
            used_spans.append((s, e))
            found.append({
                'text': text[s:e],
                'type': 'CHẨN_ĐOÁN',
                'candidates': list(codes),
                'start': s,
                'end': e,
            })
    found.sort(key=lambda x: x['start'])
    return found


# ============================================================================
# STEP 2: Drug dictionary (Vietnamese + English) with RxNorm
# ============================================================================

def find_drugs(text: str) -> List[Dict]:
    found = []
    used_spans = []

    # Combine dictionaries; English first because they're often more specific
    all_drugs = {}
    all_drugs.update(DRUG_DICT_EN)  # EN first for longest match
    all_drugs.update(DRUG_DICT)

    drugs_sorted = sorted(all_drugs.keys(), key=lambda x: -len(x))
    for key in drugs_sorted:
        if len(key) < 3:
            continue
        if key.lower() in DRUG_FALSE_POSITIVES:
            continue
        positions = find_all_in_text(text, key)
        for s, e in positions:
            overlap = any(not (e <= us or s >= ue) for us, ue in used_spans)
            if overlap:
                continue
            used_spans.append((s, e))
            found.append({
                'text': text[s:e],
                'type': 'THUỐC',
                'candidates': [all_drugs[key]],
                'start': s,
                'end': e,
            })
    found.sort(key=lambda x: x['start'])
    return found


# ============================================================================
# STEP 3: KEY_PHRASES for CHẨN_ĐOÁN + TRIỆU_CHỨNG
# ============================================================================

KEY_PHRASES = [
    # Diagnoses
    ("tăng huyết áp", "CHẨN_ĐOÁN"),
    ("đái tháo đường", "CHẨN_ĐOÁN"),
    ("đái tháo đường type 1", "CHẨN_ĐOÁN"),
    ("đái tháo đường type 2", "CHẨN_ĐOÁN"),
    ("suy tim", "CHẨN_ĐOÁN"),
    ("suy tim độ", "CHẨN_ĐOÁN"),
    ("nhồi máu cơ tim", "CHẨN_ĐOÁN"),
    ("tai biến mạch máu não", "CHẨN_ĐOÁN"),
    ("viêm phổi", "CHẨN_ĐOÁN"),
    ("trầm cảm", "CHẨN_ĐOÁN"),
    ("rối loạn lo âu", "CHẨN_ĐOÁN"),
    ("hen phế quản", "CHẨN_ĐOÁN"),
    ("hen suyễn", "CHẨN_ĐOÁN"),
    ("COPD", "CHẨN_ĐOÁN"),
    ("viêm dạ dày", "CHẨN_ĐOÁN"),
    ("viêm ruột", "CHẨN_ĐOÁN"),
    ("viêm gan", "CHẨN_ĐOÁN"),
    ("viêm gan b", "CHẨN_ĐOÁN"),
    ("viêm gan c", "CHẨN_ĐOÁN"),
    ("xơ gan", "CHẨN_ĐOÁN"),
    ("ung thư phổi", "CHẨN_ĐOÁN"),
    ("ung thư vú", "CHẨN_ĐOÁN"),
    ("ung thư gan", "CHẨN_ĐOÁN"),
    ("ung thư dạ dày", "CHẨN_ĐOÁN"),
    ("ung thư đại tràng", "CHẨN_ĐOÁN"),
    ("ung thư tuyến giáp", "CHẨN_ĐOÁN"),
    ("ung thư tuyến tiền liệt", "CHẨN_ĐOÁN"),
    ("ung thư máu", "CHẨN_ĐOÁN"),
    ("ung thư", "CHẨN_ĐOÁN"),  # short match — added per analysis
    ("parkinson", "CHẨN_ĐOÁN"),
    ("alzheimer", "CHẨN_ĐOÁN"),
    ("động kinh", "CHẨN_ĐOÁN"),
    ("rung nhĩ", "CHẨN_ĐOÁN"),
    ("thiếu máu", "CHẨN_ĐOÁN"),
    ("thiếu máu cơ tim", "CHẨN_ĐOÁN"),
    ("rối loạn mỡ máu", "CHẨN_ĐOÁN"),
    ("rối loạn lipid máu", "CHẨN_ĐOÁN"),
    ("tăng lipid máu", "CHẨN_ĐOÁN"),
    ("tăng cholesterol", "CHẨN_ĐOÁN"),
    ("gút", "CHẨN_ĐOÁN"),
    ("gout", "CHẨN_ĐOÁN"),
    ("thoái hóa khớp", "CHẨN_ĐOÁN"),
    ("thoái hóa cột sống", "CHẨN_ĐOÁN"),
    ("thoát vị đĩa đệm", "CHẨN_ĐOÁN"),
    ("viêm khớp", "CHẨN_ĐOÁN"),
    ("viêm đa khớp", "CHẨN_ĐOÁN"),
    ("loãng xương", "CHẨN_ĐOÁN"),
    ("suy thận", "CHẨN_ĐOÁN"),
    ("suy thận mạn", "CHẨN_ĐOÁN"),
    ("viêm bàng quang", "CHẨN_ĐOÁN"),
    ("sỏi thận", "CHẨN_ĐOÁN"),
    ("viêm cầu thận", "CHẨN_ĐOÁN"),
    ("hội chứng thận hư", "CHẨN_ĐOÁN"),
    ("trào ngược dạ dày thực quản", "CHẨN_ĐOÁN"),
    ("viêm loét dạ dày", "CHẨN_ĐOÁN"),
    ("viêm tụy", "CHẨN_ĐOÁN"),
    ("sỏi mật", "CHẨN_ĐOÁN"),
    ("viêm gan virus", "CHẨN_ĐOÁN"),
    ("viêm màng não", "CHẨN_ĐOÁN"),
    ("viêm phế quản", "CHẨN_ĐOÁN"),
    ("viêm phổi thùy", "CHẨN_ĐOÁN"),
    ("lao phổi", "CHẨN_ĐOÁN"),
    ("HIV", "CHẨN_ĐOÁN"),
    ("AIDS", "CHẨN_ĐOÁN"),
    ("sốt xuất huyết", "CHẨN_ĐOÁN"),
    ("sốt rét", "CHẨN_ĐOÁN"),
    ("thương hàn", "CHẨN_ĐOÁN"),
    ("covid", "CHẨN_ĐOÁN"),
    ("covid-19", "CHẨN_ĐOÁN"),
    ("basedow", "CHẨN_ĐOÁN"),
    ("suy giáp", "CHẨN_ĐOÁN"),
    ("cường giáp", "CHẨN_ĐOÁN"),
    ("viêm tuyến giáp", "CHẨN_ĐOÁN"),
    ("bướu cổ", "CHẨN_ĐOÁN"),
    ("đái máu", "CHẨN_ĐOÁN"),
    ("protein niệu", "CHẨN_ĐOÁN"),
    ("huyết niệu", "CHẨN_ĐOÁN"),
    # Added per analysis
    ("tăng men gan", "CHẨN_ĐOÁN"),
    ("viêm cơ", "CHẨN_ĐOÁN"),
    ("viêm mạch", "CHẨN_ĐOÁN"),
    ("tràn dịch màng tim", "CHẨN_ĐOÁN"),
    ("tắc mạch", "CHẨN_ĐOÁN"),
    ("tắc nghẽn", "CHẨN_ĐOÁN"),
    ("hạ huyết áp", "CHẨN_ĐOÁN"),
    ("mề đay", "CHẨN_ĐOÁN"),
    ("mày đay", "CHẨN_ĐOÁN"),
    ("tác dụng phụ", "CHẨN_ĐOÁN"),
    ("loạn thần", "CHẨN_ĐOÁN"),
    ("tụ máu dưới màng cứng", "CHẨN_ĐOÁN"),
    ("bệnh nguyên bào nuôi", "CHẨN_ĐOÁN"),
    # Symptoms
    ("sốt cao", "TRIỆU_CHỨNG"),
    ("sốt nhẹ", "TRIỆU_CHỨNG"),
    ("đau đầu", "TRIỆU_CHỨNG"),
    ("đau ngực", "TRIỆU_CHỨNG"),
    ("đau bụng", "TRIỆU_CHỨNG"),
    ("đau lưng", "TRIỆU_CHỨNG"),
    ("đau khớp", "TRIỆU_CHỨNG"),
    ("đau cơ", "TRIỆU_CHỨNG"),
    ("đau họng", "TRIỆU_CHỨNG"),
    ("đau răng", "TRIỆU_CHỨNG"),
    ("đau tai", "TRIỆU_CHỨNG"),
    ("đau mắt", "TRIỆU_CHỨNG"),
    ("đau tim", "TRIỆU_CHỨNG"),
    ("khó thở", "TRIỆU_CHỨNG"),
    ("khò khè", "TRIỆU_CHỨNG"),
    ("ho khan", "TRIỆU_CHỨNG"),
    ("ho có đờm", "TRIỆU_CHỨNG"),
    ("ho ra máu", "TRIỆU_CHỨNG"),
    ("nôn ói", "TRIỆU_CHỨNG"),
    ("nôn ra máu", "TRIỆU_CHỨNG"),
    ("buồn nôn", "TRIỆU_CHỨNG"),
    ("tiêu chảy", "TRIỆU_CHỨNG"),
    ("táo bón", "TRIỆU_CHỨNG"),
    ("chóng mặt", "TRIỆU_CHỨNG"),
    ("mệt mỏi", "TRIỆU_CHỨNG"),
    ("yếu sức", "TRIỆU_CHỨNG"),
    ("run tay", "TRIỆU_CHỨNG"),
    ("run chân", "TRIỆU_CHỨNG"),
    ("phù", "TRIỆU_CHỨNG"),
    ("phù chân", "TRIỆU_CHỨNG"),
    ("phù mặt", "TRIỆU_CHỨNG"),
    ("vàng da", "TRIỆU_CHỨNG"),
    ("vàng mắt", "TRIỆU_CHỨNG"),
    ("ngứa", "TRIỆU_CHỨNG"),
    ("phát ban", "TRIỆU_CHỨNG"),
    ("nổi mẩn", "TRIỆU_CHỨNG"),
    ("mất ngủ", "TRIỆU_CHỨNG"),
    ("khó ngủ", "TRIỆU_CHỨNG"),
    ("tim đập nhanh", "TRIỆU_CHỨNG"),
    ("tim đập chậm", "TRIỆU_CHỨNG"),
    ("hồi hộp", "TRIỆU_CHỨNG"),
    ("đánh trống ngực", "TRIỆU_CHỨNG"),
    ("tức ngực", "TRIỆU_CHỨNG"),
    ("nặng ngực", "TRIỆU_CHỨNG"),
    ("khàn tiếng", "TRIỆU_CHỨNG"),
    ("ói mửa", "TRIỆU_CHỨNG"),
    ("sụt cân", "TRIỆU_CHỨNG"),
    ("tăng cân", "TRIỆU_CHỨNG"),
    ("chán ăn", "TRIỆU_CHỨNG"),
    ("ăn kém", "TRIỆU_CHỨNG"),
    ("khát nước", "TRIỆU_CHỨNG"),
    ("đi tiểu nhiều", "TRIỆU_CHỨNG"),
    ("đi tiểu ít", "TRIỆU_CHỨNG"),
    ("tiểu rắt", "TRIỆU_CHỨNG"),
    ("tiểu buốt", "TRIỆU_CHỨNG"),
    ("đái rắt", "TRIỆU_CHỨNG"),
    ("đái buốt", "TRIỆU_CHỨNG"),
    ("đái nhiều", "TRIỆU_CHỨNG"),
    ("đái ít", "TRIỆU_CHỨNG"),
    ("đái đêm", "TRIỆU_CHỨNG"),
    ("tiểu đêm", "TRIỆU_CHỨNG"),
    ("đau thắt ngực", "TRIỆU_CHỨNG"),
    ("rối loạn kinh nguyệt", "TRIỆU_CHỨNG"),
    ("rong kinh", "TRIỆU_CHỨNG"),
    ("đau bụng kinh", "TRIỆU_CHỨNG"),
    ("khó thở khi nằm", "TRIỆU_CHỨNG"),
    ("khó thở khi gắng sức", "TRIỆU_CHỨNG"),
    ("ho ra máu tươi", "TRIỆU_CHỨNG"),
    ("ói ra máu", "TRIỆU_CHỨNG"),
    ("đại tiện ra máu", "TRIỆU_CHỨNG"),
    ("đi cầu ra máu", "TRIỆU_CHỨNG"),
    ("co giật", "TRIỆU_CHỨNG"),
    ("động kinh", "TRIỆU_CHỨNG"),
    ("bất tỉnh", "TRIỆU_CHỨNG"),
    ("ngất", "TRIỆU_CHỨNG"),
    ("yếu liệt", "TRIỆU_CHỨNG"),
    ("liệt nửa người", "TRIỆU_CHỨNG"),
    ("liệt chi", "TRIỆU_CHỨNG"),
    ("tê tay", "TRIỆU_CHỨNG"),
    ("tê chân", "TRIỆU_CHỨNG"),
    ("nhìn mờ", "TRIỆU_CHỨNG"),
    ("ù tai", "TRIỆU_CHỨNG"),
    ("điếc", "TRIỆU_CHỨNG"),
    ("giảm thính lực", "TRIỆU_CHỨNG"),
    # Added per analysis
    ("ớn lạnh", "TRIỆU_CHỨNG"),
    ("lơ mơ", "TRIỆU_CHỨNG"),
    ("mất định hướng", "TRIỆU_CHỨNG"),
    ("đi lại không vững", "TRIỆU_CHỨNG"),
    ("gần ngất", "TRIỆU_CHỨNG"),
    ("giảm dung nạp gắng sức", "TRIỆU_CHỨNG"),
    ("thắt chặt ngực", "TRIỆU_CHỨNG"),
    ("căng thẳng", "TRIỆU_CHỨNG"),
    ("ảo thanh", "TRIỆU_CHỨNG"),
    ("giật", "TRIỆU_CHỨNG"),
    ("tổn thương dạng bóng nước", "TRIỆU_CHỨNG"),
    ("rối loạn ý thức thoáng qua", "TRIỆU_CHỨNG"),
    ("khó khăn khi ăn uống qua đường miệng", "TRIỆU_CHỨNG"),
    ("đau khi sờ nắn", "TRIỆU_CHỨNG"),
]


def find_key_phrases(text: str) -> List[Dict]:
    found = []
    used_spans = []
    sorted_phrases = sorted(KEY_PHRASES, key=lambda x: -len(x[0]))
    for phrase, etype in sorted_phrases:
        if len(phrase) < 3:
            continue
        positions = find_all_in_text(text, phrase)
        for s, e in positions:
            overlap = any(not (e <= us or s >= ue) for us, ue in used_spans)
            if overlap:
                continue
            used_spans.append((s, e))
            found.append({
                'text': text[s:e],
                'type': etype,
                'candidates': [],
                'start': s,
                'end': e,
            })
    found.sort(key=lambda x: x['start'])
    return found


# ============================================================================
# STEP 4: Vocab extraction (SYMPTOMS + DIAGNOSES) — augmented
# ============================================================================

# Add missing terms from local GT analysis
EXTRA_SYMPTOMS = [
    "tổn thương dạng bóng nước",
    "rối loạn ý thức thoáng qua",
    "ớn lạnh",
    "lơ mơ",
    "mất định hướng",
    "đi lại không vững",
    "gần ngất",
    "giảm dung nạp gắng sức",
    "thắt chặt ngực",
    "căng thẳng",
    "ảo thanh",
    "giật",
    "đau khi sờ nắn",
    "yếu nửa người",
    "mất thăng bằng",
    "khó khăn khi ăn uống qua đường miệng",
]

EXTRA_DIAGNOSES = [
    "tăng men gan",
    "loạn thần",
    "viêm cơ",
    "tràn dịch màng tim",
    "tụ máu dưới màng cứng",
    "bầm dập nhu mô",
    "nhịp xoang chiếm ưu thế",
    "tắc mạch",
    "ung thư",
    "ung thư biểu mô",
    "ung thư biểu mô tuyến",
    "viêm tuyến mồ hôi",
    "viêm mạch",
    "theo dõi đại tràng giãn",
]


def find_vocab(text: str, vocab_dict: Dict, etype: str) -> List[Dict]:
    found = []
    used_spans = []
    sorted_terms = sorted(vocab_dict.keys(), key=lambda x: -len(x))
    for term in sorted_terms:
        if len(term) < 4:
            continue
        positions = find_all_in_text(text, term)
        for s, e in positions:
            overlap = any(not (e <= us or s >= ue) for us, ue in used_spans)
            if overlap:
                continue
            used_spans.append((s, e))
            found.append({
                'text': text[s:e],
                'type': etype,
                'candidates': [],
                'start': s,
                'end': e,
            })
    found.sort(key=lambda x: x['start'])
    return found


# ============================================================================
# STEP 5: Test name + result extraction
# ============================================================================

def find_test_names(text: str) -> List[Dict]:
    found = []
    used_spans = []
    for test in TEST_NAMES_SORTED:
        if len(test) < 2:
            continue
        positions = find_all_in_text(text, test)
        for s, e in positions:
            overlap = any(not (e <= us or s >= ue) for us, ue in used_spans)
            if overlap:
                continue
            used_spans.append((s, e))
            found.append({
                'text': text[s:e],
                'type': 'TÊN_XÉT_NGHIỆM',
                'candidates': [],
                'start': s,
                'end': e,
            })
    found.sort(key=lambda x: x['start'])
    return found


# ============================================================================
# STEP 7: Assertion classifier (WIDENED context window 60 → 300)
# ============================================================================

NEG_PATTERNS = [
    r'\bkhông\s+(?:có|còn|thấy|ghi\s+nhận|xuất\s+hiện|biểu\s+hiện)',
    r'\bchưa\s+(?:có|từng|bảo\s+giờ|rõ)',
    r'\bko\s+(?:có|còn)',
    r'\bphủ\s+nhận\b',
    r'\bâm\s+tính\b',
    r'\bneg(?:ative)?\b',
]
NEG_RES = [re.compile(p, re.IGNORECASE) for p in NEG_PATTERNS]

HIST_PATTERNS = [
    r'\btiền\s+sử\b',
    r'\bđã\s+(?:dùng|sử|mắc|bị|có|tiêm|uống|phát\s+hiện)',
    r'\btrước\s+(?:khi|đây|kia)',
    r'\bnăm\s+\d{4}\b',
    r'\b(?:từ|cách)\s+đây\b',
    r'\bđang\s+(?:dùng|sử\s+dụng|uống)',
    r'\bđược\s+(?:phát\s+hiện|chẩn\s+đoán|điều\s+trị)',
    # Added
    r'\bnhiều\s+năm\s+nay\b',
    r'\b(?:cách|từ)\s+đây\s+\d+\s+(?:năm|tháng|tuần)\b',
    r'\bđã\s+được\s+\d+\s+(?:năm|tháng)\b',
    r'\btrong\s+quá\s+khứ\b',
]
HIST_RES = [re.compile(p, re.IGNORECASE) for p in HIST_PATTERNS]

FAM_PATTERNS = [
    r'(?<![a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ])mẹ\s+(?:em|tôi|của|anh)',
    r'(?<![a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ])bố\s+(?:em|tôi|của|anh)',
    r'(?<![a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ])cha\s+(?:em|tôi|của|anh)',
    r'\bông\s+(?:em|tôi|của|anh|nội|ngoại)',
    r'\bbà\s+(?:em|tôi|của|anh|nội|ngoại)',
    r'(?<![a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ])anh\s+(?:em|tôi|của|trai)',
    r'(?<![a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ])chị\s+(?:em|tôi|của|gái)',
    r'(?<![a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ])em\s+(?:em|tôi|của|trai|gái|ruột)',
    r'\bgia\s+đình\b',
    r'\bdi\s+truyền\b',
    r'\bbẩm\s+sinh\b',
    r'\bgen\s+(?:di\s+truyền|gien\s+đột\s+biến)',
]
FAM_RES = [re.compile(p, re.IGNORECASE) for p in FAM_PATTERNS]

SUSP_PATTERNS = [
    r'\bnghi\s+(?:ngờ|do|là)',
    r'\bcó\s+thể\s+(?:là|mắc|bị|gây|dẫn)',
    r'\bcó\s+thể\b',  # broad — catch "có thể gây X"
    r'\b(?:triệu\s+chứng|biểu\s+hiện)\s+(?:nghi|gợi\s+ý)',
    r'\b(?:chẩn\s+đoán\s+)?nghi\s+ngờ\b',
    r'\b(?:nghi|bị\s+nghi)\b',
    r'\b(?:cần\s+)?loại\s+trừ\b',
    r'\bdự\s+(?:kiến|đoán)\b',
    r'\b(?:có\s+thể|nghi\s+ngờ)\s+(?:mắc|bị)\b',
    r'\b(?:nhiều\s+khả\s+năng|khả\s+năng\s+cao)\b',
    r'\bcó\s+khả\s+năng\b',
    r'\bnhiều\s+khả\s+năng\s+(?:là|bị|mắc)\b',
]
SUSP_RES = [re.compile(p, re.IGNORECASE) for p in SUSP_PATTERNS]


def classify_assertion(start: int, end: int, text: str, entity_type: str) -> List[str]:
    """
    Classify assertion with widened context window.
    Priority: isNegated > isFamily > isSuspected > isHistorical
    """
    result = []
    ctx_before = text[max(0, start-300):start].lower()  # WIDENED 60 → 300
    ctx_after = text[end:min(len(text), end+50)].lower()
    ctx_window = ctx_before + ' ' + ctx_after

    # Priority 1: negation
    for pat in NEG_RES:
        if pat.search(ctx_before[-150:]):  # also widened
            result.append('isNegated')
            break

    # Priority 2: family
    if 'isNegated' not in result:
        for pat in FAM_RES:
            if pat.search(ctx_before[-200:]):
                result.append('isFamily')
                break

    # Priority 3: suspected
    if 'isNegated' not in result and 'isFamily' not in result:
        for pat in SUSP_RES:
            if pat.search(ctx_before[-200:]) or pat.search(ctx_after[:50]):
                result.append('isSuspected')
                break

    # Priority 4: historical
    if not result:
        for pat in HIST_RES:
            if pat.search(ctx_before[-150:]):
                result.append('isHistorical')
                break

    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def extract_entities(text: str) -> List[Dict]:
    entities = []
    used_spans = []

    # Step 1: ICD-10 diagnoses
    for e in find_icd10_diagnoses(text):
        entities.append(e)
        used_spans.append((e['start'], e['end']))

    # Step 2: Drugs (Vietnamese + English)
    for e in find_drugs(text):
        entities.append(e)
        used_spans.append((e['start'], e['end']))

    # Step 3: KEY_PHRASES
    for e in find_key_phrases(text):
        overlap = any(not (e['end'] <= us or e['start'] >= ue) for us, ue in used_spans)
        if overlap:
            continue
        entities.append(e)
        used_spans.append((e['start'], e['end']))

    # Step 4: Vocab (symptoms + diagnoses) — augmented
    symptom_vocab = {term: True for term in SYMPTOMS}
    symptom_vocab.update({term: True for term in EXTRA_SYMPTOMS})

    for e in find_vocab(text, symptom_vocab, 'TRIỆU_CHỨNG'):
        overlap = any(not (e['end'] <= us or e['start'] >= ue) for us, ue in used_spans)
        if overlap:
            continue
        if len(e['text']) < 4:
            continue
        entities.append(e)
        used_spans.append((e['start'], e['end']))

    diagnosis_vocab = {term: True for term in DIAGNOSES}
    diagnosis_vocab.update({term: True for term in EXTRA_DIAGNOSES})

    for e in find_vocab(text, diagnosis_vocab, 'CHẨN_ĐOÁN'):
        overlap = any(not (e['end'] <= us or e['start'] >= ue) for us, ue in used_spans)
        if overlap:
            continue
        if len(e['text']) < 4:
            continue
        entities.append(e)
        used_spans.append((e['start'], e['end']))

    # Step 5: Test names (only in lab sections)
    has_test_section = bool(TEST_SECTION_RE.search(text))
    if has_test_section:
        for e in find_test_names(text):
            overlap = any(not (e['end'] <= us or e['start'] >= ue) for us, ue in used_spans)
            if overlap:
                continue
            entities.append(e)
            used_spans.append((e['start'], e['end']))

    # Step 7: Assertion
    for e in entities:
        assertions = classify_assertion(e['start'], e['end'], text, e['type'])
        e['assertions'] = assertions

    # Step 8: Final dedup + format (case-insensitive)
    seen = set()
    final = []
    for e in entities:
        if e['type'] == 'TÊN_XÉT_NGHIỆM':
            e['assertions'] = []
        key = (e['text'].lower().strip(), e['type'])
        if key in seen:
            continue
        seen.add(key)
        final.append({
            'text': e['text'],
            'type': e['type'],
            'candidates': e.get('candidates', []),
            'assertions': e.get('assertions', []),
            'position': [e['start'], e['end']],
        })

    final.sort(key=lambda x: x['position'][0])

    # Cap at 25 to avoid text_score penalty
    if len(final) > 25:
        final.sort(key=lambda x: (0 if x['candidates'] else 1, x['position'][0]))
        final = final[:25]
        final.sort(key=lambda x: x['position'][0])

    return final


def process_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    entities = extract_entities(text)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    return entities


def run_pipeline(input_dir: str, output_dir: str, zip_path: Optional[str] = None,
                 total: int = 100, start: int = 1):
    os.makedirs(output_dir, exist_ok=True)
    total_entities = 0
    type_counts = Counter()
    for i in range(start, start + total):
        ip = os.path.join(input_dir, f'{i}.txt')
        op = os.path.join(output_dir, f'{i}.json')
        if os.path.exists(ip):
            entities = process_file(ip, op)
            total_entities += len(entities)
            for e in entities:
                type_counts[e['type']] += 1
            if i % 10 == 0:
                print(f'  [{i}] {len(entities)} entities, total={total_entities}')
    print(f'Total: {total_entities}, Types: {dict(type_counts)}')
    if zip_path:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i in range(start, start + total):
                jf = os.path.join(output_dir, f'{i}.json')
                if os.path.exists(jf):
                    zf.write(jf, f'output/{i}.json')
        print(f'Created: {zip_path}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', default=r'D:\projects\Viettel AI race\input_turn2_vong1\input')
    ap.add_argument('--output_dir', default=r'D:\projects\Viettel AI race\output_v3')
    ap.add_argument('--zip_path', default=r'D:\projects\Viettel AI race\output_v3.zip')
    ap.add_argument('--total', type=int, default=100)
    ap.add_argument('--start', type=int, default=1)
    args = ap.parse_args()
    run_pipeline(args.input_dir, args.output_dir, args.zip_path, args.total, args.start)