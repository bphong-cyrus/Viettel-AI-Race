# -*- coding: utf-8 -*-
"""
Optimized Vietnamese Medical NER Pipeline v3
Designed to maximize: 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score

Key improvements over v2:
- Better recall for common medical terms
- Tightened word-boundary matching
- Better assertion classification
- Better CUI mapping
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
# 1) KEY_PHRASES - High priority patterns
# ============================================================================
KEY_PHRASES = [
    # Diagnoses - complex multi-word
    (r'\bnhịp\s+xoang(?:\s+chiếm\s+ưu\s+thế)?\b', 'CHẨN_ĐOÁN'),
    (r'\bngoại\s+tâm\s+thu\s+nhĩ\b', 'CHẨN_ĐOÁN'),
    (r'\bngoại\s+tâm\s+thu\s+thất\b', 'CHẨN_ĐOÁN'),
    (r'\bnghẽn\s+tắc\s+và\s+hẹp\s+động\s+mạch\s+cảnh\b', 'CHẨN_ĐOÁN'),
    (r'\bhội\s+chứng\s+mạch\s+vành\s+cấp\b', 'CHẨN_ĐOÁN'),
    (r'\bnhồi\s+máu\s+cơ\s+tim\b', 'CHẨN_ĐOÁN'),
    (r'\bbệnh\s+mạch\s+vành\s+(?:mạn(?:\s+tính)?)?\b', 'CHẨN_ĐOÁN'),
    (r'\bbệnh\s+thiếu\s+máu\s+cơ\s+tim\b', 'CHẨN_ĐOÁN'),
    (r'\bbệnh\s+tim\s+mạch\s+do\s+xơ\s+vữa\b', 'CHẨN_ĐOÁN'),
    (r'\bhẹp\s+động\s+mạch\s+cảnh\b', 'CHẨN_ĐOÁN'),
    (r'\bđau\s+thắt\s+ngực(?:\s+(?:ổn\s+định|không\s+ổn\s+định|kiểu\s+co\s+thắt|thể\s+tự\s+nhiên))?\b', 'CHẨN_ĐOÁN'),
    (r'\bsuy\s+tim(?:\s+(?:sung\s+huyết|trái|phải|toàn\s+bộ|mạn(?:\s+tính)?|cấp|độ\s+[iIvV]+))?\b', 'CHẨN_ĐOÁN'),
    (r'\bđái\s+tháo\s+đường(?:\s+(?:type\s*[12]|týp\s*[12]))?\b', 'CHẨN_ĐOÁN'),
    (r'\btăng\s+huyết\s+áp\b', 'CHẨN_ĐOÁN'),
    (r'\bhạ\s+huyết\s+áp\b', 'CHẨN_ĐOÁN'),
    (r'\b béo\s+phì\b', 'CHẨN_ĐOÁN'),
    (r'\bbệnh\s+kawasaki\b', 'CHẨN_ĐOÁN'),
    (r'\bviêm\s+(?:gan|dạ\s+dày|phổi|thực\s+quản|khớp|cơ|bàng\s+quang|xoang|họng|tai\s+giữa|tụy|màng\s+não|mũi|đa\s+khớp)\b', 'CHẨN_ĐOÁN'),
    (r'\bviêm\s+dạ\s+dày(?:\s+ruột)?\b', 'CHẨN_ĐOÁN'),
    (r'\bviêm\s+tủy\s+xương\b', 'CHẨN_ĐOÁN'),
    (r'\bviêm\s+xương\b', 'CHẨN_ĐOÁN'),
    (r'\bthoái\s+hóa\s+(?:khớp|cột\s+sống)\b', 'CHẨN_ĐOÁN'),
    (r'\bthoa\s+hóa\s+(?:khớp|cột\s+sống)\b', 'CHẨN_ĐOÁN'),
    (r'\btrào\s+ngược(?:\s+dạ\s+dày\s+thực\s+quản)?\b', 'CHẨN_ĐOÁN'),
    (r'\brung\s+nhĩ\b', 'CHẨN_ĐOÁN'),
    (r'\bxơ\s+(?:gan|vữa|phổi)\b', 'CHẨN_ĐOÁN'),
    (r'\bsuy\s+(?:thận|gan|hô\s+hấp|giáp|thượng\s+thận)(?:\s+(?:mạn(?:\s+tính)?|cấp))?\b', 'CHẨN_ĐOÁN'),
    (r'\bthiếu\s+máu(?:\s+(?:cơ\s+tim|não|cấp|mạn|nặng|nhẹ|tan\s+huyết|thiếu\s+sắt))?\b', 'CHẨN_ĐOÁN'),
    (r'\btăng\s+(?:huyết\s+áp|kali\s+máu|đường\s+huyết|men\s+gan|natri\s+máu|triglyceride|cholesterol|bilirubin|bilirubin\s+máu)\b', 'CHẨN_ĐOÁN'),
    (r'\bhạ\s+(?:kali\s+máu|đường\s+huyết|canxi\s+máu|magne\s+máu|natri\s+máu)\b', 'CHẨN_ĐOÁN'),
    (r'\bđa\s+u\s+tuỷ\s+xương\b', 'CHẨN_ĐOÁN'),
    (r'\bung\s+thư\b', 'CHẨN_ĐOÁN'),
    (r'\bhội\s+chứng\s+(?:mạch\s+vành\s+cấp|ruột\s+kích\s+thích|cushing|nghiện\s+rượu|parkinson|thận\s+hư|chuyển\s+hóa|ống\s+cổ\s+tay|thực\s+bào)\b', 'CHẨN_ĐOÁN'),
    (r'\bphình\s+(?:động\s+mạch|mạch|giãn\s+động\s+mạch)\b', 'CHẨN_ĐOÁN'),
    (r'\btắc\s+(?:mạch|ruột|nghẽn|động\s+mạch)\b', 'CHẨN_ĐOÁN'),
    (r'\bnhiễm\s+(?:trùng\s+huyết|khuẩn|độc)\b', 'CHẨN_ĐOÁN'),
    (r'\bnhịp\s+(?:nhanh|chậm|xoang)\b', 'CHẨN_ĐOÁN'),
    (r'\bsỏi\s+(?:thận|mật|đường\s+mật)\b', 'CHẨN_ĐOÁN'),
    (r'\btai\s+biến\s+(?:mạch\s+máu\s+)?não\b', 'CHẨN_ĐOÁN'),
    (r'\bđột\s+quỵ\b', 'CHẨN_ĐOÁN'),
    (r'\bCOPD\b', 'CHẨN_ĐOÁN'),
    (r'\bAsthma\b', 'CHẨN_ĐOÁN'),
    (r'\bhen\s+(?:phế\s+quản|suyễn|asthma)\b', 'CHẨN_ĐOÁN'),
    (r'\bhội\s+chứng\s+andersen\b', 'CHẨN_ĐOÁN'),

    # Symptoms - complex multi-word
    (r'\bkhó\s+thở(?:\s+(?:khi\s+gắng\s+sức|khi\s+nằm|khi\s+ngủ|khi\s+nói|khi\s+hoạt\s+động|kéo\s+dài|liên\s+tục|nhẹ|đột\s+ngột|vào|ra|không\s+thở\s+được))?\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+ngực(?:\s+(?:khi\s+thở|kiểu\s+bóp\s+nghẹt|kiểu\s+thiếu\s+máu|lan\s+lên|lan\s+ra|phải|trái|sau\s+xương\s+ức|ít|thắt\s+ngực))?\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+bụng(?:\s+(?:dưới|dữ\s+dội|kinh|trên|quanh\s+rốn|âm\s+ỉ|cơn|vùng\s+hạ\s+sườn\s+phải|vùng\s+thượng\s+vị|từng\s+cơn|ngày\s+càng\s+nặng|liên\s+tục|trở\s+nên\s+tồi\s+ter))?\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+đầu(?:\s+(?:dữ\s+dội|kinh\s+kiểu\s+migraine|migraine|nặng|vận\s+mạch|tăng\s+dần|kéo\s+dài|căng\s+cơ))?\b', 'TRIỆU_CHỨNG'),
    (r'\bgiảm\s+dung\s+nạp\s+gắng\s+sức\b', 'TRIỆU_CHỨNG'),
    (r'\btê\s+(?:bì|tay|chân|nửa\s+mặt)(?:\s+(?:vùng\s+trán\s+phải|nửa\s+mặt\s+phải|chân\s+tay|ở\s+cánh\s+tay\s+trái))?\b', 'TRIỆU_CHỨNG'),
    (r'\byếu\s+(?:nửa\s+người|sức|tay|chân|cơ|toàn\s+thân)\b', 'TRIỆU_CHỨNG'),
    (r'\bphù\s+(?:mắt\s+cá\s+chân|chân|tay|hai\s+chân|hai\s+bên|ngoại\s+vi|toàn\s+thân|não|phổi|gan)\b', 'TRIỆU_CHỨNG'),
    (r'\bsốt(?:\s+(?:cao|nhẹ|không\s+rõ\s+nguyên\s+nhân|rét\s+run|về\s+chiều|về\s+đêm|kéo\s+dài|phát\s+ban))?\b', 'TRIỆU_CHỨNG'),
    (r'\bho(?:\s+(?:ra\s+máu|khan|có\s+đờm|máu|máu\s+tươi|mạn\s+tính|ra\s+đờm))?\b', 'TRIỆU_CHỨNG'),
    (r'\bnôn(?:\s+(?:ra\s+máu|khan|ói|dịch\s+vàng|ra\s+thức\s+ăn))?\b', 'TRIỆU_CHỨNG'),
    (r'\bói(?:\s+(?:mửa|ra\s+máu))?\b', 'TRIỆU_CHỨNG'),
    (r'\bbuồn\s+nôn(?:\s+(?:nhẹ|sau\s+ăn))?\b', 'TRIỆU_CHỨNG'),
    (r'\bđánh\s+trống\s+ngực(?:\s+(?:khi\s+gắng\s+sức|liên\s+hồi|liên\s+tục|từng\s+cơn))?\b', 'TRIỆU_CHỨNG'),
    (r'\btăng\s+đánh\s+trống\s+ngực\b', 'TRIỆU_CHỨNG'),
    (r'\bchóng\s+mặt(?:\s+(?:khi\s+thay\s+đổi\s+tư\s+thế|từng\s+đợt))?\b', 'TRIỆU_CHỨNG'),
    (r'\bmất\s+ngủ\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+vùng\s+hạ\s+sườn\s+phải\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+hạ\s+sườn\s+(?:phải|trái)\b', 'TRIỆU_CHỨNG'),
    (r'\bngất(?:\s+(?:xỉu|do\s+tim|khi\s+thay\s+đổi\s+tư\s+thế))?\b', 'TRIỆU_CHỨNG'),
    (r'\bngất\s+xỉu\b', 'TRIỆU_CHỨNG'),
    (r'\bgắng\s+sức\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+nhức\b', 'TRIỆU_CHỨNG'),
    (r'\blo\s+âu\b', 'TRIỆU_CHỨNG'),
    (r'\bhụt\s+hơi\b', 'TRIỆU_CHỨNG'),
    (r'\bmệt\s+mỏi\b', 'TRIỆU_CHỨNG'),
    (r'\btáo\s+bón\b', 'TRIỆU_CHỨNG'),
    (r'\btiêu\s+chảy\b', 'TRIỆU_CHỨNG'),
    (r'\bmất\s+thăng\s+bằng\b', 'TRIỆU_CHỨNG'),
    (r'\bgần\s+ngất\b', 'TRIỆU_CHỨNG'),
    (r'\bho\s+ra\s+máu\b', 'TRIỆU_CHỨNG'),
    (r'\bxuất\s+huyết(?:\s+(?:não|dưới\s+da|đường\s+tiêu\s+hoá|tiêu\s+hóa))?\b', 'TRIỆU_CHỨNG'),
    (r'\bco\s+giật(?:\s+(?:toàn\s+thân|cục\s+bộ|kiểu\s+động\s+kinh))?\b', 'TRIỆU_CHỨNG'),
    (r'\bđổ\s+mồ\s+hôi\b', 'TRIỆU_CHỨNG'),
    (r'\bvàng\s+da\b', 'TRIỆU_CHỨNG'),
    (r'\bvàng\s+mắt\b', 'TRIỆU_CHỨNG'),
    (r'\bvàng\s+da\s+vàng\s+mắt\b', 'TRIỆU_CHỨNG'),
    (r'\bhôn\s+mê\b', 'TRIỆU_CHỨNG'),
    (r'\blú\s+lẫn\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+lưng(?:\s+âm\s+ỉ)?\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+chân\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+hông\b', 'TRIỆU_CHỨNG'),
    (r'\bphù\s+mắt\s+cá\s+chân\b', 'TRIỆU_CHỨNG'),
    (r'\bchướng\s+bụng\b', 'TRIỆU_CHỨNG'),
    (r'\bCăng\s+thẳng\b', 'TRIỆU_CHỨNG'),
    (r'\bcăng\s+thẳng\b', 'TRIỆU_CHỨNG'),
    (r'\bthắt\s+chặt\s+ngực\b', 'TRIỆU_CHỨNG'),
    (r'\bkhó\s+nuốt\b', 'TRIỆU_CHỨNG'),
    (r'\bkhó\s+tiêu\b', 'TRIỆU_CHỨNG'),
    (r'\bphù\s+nề\w*\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+thượng\s+vị\b', 'TRIỆU_CHỨNG'),
    (r'\bđầy\s+bụng\b', 'TRIỆU_CHỨNG'),
    (r'\bợ\s+(?:nóng|hơi|chua)\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+quanh\s+rốn\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+hạ\s+vị\b', 'TRIỆU_CHỨNG'),
    (r'\bchoáng\s+váng\b', 'TRIỆU_CHỨNG'),
    (r'\brun\s+(?:tay|tay\s+chân|rẩy|chi)\b', 'TRIỆU_CHỨNG'),
    (r'\brét\s+run\b', 'TRIỆU_CHỨNG'),
    (r'\bớn\s+lạnh\b', 'TRIỆU_CHỨNG'),
    (r'\bnhức\s+đầu\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+vai\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+cổ\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+khớp\b', 'TRIỆU_CHỨNG'),
    (r'\bkhò\s+khè\b', 'TRIỆU_CHỨNG'),
    (r'\bkhàn\s+tiếng\b', 'TRIỆU_CHỨNG'),
    (r'\bgiọng\s+khàn\b', 'TRIỆU_CHỨNG'),
    (r'\btim\s+đập\s+nhanh\b', 'TRIỆU_CHỨNG'),
    (r'\bnước\s+tiểu\s+sẫm\s+màu\b', 'TRIỆU_CHỨNG'),
    (r'\bnước\s+tiểu\s+ít\b', 'TRIỆU_CHỨNG'),
    (r'\btiểu\s+ra\s+máu(?:\s+không\s+đau)?\b', 'TRIỆU_CHỨNG'),
    (r'\btiểu\s+(?:ít|nhiều|đêm|buốt|rát|khó)\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\s+rát\b', 'TRIỆU_CHỨNG'),
    (r'\bphát\s+ban\b', 'TRIỆU_CHỨNG'),
    (r'\bngứa(?:\s+(?:toàn\s+thân|nhiều))?\b', 'TRIỆU_CHỨNG'),
    (r'\bdị\s+ứng\b', 'TRIỆU_CHỨNG'),
    (r'\bsưng(?:\s+(?:phù|đỏ))?\b', 'TRIỆU_CHỨNG'),
    (r'\btăng\s+cân\b', 'TRIỆU_CHỨNG'),
    (r'\bgiảm\s+cân\b', 'TRIỆU_CHỨNG'),
    (r'\bbụng\s+báng\b', 'TRIỆU_CHỨNG'),
    (r'\btúi\s+mật\s+giãn\b', 'CHẨN_ĐOÁN'),
    (r'\btăng\s+bilirubin\b', 'CHẨN_ĐOÁN'),
    (r'\btăng\s+men\s+gan\b', 'CHẨN_ĐOÁN'),
    (r'\btăng\s+huyết\s+áp\b', 'CHẨN_ĐOÁN'),
    (r'\bviêm\s+gan\s+virus\b', 'CHẨN_ĐOÁN'),
    (r'\bcổ\s+trướng\b', 'CHẨN_ĐOÁN'),
    (r'\bliệt(?:\s+(?:nửa\s+người|chân|tay))?\b', 'TRIỆU_CHỨNG'),
    (r'\bloét(?:\s+(?:dạ\s+dày|tá\s+tràng|hành\s+tá\s+tràng|mép))?\b', 'CHẨN_ĐOÁN'),
    (r'\bgiãn\s+(?:tĩnh\s+mạch|động\s+mạch)\b', 'CHẨN_ĐOÁN'),
    (r'\bdày\s+thất\b', 'CHẨN_ĐOÁN'),
    (r'\bdày\s+thành\b', 'CHẨN_ĐOÁN'),
    (r'\bcơn\s+(?:đau\s+thắt\s+ngực|hen|cường\s+giáp)\b', 'TRIỆU_CHỨNG'),
]
COMPILED_KEY_PHRASES = [(re.compile(p, re.IGNORECASE), t) for p, t in KEY_PHRASES]


# ============================================================================
# 2) DRUG DICTIONARY
# ============================================================================
DRUG_KEYS_SORTED = sorted(DRUG_DICT.keys(), key=lambda k: -len(k))


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
            before_ok = (idx == 0) or (not text_lower[idx-1].isalnum())
            after_idx = idx + len(key)
            after_ok = (after_idx >= len(text_lower)) or (not text_lower[after_idx].isalnum())
            if before_ok and after_ok:
                overlap = any(not (after_idx <= us or idx >= ue) for us, ue in used_spans)
                if not overlap:
                    end = after_idx
                    # Extend drug name with dose and route/freq - tighter
                    tail_match = re.match(
                        r'(?:\s*\d+(?:\.\d+)?(?:\s*(?:mg|mcg|μg|ug|g|ml|iu|ui|meq|mEq|%)))'
                        r'(?:\s+x\s+\d+)?'
                        r'(?:\s+(?:po|iv|im|sc|ng|sl|pr|top|uống|tiêm|đặt|qd|bid|tid|qid|qhs|prn|daily|ngày|hours?|giờ|h|am|pm|q\d+h?))?'
                        r'(?:\s+(?:po|iv|im|sc|ng|sl|pr|top|qd|bid|tid|qid|qhs|prn|daily|q\d+h?))?',
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
    if t in DRUG_DICT:
        cuis.add(DRUG_DICT[t])
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
# 3) VOCAB MATCHING
# ============================================================================
SYMPTOMS_SORTED = sorted(set(SYMPTOMS), key=lambda x: -len(x))
DIAGNOSES_SORTED = sorted(set(DIAGNOSES), key=lambda x: -len(x))


# Terms too short/generic to match alone (word boundary only)
ULTRA_SHORT_BLACKLIST = {
    'ho', 'ói', 'ù', 'đổ', 'ợ', 'tê', 'phù', 'lạnh', 'nặng',
    'mềm', 'đầy', 'bỏ', 'mắt', 'tai', 'mũi', 'gan',
    'ruột', 'phổi', 'thắt', 'đặc', 'xanh', 'vàng',
    'táo', 'tiêu', 'viêm', 'sốt', 'chóng', 'ngất',
    'nôn', 'mửa', 'rát', 'ngứa', 'run',
    'yếu', 'mệt', 'mất', 'suy', 'khó', 'đau',
    'xỉu', 'váng', 'lo', 'buồn', 'chán',
    'rung', 'loét', 'rối', 'lú', 'ớn', 'mày',
    'gầy', 'mập', 'bí', 'ngoái', 'gồng',
    'nhức', 'sợi', 'rứt', 'bứt', 'quàng',
    'thấp', 'cao', 'cứng', 'khác', 'phụ',
    'nghén', 'có', 'không', 'sưng',
    'lạnh', 'gan', 'thận',
}

# Stopwords
_STOPWORDS = {'và', 'của', 'cho', 'trong', 'trên', 'với', 'không', 'có', 'là', 'được',
              'này', 'kia', 'đó', 'thì', 'mà', 'như', 'nên', 'rất', 'cũng'}


def _filter_terms(terms_list):
    """Filter out ultra-short and stopword terms."""
    filtered = []
    for t in terms_list:
        tl = t.lower().strip()
        if len(tl) <= 2:
            continue
        if tl in ULTRA_SHORT_BLACKLIST:
            continue
        if tl in _STOPWORDS:
            continue
        # Filter pure stopword phrases
        words = tl.split()
        if all(w in _STOPWORDS for w in words):
            continue
        filtered.append(t)
    return filtered


SYMPTOMS_SORTED = _filter_terms(SYMPTOMS_SORTED)
DIAGNOSES_SORTED = _filter_terms(DIAGNOSES_SORTED)


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
        if term_lower in ULTRA_SHORT_BLACKLIST:
            continue
        if term_lower in _STOPWORDS:
            continue
        if tlen <= 2:
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
    (r'thuốc\s+(?:trước|đang|hiện|dùng|tại|nhà|sau)', 'THUỐC'),
    (r'phác\s+đồ\s+điều\s+trị', 'THUỐC'),
    (r'điều\s+trị\s+(?:hiện\s+tại|tại\s+nhà|trước\s+đây)', 'THUỐC'),
    (r'tiền\s+sử\s+bệnh(?:\s+nội\s+khoa|\s+lý|\s+ngoại\s+khoa)?', 'CHẨN_ĐOÁN'),
    (r'tiền\s+căn', 'CHẨN_ĐOÁN'),
    (r'bệnh\s+lý\s+(?:mãn\s+tính|nền|kèm|theo)', 'CHẨN_ĐOÁN'),
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
    (r'kết\s+quả\s+(?:xét\s+nghiệm|chẩn\s+đoán)', 'CHẨN_ĐOÁN'),
    (r'kết\s+quả\s+chẩn\s+đoán\s+hình\s+ảnh', 'CHẨN_ĐOÁN'),
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
            if i > 0:
                return sections[i - 1] if i - 1 < len(sections) else None
            return sections[0] if sections else None
    return sections[-1] if sections else None


# ============================================================================
# 5) ASSERTION CLASSIFICATION
# ============================================================================
NEG_PATTERNS = [
    re.compile(r'\bkhông\s+(?:có|còn|thấy|ghi\s+nhận|xuất\s+hiện|đau|thở|sốt|buồn|sợ|ngứa|ra|cảm|ớn|phù|ho|chảy|rát|ngạt|bị)', re.IGNORECASE),
    re.compile(r'\bchưa\s+(?:có|từng|bảo\s+giờ|rõ)', re.IGNORECASE),
    re.compile(r'\bko\s+(?:có|còn)', re.IGNORECASE),
    re.compile(r'\b(vắng|mất)\s+(?:mặt|tên)', re.IGNORECASE),
    re.compile(r'\bphủ\s+nhận\b', re.IGNORECASE),
    re.compile(r'\bchưa\s+xác\s+định\b', re.IGNORECASE),
    re.compile(r'\bkhông\s+đặc\s+hiệu\b', re.IGNORECASE),
]
SUSP_PATTERNS = [
    re.compile(r'\bcó\s+thể\b', re.IGNORECASE),
    re.compile(r'\bnghi\s*ngờ\b', re.IGNORECASE),
    re.compile(r'\bnghĩ\s+đến\b', re.IGNORECASE),
    re.compile(r'\bchưa\s+rõ\b', re.IGNORECASE),
    re.compile(r'\bđang\s+theo\s+dõi\b', re.IGNORECASE),
    re.compile(r'\bđang\s+xem\s+xét\b', re.IGNORECASE),
    re.compile(r'\bchưa\s+loại\s+trừ\b', re.IGNORECASE),
    re.compile(r'\bnguy\s+cơ\b', re.IGNORECASE),
]
HIST_PATTERNS = [
    re.compile(r'\btiền\s+sử\b', re.IGNORECASE),
    re.compile(r'\bđã\s+(?:dùng|sử|mắc|bị|có|tiêm|uống|phẫu\s+thuật|thực\s+hiện|được|từng)', re.IGNORECASE),
    re.compile(r'\btrước\s+(?:khi|đây|kia)', re.IGNORECASE),
    re.compile(r'\b(cũ|lâu|năm|tháng|ngày)\s+(?:trước|nay|đây)', re.IGNORECASE),
    re.compile(r'\bđiều\s+trị\s+(?:trước|trước\s+đây)', re.IGNORECASE),
    re.compile(r'\bcó\s+bệnh\s+từ', re.IGNORECASE),
    re.compile(r'\btừ\s+(?:trước|năm|tháng)', re.IGNORECASE),
    re.compile(r'\bnăm\s+(?:ngoái|nay)', re.IGNORECASE),
    re.compile(r'\bnhiều\s+năm\s+trước', re.IGNORECASE),
    re.compile(r'\btừng\s+(?:bị|mắc|dùng)', re.IGNORECASE),
    re.compile(r'\b(?:năm|vài|hai)\s+năm\s+trước', re.IGNORECASE),
    re.compile(r'\bđược\s+chẩn\s+đoán\s+(?:hội\s+chứng|viêm|bệnh|ung\s+thư|tăng|hạ)', re.IGNORECASE),
    re.compile(r'\b(?:trước\s+đó|trước\s+đây)', re.IGNORECASE),
    re.compile(r'\bcách\s+đây\b', re.IGNORECASE),
    re.compile(r'\b(?:trước|khi)\s+nhập\s+viện\b', re.IGNORECASE),
    re.compile(r'\btự\s+điều\s+trị', re.IGNORECASE),
    re.compile(r'\b(?:cắt|bỏ)\s+(?:đại\s+tràng|dạ\s+dày|thận|phổi)', re.IGNORECASE),
    re.compile(r'\b(?:đã\s+ngừng|ngừng)\s+sử\s+dụng', re.IGNORECASE),
    re.compile(r'\bnội\s+soi\b', re.IGNORECASE),
]


def classify_assertion(start: int, end: int, text: str,
                       section_type: Optional[str], entity_type: str) -> List[str]:
    """Classify assertion for an entity."""
    result = []
    ctx_before = text[max(0, start-100):start].lower()
    ctx_after = text[end:min(len(text), end+50)].lower()
    ctx_window = ctx_before + ' ' + ctx_after

    # Check negation FIRST (highest priority)
    negated = False
    for pat in NEG_PATTERNS:
        if pat.search(ctx_before[-60:]):
            result.append('isNegated')
            negated = True
            break

    # Check suspected (only if not negated)
    is_suspected = False
    if not negated:
        for pat in SUSP_PATTERNS:
            if pat.search(ctx_window):
                is_suspected = True
                break

    # Check historical
    is_hist = False
    hist_ctx = ctx_before[-80:]
    if entity_type == 'THUỐC':
        if section_type == 'THUỐC':
            # Drug section: default to historical
            past_indicators = ['trước', 'đã', 'từng', 'cũ', 'tự', 'cách đây', 'sử dụng', 'đang']
            if any(p in hist_ctx for p in past_indicators):
                is_hist = True
            else:
                is_hist = True
        else:
            for pat in HIST_PATTERNS:
                if pat.search(hist_ctx):
                    is_hist = True
                    break
    elif entity_type == 'CHẨN_ĐOÁN':
        if section_type == 'CHẨN_ĐOÁN':
            # Diagnosis section: usually current, not historical
            if any(pat.search(hist_ctx) for pat in HIST_PATTERNS):
                is_hist = True
            # Don't mark as historical if in diagnosis results
            elif re.search(r'(?:kết\s+quả\s+(?:xét\s+nghiệm|chẩn\s+đoán)|chẩn\s+đoán)', hist_ctx):
                is_hist = False
        else:
            for pat in HIST_PATTERNS:
                if pat.search(hist_ctx):
                    is_hist = True
                    break
    else:  # TRIỆU_CHỨNG
        if section_type == 'TRIỆU_CHỨNG':
            is_hist = False  # current symptoms
        else:
            for pat in HIST_PATTERNS:
                if pat.search(hist_ctx):
                    is_hist = True
                    break

    # Negation takes priority over everything
    if 'isNegated' not in result:
        # If suspected, that's the primary assertion (don't add historical)
        if is_suspected:
            result.append('isSuspected')
        elif is_hist:
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
            ph_lower = phrase.lower()
            if len(ph_lower) <= 2:
                continue
            if ph_lower in ULTRA_SHORT_BLACKLIST:
                continue
            # Already in used spans?
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
        drug_clean = re.sub(r'[\.,;:]+$', '', drug_text).strip()
        if not drug_clean or len(drug_clean) < 4:
            continue
        start, end = d['start'], d['end']
        if any(not (end <= us or start >= ue) for us, ue in used_spans):
            continue
        used_spans.append((start, end))
        section_type = get_section_for_pos(start, line_offsets, sections)
        assertions = classify_assertion(start, end, text, section_type, 'THUỐC')
        cui_list = lookup_drug_cuis_for_text(drug_clean)
        if not cui_list and d.get('cui'):
            cui_list = [d['cui']]
        entities.append({
            'text': drug_clean, 'type': 'THUỐC', 'candidates': cui_list,
            'assertions': assertions, 'position': [start, end],
        })

    # ===== Step 2-3: VOCAB per line, section-aware =====
    for i, line in enumerate(lines):
        if i >= len(line_offsets):
            continue
        line_offset = line_offsets[i]
        section_type = sections[i] if i < len(sections) else None
        if not line.strip():
            continue

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

    # ===== Step 4: Lý do nhập viện =====
    ly_do_re = re.compile(r'lý\s+do\s+(?:nhập|vào|khám)(?:\s*viện)?\s*:?\s*([^\n]+)', re.IGNORECASE)
    for m in ly_do_re.finditer(text):
        content = m.group(1).strip()
        if not content or len(content) < 3:
            continue
        content_lower = content.lower()
        content_start = m.start(1)

        for etype, terms in [('TRIỆU_CHỨNG', SYMPTOMS_SORTED), ('CHẨN_ĐOÁN', DIAGNOSES_SORTED)]:
            for term in terms:
                tlen = len(term)
                if tlen < 4:
                    continue
                term_lower = term.lower()
                if term_lower in ULTRA_SHORT_BLACKLIST:
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

    # ===== Step 5: Cleanup =====
    cleaned = []
    for e in entities:
        t = e['text']
        if re.search(r'(?:\s|^)(?:em|anh|chị|bạn|bác|con|người|ông|bà)(?:\s|$)', t.lower()):
            # Keep if it's a critical part of phrase like "cho con bú"
            if not re.search(r'\s+cho\s+con\s+bú', t.lower()):
                continue
        if re.search(r'\s+(?:em|anh|chị|bạn)\s*$', t.lower()):
            continue
        cleaned.append(e)
    entities = cleaned

    # Filter drug names that are lab values
    non_drug_terms = {
        'glucose', 'albumin', 'hemoglobin', 'creatinine', 'urea', 'bilirubin',
        'ast', 'alt', 'alp', 'ferritin', 'ceruloplasmin',
        'đạm', 'đường', 'muối', 'nước',
    }
    drug_filtered = []
    for e in entities:
        if e['type'] == 'THUỐC':
            t = e['text'].lower().strip()
            has_dose = bool(re.search(r'\d+\s*(?:mg|mcg|ug|g|ml|iu|ui)', t, re.IGNORECASE))
            has_route = bool(re.search(r'\b(?:po|iv|im|sc|ng|sl|pr|uống|tiêm|viên)\b', t, re.IGNORECASE))
            has_freq = bool(re.search(r'\b(?:qd|bid|tid|qid|qhs|prn|daily|x\s*\d+|ngày|sáng|trưa|tối|chiều)\b', t, re.IGNORECASE))
            if t in non_drug_terms and not has_dose and not has_route and not has_freq:
                continue
        drug_filtered.append(e)
    entities = drug_filtered

    # Sort by position
    entities.sort(key=lambda x: x['position'][0])

    # Final dedup: exact position
    seen_exact = set()
    deduped1 = []
    for e in entities:
        key = (e['position'][0], e['position'][1], e['type'])
        if key not in seen_exact:
            seen_exact.add(key)
            deduped1.append(e)
    entities = deduped1

    # De-overlap: prefer longer
    final = []
    entities_by_pos = sorted(entities, key=lambda x: (x['position'][0], -(x['position'][1] - x['position'][0])))
    occupied = []
    for e in entities_by_pos:
        s, en = e['position']
        contained = False
        for os_, oe in occupied:
            if os_ <= s and en <= oe:
                contained = True
                break
        if contained:
            continue
        new_occupied = []
        to_remove = []
        for i, (os_, oe) in enumerate(occupied):
            if s <= os_ and oe <= en:
                to_remove.append(i)
            else:
                new_occupied.append((os_, oe))
        occupied = new_occupied
        final.append(e)
        occupied.append((s, en))

    # Final dedup by text+type within similar positions
    seen_text_pos = set()
    result = []
    for e in final:
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
