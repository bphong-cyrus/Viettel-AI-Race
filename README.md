# Vietnamese Medical NER - Viettel AI Race

Hệ thống trích xuất thực thể y tế từ hồ sơ bệnh án tiếng Việt (không dùng LLM).

## Bài toán

Trích xuất các thực thể y tế từ văn bản hồ sơ bệnh án:
- **THUỐC**: Tên thuốc, liều lượng, đường dùng, tần suất
- **TRIỆU_CHỨNG**: Triệu chứng bệnh nhân  
- **CHẨN_ĐOÁN**: Chẩn đoán của bác sĩ

### Output Format

```json
{
  "text": "Aspirin",
  "type": "THUỐC",
  "candidates": ["N0000148261"],
  "assertions": ["isHistorical"],
  "position": [100, 107]
}
```

- `text`: Tên thực thể
- `type`: THUỐC | TRIỆU_CHỨNG | CHẨN_ĐOÁN
- `candidates`: Danh sách RxNorm CUI (THUỐC) hoặc rỗng
- `assertions`: isHistorical | isNegated | isSuspected
- `position`: [start, end] byte offset trong văn bản gốc

### Scoring

```
final_score = 0.3 × text_score + 0.3 × assertions_score + 0.4 × candidates_score
```

- **text_score**: 1 - WER (Word Error Rate) giữa predicted text và ground truth
- **assertions_score**: Jaccard similarity cho assertions
- **candidates_score**: Jaccard similarity có trọng số cho RxNorm candidates

---

## Thuật toán Pipeline (v20)

Pipeline gồm **6 bước**, xử lý tuần tự:

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: văn bản hồ sơ bệnh án (VD: "BN Nam, 65t...")         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 0: KEY_PHRASES (Regex patterns ưu tiên cao)             │
│  - nhịp xoang, suy tim, đái tháo đường...                    │
│  - dùng word boundaries (không trùng với vocab)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Drug Dictionary (find_drugs_in_text)                   │
│  - Tra cứu drug_dict_v3.py (RxNorm database)                   │
│  - Lọc bằng DRUG_BLACKLIST                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Verb-pattern Drug Extraction                           │
│  - Regex: "uống|tiêm|truyền|dùng" + tên thuốc               │
│  - Kiểm tra RxNorm CUI exists                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3-4: Section-aware Vocab Extraction                       │
│  - Phát hiện section (THUỐC/CHẨN_ĐOÁN/TRIỆU_CHỨNG)           │
│  - Extract từ vocab_v5.py (SYMPTOMS/DIAGNOSES)                │
│  - THUỐC section → TRIỆU_CHỨNG trước, CHẨN_ĐOÁN sau        │
│  - CHẨN_ĐOÁN section → ngược lại                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Lý Do Nhập Viện Extraction                           │
│  - Regex: "lý do nhập viện: ..."                              │
│  - Extract triệu chứng/chẩn đoán trong dòng này              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Viêm Reclassification                                 │
│  - "viêm X" trong context "cho/vì/do" → TRIỆU_CHỨNG        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  POST-PROCESS: Assertion Classification                         │
│  - isHistorical: tiền sử, đã, trước...                       │
│  - isNegated: không, chưa, ko...                              │
│  - isSuspected: có thể, nghi ngờ...                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: JSON array entities + Deduplication                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Chi tiết từng bước

### Step 0: KEY_PHRASES

Regex patterns ưu tiên cao cho các thuật ngữ y khoa phức tạp:

```python
KEY_PHRASES = [
    (r'nhịp\s+xoang\s+chiếm\s+ưu\s+thế', 'CHẨN_ĐOÁN'),
    (r'ngoại\s+tâm\s+thu\s+(?:nhĩ|thất)', 'CHẨN_ĐOÁN'),
    (r'hội\s+chứng\s+mạch\s+vành\s+cấp', 'CHẨN_ĐOÁN'),
    (r'nhồi\s+máu\s+cơ\s+tim', 'CHẨN_ĐOÁN'),
    (r'đái\s+tháo\s+đường(?:\s+type\s*[12])?', 'CHẨN_ĐOÁN'),
    (r'tăng\s+huyết\s+áp', 'CHẨN_ĐOÁN'),
    # Single-word symptoms
    (r'\bsốt\b', 'TRIỆU_CHỨNG'),
    (r'\bđau\b', 'TRIỆU_CHỨNG'),
    (r'\bho\b', 'TRIỆU_CHỨNG'),
    (r'\bnôn\b', 'TRIỆU_CHỨNG'),
    # ...
]
```

Dùng `(?<!\S)...(?!\S)` word boundaries để tránh trùng với vocab.

### Step 1: Drug Dictionary

```python
from drug_dict_v3 import find_drugs_in_text, lookup_drug_cuis

drugs = find_drugs_in_text(text)
# → [{'text': 'Aspirin 100mg', 'start': 100, 'end': 113, 'cui': 'N0000148261'}]

cui_list = lookup_drug_cuis('Aspirin')
# → ['N0000148261', 'N0000148262']  # Multiple CUI options
```

DRUG_BLACKLIST lọc các từ không phải thuốc:
```python
DRUG_BLACKLIST = {
    'thuốc trước', 'thuốc sau', 'thuốc nào', 'thuốc đó',
    'bệnh nhân', 'bệnh viện', 'bác sĩ',
    'siêu âm', 'x-quang', 'bình thường',
}
```

### Step 2: Verb-pattern Drug Extraction

```python
# Pattern: "uống|tiêm|truyền|dùng" + drug name
r'\b(?:uống|tiêm|truyền|dùng|sử\s*dụng|đang\s*dùng)\s+([a-zà-ỹ]...)'
```

Ví dụ: "bệnh nhân uống Aspirin 2 viên/ngày" → extract "Aspirin"

### Step 3-4: Section-aware Extraction

**Section Detection** bằng regex patterns:

```python
SECTION_PATTERNS = [
    (r'thuốc\s+(?:đang|hiện|dùng|tại|nhà)', 'THUỐC'),
    (r'tiền\s+sử\s+bệnh', 'CHẨN_ĐOÁN'),
    (r'chẩn\s+đoán', 'CHẨN_ĐOÁN'),
    (r'triệu\s+chứng', 'TRIỆU_CHỨNG'),
    (r'bệnh\s+sử', 'TRIỆU_CHỨNG'),
    (r'lý\s+do\s+nhập', 'TRIỆU_CHỨNG'),
]
```

**Vocab Extraction** từ vocab_v5.py:
- `SYMPTOMS`: ~1000 triệu chứng tiếng Việt
- `DIAGNOSES`: ~500 chẩn đoán tiếng Việt

Thứ tự ưu tiên theo section:
- THUỐC section → TRIỆU_CHỨNG trước, CHẨN_ĐOÁN sau
- CHẨN_ĐOÁN section → ngược lại

### Step 5: Lý Do Nhập Viện

```python
ly_do_re = r'lý\s+do\s+(?:nhập|vào|khám)(?:\s*viện)?\s*:?\s*([^\n]+)'
```

### Step 6: Viêm Reclassification

```python
# "viêm X" sau "cho/vì/do" → TRIỆU_CHỨNG (không phải CHẨN_ĐOÁN)
if e['type'] == 'CHẨN_ĐOÁN' and e['text'].startswith('viêm '):
    ctx = text[max(0, start-30):start].lower()
    if re.search(r'\b(cho|vì|do|biểu\s+hiện)\s*$', ctx):
        e['type'] = 'TRIỆU_CHỨNG'
```

---

## Assertion Classification

```python
def classify_assertion(start, end, text, section_type, entity_type):
    # isNegated: không, chưa, ko, không đặc hiệu...
    NEG_RE = [
        r'\bkhông\s+(?:có|còn|thấy|ghi\s+nhận|xuất\s+hiện|...)',
        r'\bchưa\s+(?:có|từng|bảo\s+giờ)',
        r'\bko\s+(?:có|còn)',
    ]
    
    # isHistorical: tiền sử, đã, trước...
    HIST_RE = [
        r'\btiền\s+sử\b',
        r'\bđã\s+(?:dùng|sử|mắc|bị|có|tiêm|uống|...)',
        r'\btrước\s+(?:khi|đây|kia)',
        r'\bcó\s+bệnh\s+từ',
    ]
    
    # isSuspected: có thể, nghi ngờ, chưa rõ...
    SUSP_RE = [
        r'\bcó\s+thể\b',
        r'\bnghi\s*ngờ\b',
        r'\bchưa\s+rõ\b',
    ]
```

Logic:
- THUỐC trong THUỐC section → isHistorical
- CHẨN_ĐOÁN trong CHẨN_ĐOÁN section (không phải "kết quả") → isHistorical
- TRIỆU_CHỨNG trong context tiền sử → isHistorical
- Context chứa "không/chưa" → isNegated
- Context chứa "có thể/nghi ngờ" → isSuspected

---

## Cấu trúc file

```
project/
├── v20_pipeline.py        # Pipeline chính (cần cải thiện)
├── drug_dict_v3.py        # Drug dictionary + RxNorm lookup
├── vocab_v5.py            # SYMPTOMS + DIAGNOSES vocabulary
├── scorer.py              # Scoring function (WER + Jaccard)
├── requirements.txt       # Dependencies
├── README.md              # File này
│
├── input/
│   └── input/
│       ├── 1.txt ... 100.txt    # Input files (100 records)
│
├── bootstrap_gt/          # Ground truth (training data)
│   └── {1..100}.json
│
└── output_v20/           # Output predictions
    └── {1..100}.json
```

---

## Chạy Pipeline

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy pipeline
python v20_pipeline.py

# Output: output_v20/ directory + output_v20.zip
```

---

## Cải thiện điểm số

### Điểm hiện tại
- Private test: **28.87** (v20)

### Hướng cải thiện tiềm năng

1. **Tăng GT Coverage** (hiện tại ~97.7%)
   - Thêm các GT items còn thiếu vào KEY_PHRASES
   - Kiểm tra vocab_v5.py có thiếu terms không

2. **Giảm Noise** (hiện tại ~379 wrong extractions)
   - Mở rộng FRAGMENT_BLACKLIST
   - Thêm DRUG_BLACKLIST entries

3. **Cải thiện Assertion Classification**
   - Tinh chỉnh NEG/HIST/SUSP regex patterns
   - Context-aware assertion classification

4. **Substring Handling**
   - Khi có "viêm gan virus" trong text
   - Cần extract đúng "viêm gan virus" thay vì "viêm gan" hoặc "virus"

5. **Section Detection**
   - Cải thiện regex patterns cho section headers
   - Xử lý edge cases (section không clear)

---

## Dependencies

```
requests
tqdm
editdistance (cho WER calculation)
```

---

## Scoring Details (scorer.py)

```python
# WER Calculation
def word_error_rate(pred_text, gt_text):
    # Split by whitespace, calculate edit distance
    pred_words = pred_text.split()
    gt_words = gt_text.split()
    return edit_distance(pred_words, gt_words) / max(len(pred_words), len(gt_words))

# Final score
text_score = 1 - WER
assertions_score = jaccard(pred_assertions, gt_assertions)
candidates_score = weighted_jaccard(pred_candidates, gt_candidates)

final_score = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score
```
