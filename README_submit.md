# Vietnamese Medical NER

NER pipeline cho bài toán Viettel AI Race.

## Bài toán

Trích xuất thực thể y tế từ hồ sơ bệnh án tiếng Việt:
- **THUỐC**: Tên thuốc
- **TRIỆU_CHỨNG**: Triệu chứng bệnh nhân
- **CHẨN_ĐOÁN**: Chẩn đoán của bác sĩ

## Output Format

```json
[
  {
    "text": "Aspirin",
    "type": "THUỐC",
    "candidates": ["N0000148261"],
    "assertions": ["isHistorical"],
    "position": [100, 107]
  }
]
```

## Chạy

```bash
pip install -r requirements.txt
python v20_pipeline.py
```

Output: `output_v20/1.json` → `output_v20/100.json`
