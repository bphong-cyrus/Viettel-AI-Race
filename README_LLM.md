# Vietnamese Medical NER — LLM Extension

Bổ sung cho pipeline v20: thêm khả năng dùng LLM chuyên dụng để tăng chất lượng trích xuất thực thể y tế tiếng Việt.

## 📊 Tình hình hiện tại

| Pipeline | Test set thật (BTC) | bootstrap_gt (100 mẫu) |
|----------|---------------------|-------------------------|
| v20 (rule-based) | **28.87** | 0.83 |

Vấn đề chính: `text_score` thấp (1-WER = 0.66) → rules miss nhiều triệu chứng/chẩn đoán, WER cao.

## 🤖 Đánh giá II-Medical-8B

**KHÔNG phù hợp** cho task này vì:
- Base = Qwen3-8B, fine-tune trên medical reasoning **tiếng Anh/Trung** (MedMCQA, MedQA, PubMedQA, HealthBench)
- Được train cho **multiple-choice Q&A + chain-of-thought**, không phải **token-level NER**
- Token tiếng Việt không được học kỹ

## ✅ Models đề xuất (xếp theo độ phù hợp)

| Model | Size | Tiếng Việt | JSON format | Ghi chú |
|---|---|---|---|---|
| **Vistral-7B-chat** | 7B | Tốt nhất | Trung bình | Mistral + continue-pretrain VN corpus |
| **Qwen2.5-7B-Instruct** | 7B | Tốt | Tốt nhất | Multilingual mạnh, theo JSON tốt ⭐ default |
| **Qwen2.5-3B-Instruct** | 3B | OK | Tốt | Fit 6GB VRAM với 4-bit |
| **PhoGPT-7B5-Instruct** | 7.5B | Native | Trung bình | BLOOM base |

## 🏗️ Kiến trúc

```
┌──────────────────────────────────────────────────────────────┐
│                     INPUT: hồ sơ bệnh án                      │
└──────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                ▼                            ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│  BRANCH A: v20 Rules     │    │  BRANCH B: LLM (Qwen2.5-7B)  │
│  - Drug dictionary       │    │  - Prompt + few-shot         │
│  - Verb patterns         │    │  - JSON output                │
│  - Section vocab         │    │  - Position validation        │
│  - KEY_PHRASES           │    │  - Hallucination cleanup      │
│  - Assertion classifier  │    │                               │
└──────────┬───────────────┘    └──────────────┬───────────────┘
           │                                   │
           └─────────────┬─────────────────────┘
                         ▼
              ┌──────────────────────┐
              │  MERGER (IoU ≥ 0.5)  │
              │  - Rule wins on      │
              │    overlap (drug CUI)│
              │  - LLM adds missing  │
              │  - LLM enriches      │
              │    assertions        │
              └──────────┬───────────┘
                         ▼
                ┌─────────────────┐
                │  OUTPUT (JSON)  │
                └─────────────────┘
```

## 📁 Cấu trúc file mới

| File | Mục đích |
|------|---------|
| `llm_ner_config.py` | Registry model + hardware-aware loading |
| `llm_prompts.py` | System prompt + few-shot examples + parser |
| `llm_inference.py` | LLMExtractor class — load + extract + parse JSON |
| `hybrid_pipeline.py` | Kết hợp rules + LLM (merge với IoU threshold) |
| `sft_train_lora.py` | LoRA/QLoRA fine-tuning trên bootstrap_gt |
| `evaluate.py` | Eval harness trên bootstrap_gt |
| `run.py` | Unified CLI: `rules` / `llm` / `hybrid` / `eval` / `compare` |
| `kaggle_sft_train.ipynb` | Notebook train trên Kaggle |
| `kaggle_inference.ipynb` | Notebook inference + submit trên Kaggle |

## 🚀 Quick start

### 1. Rules-only baseline (chạy được trên máy hiện tại)

```bash
python run.py rules
python run.py eval --pred_dir output_v20
```

Kết quả trên bootstrap_gt: **0.83** final score.

### 2. Test prompt building (no GPU)

```python
from llm_prompts import build_user_prompt
text = "Bệnh nhân nam 65 tuổi, tiền sử tăng huyết áp, đang dùng amlodipine 5mg."
prompt = build_user_prompt(text)
print(prompt)
```

### 3. LLM inference (cần GPU ≥16GB)

```bash
# Trên máy có GPU mạnh (Linux/Kaggle/colab)
python run.py llm --model qwen2.5-7b-instruct --vram 16
python run.py eval --pred_dir output_llm
```

### 4. Hybrid (rules + LLM)

```bash
python run.py llm --model qwen2.5-7b-instruct --vram 16 --with_rules
python run.py eval --pred_dir output_llm
```

### 5. So sánh nhiều runs

```bash
python run.py compare --runs output_v20 output_llm
```

### 6. SFT fine-tuning trên Kaggle

Upload `bootstrap_gt/` + code files làm Kaggle dataset, sau đó:

```python
# Trong kaggle_sft_train.ipynb
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
USE_QLORA = True
EPOCHS = 5
# Run all cells → adapter saved to /kaggle/working/lora_adapter
```

### 7. Inference với LoRA adapter

```bash
python run.py llm --model qwen2.5-7b-instruct --vram 16 \
    --lora_path ./lora_adapter
```

## 🔧 Hardware matrix

| VRAM | Model | Quantization | Tốc độ ước tính |
|------|-------|--------------|-----------------|
| 6GB | Qwen2.5-3B-Instruct | 4-bit | ~5-10s/sample |
| 8GB | Qwen2.5-7B-Instruct | 8-bit | ~8-15s/sample |
| 16GB | Qwen2.5-7B-Instruct / Vistral-7B | bf16 | ~10-20s/sample |
| 16GB (Kaggle T4) | Qwen2.5-7B-Instruct | 4-bit (QLoRA) | ~12-25s/sample |
| 24GB+ | Bất kỳ 7-8B model | bf16 | ~5-10s/sample |

## 📈 Metrics kỳ vọng (sau SFT)

- **text_score**: 0.66 → 0.78+ (LLM extract chính xác hơn)
- **assertions_score**: 0.82 → 0.85+ (LLM tốt hơn ở negated/suspected)
- **candidates_score**: 0.96 → giữ nguyên (rules đã rất tốt cho CUI)
- **Final**: 0.83 → 0.87+ trên bootstrap_gt, tương ứng **35-40+** trên test BTC

## ⚠️ Lưu ý quan trọng cho metric BTC

Từ đề bài:
> "Trong trường hợp đoán đúng phần text của khái niệm nhưng sai loại, khái niệm sẽ bị tính 2 lần"

→ Sai type cực kỳ tốn điểm. Cần rule "viêm X trong context triệu chứng" → TRIỆU_CHỨNG (đã có trong v20 step 6).

→ LLM phải được fine-tune để output chính xác 3 loại, không hallucinate "BỆNH", "XÉT_NGHIỆM" etc.

## 📋 Workflow đề xuất

1. **Baseline (đã có)**: `python run.py rules` → ~28.87
2. **LLM zero-shot**: `python run.py llm --model qwen2.5-7b-instruct` (Kaggle)
3. **Hybrid**: `python run.py llm --with_rules` (Kaggle)
4. **SFT fine-tune**: train LoRA trên Kaggle (kaggle_sft_train.ipynb)
5. **LLM + LoRA**: `python run.py llm --lora_path ./lora_adapter` (Kaggle)
6. **Submit**: download `output_hybrid.zip` từ Kaggle

## 🐛 Debug tips

- LLM output không phải JSON: bật verbose, kiểm tra `repetition_penalty`, thử `temperature=0.0`
- LLM hallucinate type: kiểm tra system prompt có rõ ràng 3 loại không
- LLM sai position: dùng `_validate_entities` để tự relocate bằng `text.find()`
- Hybrid score thấp hơn rules-only: giảm `rule_weight` từ 0.6 xuống 0.4 (tin LLM hơn)