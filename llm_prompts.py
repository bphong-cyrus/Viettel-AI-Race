# -*- coding: utf-8 -*-
"""
Prompt templates for Vietnamese medical NER via LLM.

Three-tier prompt strategy:
1. SYSTEM: domain rules (types, assertion semantics, JSON contract)
2. FEW-SHOT: 2-3 high-quality examples covering main cases
3. INSTRUCTION: target text + output format constraint

Critical design choices:
- Ask for exact byte offsets (not character offsets) — scorer uses positions for tie-break
- Constrain to 3 types only — no hallucinated "BỆNH", "XÉT_NGHIỆM" etc.
- Output JSON array (not dict-of-lists) — matches required schema
- Few-shot uses REAL bootstrap_gt samples so format/style matches grader
- Vietnamese instructions (the model sees the prompt in VN too)
"""
import json

# ============================================================================
# SYSTEM PROMPT — explains task, types, assertions, JSON contract
# ============================================================================
SYSTEM_PROMPT = """Bạn là chuyên gia trích xuất thực thể y tế từ hồ sơ bệnh án tiếng Việt.

NHIỆM VỤ: Trích xuất TẤT CẢ các thực thể y tế trong văn bản và trả về JSON array.

CÁC LOẠI THỰC THỂ (chỉ được dùng 3 loại này):
1. "THUỐC" — Tên thuốc (kèm liều dùng nếu có). VD: "aspirin 325mg x 1", "metoprolol 25mg po bid"
2. "TRIỆU_CHỨNG" — Triệu chứng bệnh nhân. VD: "đau bụng", "khó thở", "sốt cao", "buồn nôn"
3. "CHẨN_ĐOÁN" — Bệnh/chẩn đoán của bác sĩ. VD: "đái tháo đường", "viêm phổi", "suy tim"

ASSERTIONS (thuộc tính trạng thái):
- "isHistorical": bệnh/thuốc trong quá khứ, tiền sử, đã/đang dùng từ trước
- "isNegated": bị phủ định (không, chưa, ko có, …)
- "isSuspected": nghi ngờ, có thể, chưa rõ, đang theo dõi
- Có thể có nhiều assertions cùng lúc (VD: "không có tiền sử" → isHistorical + isNegated)
- Nếu không có assertion nào, để mảng rỗng []

CANDIDATES (mã RxNorm cho THUỐC):
- Chỉ áp dụng cho loại THUỐC
- Mảng string chứa mã RxNorm CUI. VD: ["243670"] cho aspirin
- Nếu không chắc chắn, để mảng rỗng []

POSITION (offset byte trong văn bản gốc):
- "position": [start, end] — chỉ số byte (KHÔNG phải ký tự) trong TEXT gốc
- Lấy bằng cách đếm vị trí ký tự (vì văn bản tiếng Việt thường là UTF-8 1-3 byte/ký tự, nhưng scorer chấp nhận character offset)
- QUAN TRỌNG: position[0] phải trỏ đúng vào ký tự đầu tiên của thực thể, position[1] trỏ vào ký tự ngay sau ký tự cuối
- text trong output phải khớp chính xác với text[position[0]:position[1]] trong văn bản gốc

QUY TẮC QUAN TRỌNG:
- text phải KHỚP CHÍNH XÁC với chuỗi con trong văn bản (giữ nguyên chữ hoa/thường, dấu cách)
- Mỗi lần xuất hiện của thực thể phải được liệt kê riêng (trừ khi thực sự trùng position)
- Với THUỐC trong section "thuốc trước khi nhập viện" / "thuốc đang dùng" → thường là isHistorical
- Với CHẨN_ĐOÁN trong section "tiền sử" / "chẩn đoán" → thường là isHistorical
- Với câu có "không", "chưa", "ko" gần thực thể → isNegated
- "viêm X" trong câu mô tả triệu chứng ("vì viêm X", "cho viêm X") → TRIỆU_CHỨNG
- "viêm X" trong chẩn đoán xác định → CHẨN_ĐOÁN
- KHÔNG trích xuất: thuật ngữ chung chung ("bệnh", "triệu chứng"), đơn vị ("mg", "ml"), câu hoàn chỉnh

ĐỊNH DẠNG OUTPUT:
Trả về DUY NHẤT một JSON array. Mỗi phần tử:
{
  "text": "...",
  "type": "THUỐC" | "TRIỆU_CHỨNG" | "CHẨN_ĐOÁN",
  "candidates": ["..."] | [],
  "assertions": ["..."] | [],
  "position": [start, end]
}

KHÔNG thêm giải thích, KHÔNG thêm markdown code block, KHÔNG thêm text thừa trước/sau JSON.
"""


# ============================================================================
# FEW-SHOT EXAMPLES — picked to cover main cases (assertions, drug vs symptom,
# overlapping terms like "viêm gan", historical context)
# ============================================================================
FEW_SHOT_EXAMPLES = [
    {
        "text": "Tiền sử: đái tháo đường type 2, tăng huyết áp. Hiện tại: đau ngực, khó thở nhẹ. Không sốt, không ho.",
        "output": json.dumps([
            {"text": "đái tháo đường type 2", "type": "CHẨN_ĐOÁN", "candidates": [], "assertions": ["isHistorical"], "position": [8, 29]},
            {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "candidates": [], "assertions": ["isHistorical"], "position": [31, 44]},
            {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "candidates": [], "assertions": [], "position": [56, 64]},
            {"text": "khó thở nhẹ", "type": "TRIỆU_CHỨNG", "candidates": [], "assertions": [], "position": [66, 77]},
            {"text": "sốt", "type": "TRIỆU_CHỨNG", "candidates": [], "assertions": ["isNegated"], "position": [86, 89]},
            {"text": "ho", "type": "TRIỆU_CHỨNG", "candidates": [], "assertions": ["isNegated"], "position": [97, 99]}
        ], ensure_ascii=False)
    },
    {
        "text": "Thuốc đang dùng: metformin 500mg po bid, aspirin 81mg po daily. Nghi ngờ viêm phổi, có thể suy tim.",
        "output": json.dumps([
            {"text": "metformin 500mg po bid", "type": "THUỐC", "candidates": ["6809"], "assertions": ["isHistorical"], "position": [16, 39]},
            {"text": "aspirin 81mg po daily", "type": "THUỐC", "candidates": ["243670"], "assertions": ["isHistorical"], "position": [41, 62]},
            {"text": "viêm phổi", "type": "CHẨN_ĐOÁN", "candidates": [], "assertions": ["isSuspected"], "position": [74, 83]},
            {"text": "suy tim", "type": "CHẨN_ĐOÁN", "candidates": [], "assertions": ["isSuspected"], "position": [89, 96]}
        ], ensure_ascii=False)
    },
]


def build_user_prompt(text: str, max_chars: int = 3500) -> str:
    """Build the user prompt with few-shot examples + target text.

    Truncates input if too long to stay within context budget.
    For 8B model with 4k context, max ~3500 chars leaves room for prompt + output.
    """
    examples_text = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += f"\n\n--- Ví dụ {i} ---\nVĂN BẢN: {ex['text']}\nJSON: {ex['output']}"

    # Truncate if needed, keeping beginning and end
    if len(text) > max_chars:
        half = max_chars // 2 - 50
        text = text[:half] + "\n... [văn bản bị cắt giữa] ...\n" + text[-half:]

    user_prompt = f"""Trích xuất thực thể y tế từ văn bản sau theo đúng schema đã học.
{examples_text}

--- Văn bản cần trích xuất ---
{text}

--- JSON output ---"""
    return user_prompt


def build_chat_messages(text: str, use_few_shot: bool = True):
    """Return list of messages for chat template."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if use_few_shot:
        msgs.append({"role": "user", "content": build_user_prompt("").strip()})  # empty placeholder
        # Few-shot is embedded in system prompt instead for simplicity
    msgs.append({"role": "user", "content": build_user_prompt(text)})
    return msgs


# Compact prompt for token-constrained settings (no few-shot, just instructions)
def build_compact_prompt(text: str, max_chars: int = 3000) -> str:
    """Build a single-turn prompt (no chat template). Works with base models too."""
    if len(text) > max_chars:
        half = max_chars // 2 - 50
        text = text[:half] + "\n... [truncated] ...\n" + text[-half:]

    return f"""{SYSTEM_PROMPT}

VĂN BẢN:
{text}

JSON:"""


if __name__ == "__main__":
    # Test build
    sample = "Bệnh nhân nam 65 tuổi, tiền sử tăng huyết áp, đang dùng amlodipine 5mg. Hiện tại đau ngực, khó thở."
    prompt = build_user_prompt(sample)
    print("=== USER PROMPT ===")
    print(prompt)
    print(f"\n=== Length: {len(prompt)} chars ===")