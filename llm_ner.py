"""
LLM-based NER extraction using Ollama.
Outputs JSON format.
"""
import json, os, sys, time, re
sys.stdout.reconfigure(encoding='utf-8')
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b-instruct"


PROMPT_TEMPLATE = """Bạn là chuyên gia y tế trích xuất thực thể có tên (NER) trên văn bản y khoa tiếng Việt.

Hãy trích xuất TẤT CẢ các thực thể:
- "THUỐC": tên thuốc cụ thể (ví dụ: aspirin, acetaminophen 500mg, trimetazidin, Vastarel, intravenous fluids, kháng sinh)
- "TRIỆU_CHỨNG": triệu chứng bệnh (ví dụ: đau đầu, sốt cao, run tay, yếu sức, mệt mỏi)
- "CHẨN_ĐOÁN": bệnh hoặc chẩn đoán (ví dụ: viêm phổi, tăng huyết áp, Parkinson, tiểu đường)

Trích xuất ÍT NHẤT 5-15 thực thể từ văn bản.

QUY TẮC:
- Mỗi thực thể phải là cụm từ CÓ THẬT trong văn bản (copy nguyên văn)
- Text phải từ 2 đến 50 ký tự
- Trả về JSON ARRAY, mỗi phần tử có: text, type, assertions
- assertions: list rỗng hoặc chứa một trong: "isHistorical", "isNegated", "isSuspected"

VĂN BẢN:
{text}

CHỈ TRẢ LỜI BẰNG JSON ARRAY. Ví dụ:
[{{"text":"đau đầu","type":"TRIỆU_CHỨNG","assertions":["isHistorical"]}},{{"text":"Parkinson","type":"CHẨN_ĐOÁN","assertions":[]}}]

JSON:"""


def call_ollama(prompt: str, max_tokens: int = 1500) -> str:
    """Call Ollama with the prompt."""
    data = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.1,
        }
    }).encode('utf-8')

    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode('utf-8'))
        return result.get('response', '')


def parse_llm_response(response: str) -> list:
    """Parse LLM JSON response, handle non-JSON gracefully."""
    # Try to find JSON array in response
    response = response.strip()
    # Extract from ```json blocks
    m = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response)
    if m:
        response = m.group(1)
    # Find first [ to last ]
    start = response.find('[')
    end = response.rfind(']')
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(response[start:end+1])
    except json.JSONDecodeError:
        return []


def process_file(text: str) -> list:
    """Process a single file with LLM."""
    # Truncate to ~4000 chars to avoid context overflow
    text_truncated = text[:3500] if len(text) > 3500 else text

    prompt = PROMPT_TEMPLATE.format(text=text_truncated)

    try:
        response = call_ollama(prompt)
        entities = parse_llm_response(response)
    except Exception as e:
        print(f'  ERROR: {e}', file=sys.stderr)
        return []

    return entities


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', default=r'D:\projects\Viettel AI race\input_turn2_vong1\input')
    ap.add_argument('--output_dir', default=r'D:\projects\Viettel AI race\llm_gt')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=100)
    ap.add_argument('--limit', type=int, default=None, help='Limit number of files for testing')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = []
    for i in range(args.start, args.end + 1):
        p = f'{args.input_dir}/{i}.txt'
        if os.path.exists(p):
            files.append((i, p))

    if args.limit:
        files = files[:args.limit]

    print(f'Processing {len(files)} files...')
    t0 = time.time()
    total_entities = 0
    for idx, (i, p) in enumerate(files):
        with open(p, 'r', encoding='utf-8') as f:
            text = f.read()
        entities = process_file(text)
        # Add position info by finding text in original
        for e in entities:
            t = e.get('text', '')
            pos = text.lower().find(t.lower())
            if pos >= 0:
                e['position'] = [pos, pos + len(t)]
            else:
                e['position'] = [0, 0]
            e.setdefault('candidates', [])
        out_p = f'{args.output_dir}/{i}.json'
        with open(out_p, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        total_entities += len(entities)
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(files) - idx - 1) / rate
            print(f'  [{idx+1}/{len(files)}] {elapsed:.0f}s, ETA {eta:.0f}s, entities={total_entities}')

    elapsed = time.time() - t0
    print(f'Done: {total_entities} entities in {elapsed:.0f}s')
