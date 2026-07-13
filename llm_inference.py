# -*- coding: utf-8 -*-
"""
LLM inference engine for Vietnamese medical NER.

Loads a model (with optional 4-bit / 8-bit quantization), runs prompt, parses JSON.
Designed to be model-agnostic: works with II-Medical-8B, Qwen2.5, Vistral, PhoGPT.

Usage:
    from llm_inference import LLMExtractor
    extractor = LLMExtractor("qwen2.5-7b-instruct", vram_gb=16)
    entities = extractor.extract(text)
"""
import os
import sys
import json
import re
import time
from typing import List, Dict, Optional

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from llm_ner_config import get_model_config, INFERENCE_PARAMS, get_load_kwargs
from llm_prompts import build_user_prompt, build_compact_prompt, build_chat_messages


class LLMExtractor:
    """Wraps a HF model for batched NER inference."""

    def __init__(self, model_key: str = None, vram_gb: float = 24.0,
                 use_compact_prompt: bool = False, use_few_shot: bool = True):
        self.model_key = model_key
        self.cfg = get_model_config(model_key)
        self.use_compact_prompt = use_compact_prompt
        self.use_few_shot = use_few_shot
        self.model = None
        self.tokenizer = None
        self._load_kwargs = get_load_kwargs(model_key, vram_gb)
        self._loaded = False

    def load(self):
        """Lazy load model. Import heavy libs only here."""
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        load_path = self.cfg["load_path"]
        print(f"Loading model from: {load_path}")
        print(f"Load kwargs: {self._load_kwargs}")

        kwargs = {"trust_remote_code": True}

        # Quantization
        if self._load_kwargs.get("load_in_4bit"):
            qcfg = self._load_kwargs.get("quant_config", {})
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=getattr(torch, qcfg.get("bnb_4bit_compute_dtype", "bfloat16")),
                bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
            )
        elif self._load_kwargs.get("load_in_8bit"):
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        dtype_str = self._load_kwargs.get("torch_dtype", "bfloat16")
        kwargs["torch_dtype"] = getattr(torch, dtype_str)
        kwargs["device_map"] = self._load_kwargs.get("device_map", "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(load_path, **kwargs)
        self.model.eval()
        self._loaded = True
        print(f"Model loaded. Device: {next(self.model.parameters()).device}")

    def _build_prompt(self, text: str) -> str:
        if self.use_compact_prompt:
            return build_compact_prompt(text)
        # Use chat template
        msgs = [
            {"role": "system", "content": self._get_system_with_examples()},
            {"role": "user", "content": build_user_prompt(text).split("--- Văn bản cần trích xuất ---")[0]
                                            + "--- Văn bản cần trích xuất ---\n" + text +
                                            "\n--- JSON output ---"},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def _get_system_with_examples(self) -> str:
        """Returns system prompt optionally with few-shot."""
        from llm_prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
        if not self.use_few_shot:
            return SYSTEM_PROMPT
        examples_str = "\n\nVÍ DỤ MẪU:"
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            examples_str += f"\n\nVăn bản {i}: {ex['text']}\nJSON: {ex['output']}"
        return SYSTEM_PROMPT + examples_str

    def extract(self, text: str, max_retries: int = 1) -> List[Dict]:
        """Extract entities from one text. Returns list of entity dicts."""
        if not text or not text.strip():
            return []
        if not self._loaded:
            self.load()

        prompt = self._build_prompt(text)
        output_text = self._generate(prompt)
        entities = self._parse_json(output_text, text)

        # Retry if parsing failed
        for _ in range(max_retries):
            if entities is not None:
                break
            output_text = self._generate(prompt, temperature=0.0)
            entities = self._parse_json(output_text, text)

        return entities or []

    def _generate(self, prompt: str, temperature: float = None) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": INFERENCE_PARAMS["max_new_tokens"],
            "do_sample": INFERENCE_PARAMS.get("do_sample", False),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = temperature > 0
        else:
            gen_kwargs["temperature"] = INFERENCE_PARAMS.get("temperature", 0.0)
            gen_kwargs["top_p"] = INFERENCE_PARAMS.get("top_p", 1.0)
        if "repetition_penalty" in INFERENCE_PARAMS:
            gen_kwargs["repetition_penalty"] = INFERENCE_PARAMS["repetition_penalty"]

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _parse_json(self, output: str, original_text: str) -> Optional[List[Dict]]:
        """Parse LLM output as JSON. Returns None if failed."""
        # Strip markdown code fences
        s = output.strip()
        if "```" in s:
            # Extract content between first ``` and last ```
            m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", s, re.DOTALL)
            if m:
                s = m.group(1)
            else:
                # Try to find first [ ... ]
                m = re.search(r"(\[.*\])", s, re.DOTALL)
                if m:
                    s = m.group(1)

        # Try direct parse
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return self._validate_entities(data, original_text)
        except json.JSONDecodeError:
            pass

        # Try to find a JSON array in the text
        m = re.search(r"\[\s*\{.*\}\s*\]", s, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    return self._validate_entities(data, original_text)
            except json.JSONDecodeError:
                pass

        return None

    def _validate_entities(self, entities: list, original_text: str) -> List[Dict]:
        """Validate and fix entity positions."""
        result = []
        seen_spans = set()
        for e in entities:
            if not isinstance(e, dict):
                continue
            text = str(e.get("text", "")).strip()
            etype = str(e.get("type", "")).strip()
            if not text or etype not in ("THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN"):
                continue

            # Fix position: try given, else locate in text
            pos = e.get("position", [0, 0])
            if not (isinstance(pos, list) and len(pos) == 2):
                pos = [0, 0]
            start, end = int(pos[0]), int(pos[1])

            # Verify position matches text
            if start < 0 or end > len(original_text) or \
               original_text[start:end].strip() != text:
                # Try to find in text
                idx = original_text.find(text)
                if idx >= 0:
                    start, end = idx, idx + len(text)
                else:
                    # Skip — we can't place it
                    continue

            # Skip if duplicate
            span = (start, end, etype)
            if span in seen_spans:
                continue
            seen_spans.add(span)

            # Normalize assertions
            assertions = e.get("assertions", [])
            if not isinstance(assertions, list):
                assertions = []
            assertions = [a for a in assertions
                          if a in ("isHistorical", "isNegated", "isSuspected")]

            # Normalize candidates
            candidates = e.get("candidates", [])
            if not isinstance(candidates, list):
                candidates = []
            candidates = [str(c) for c in candidates if c is not None]

            # Drug type without candidates → empty is OK
            # Symptom/diagnosis with candidates → strip them
            if etype != "THUỐC":
                candidates = []

            result.append({
                "text": original_text[start:end],
                "type": etype,
                "candidates": candidates,
                "assertions": assertions,
                "position": [start, end],
            })
        return result

    def extract_batch(self, texts: List[str], show_progress: bool = True) -> List[List[Dict]]:
        """Process a list of texts. Sequential (no batching yet to keep VRAM low)."""
        results = []
        t0 = time.time()
        for i, t in enumerate(texts):
            ents = self.extract(t)
            results.append(ents)
            if show_progress and (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(texts) - i - 1)
                print(f"  [{i+1}/{len(texts)}] {elapsed:.1f}s elapsed, ETA {eta:.1f}s")
        return results


def main():
    """Quick sanity check (loads model)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Model key from MODEL_REGISTRY")
    ap.add_argument("--vram", type=float, default=24.0, help="Available VRAM in GB")
    ap.add_argument("--input", default="input/input/1.txt")
    ap.add_argument("--output", default="output_llm/1.json")
    ap.add_argument("--no-few-shot", action="store_true")
    args = ap.parse_args()

    text = open(args.input, encoding="utf-8").read()
    extractor = LLMExtractor(args.model, vram_gb=args.vram, use_few_shot=not args.no_few_shot)
    extractor.load()
    entities = extractor.extract(text)
    print(f"\n=== Extracted {len(entities)} entities ===")
    for e in entities:
        print(f"  [{e['type']}] '{e['text']}' assert={e['assertions']} CUI={e['candidates']} pos={e['position']}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()