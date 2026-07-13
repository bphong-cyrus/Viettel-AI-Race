# -*- coding: utf-8 -*-
"""
LLM-based NER configuration for Vietnamese medical records.

Supports multiple backends:
- II-Medical-8B (Qwen3-8B base, medical reasoning — limited Vietnamese)
- Vistral-7B-chat (best Vietnamese)
- Qwen2.5-7B-Instruct (strong multilingual, JSON-format)
- Qwen2.5-3B-Instruct (small, runs on 6GB VRAM)
- Llama-3.1-8B-Instruct (avoid due to repetition issues)

Choose model based on hardware:
- 6GB VRAM (e.g. RTX 3050 Laptop): Qwen2.5-3B-Instruct (4-bit)
- 16GB+ VRAM: any 7-8B model in bf16
- 24GB+ VRAM: full precision / larger context

For Kaggle T4x2 (16GB each): use bf16 with 8B model OR 4-bit with 14B.
"""
import os

# === MODEL REGISTRY ===
MODEL_REGISTRY = {
    "ii-medical-8b": {
        "hf_id": "Intelligent-Internet/II-Medical-8B",
        "local_path": r"D:\projects\Viettel AI race\II-Medical-8B",
        "vietnamese_quality": "low",   # trained on English/Chinese medical reasoning
        "json_format_quality": "low",  # reasoning model, not NER
        "recommended_for_ner": False,
        "notes": "Qwen3-8B SFT on English medical QA. NOT recommended for Vietnamese NER.",
    },
    "vistral-7b-chat": {
        "hf_id": "Viet-Mistral/Vistral-7B-chat",
        "local_path": None,
        "vietnamese_quality": "high",
        "json_format_quality": "medium",
        "recommended_for_ner": True,
        "notes": "Best Vietnamese quality. Mistral-7B continued pretraining on VN corpus.",
    },
    "qwen2.5-7b-instruct": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_path": None,
        "vietnamese_quality": "good",
        "json_format_quality": "high",
        "recommended_for_ner": True,
        "notes": "Strong multilingual, follows JSON instructions well. Recommended default.",
    },
    "qwen2.5-3b-instruct": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "local_path": None,
        "vietnamese_quality": "medium",
        "json_format_quality": "high",
        "recommended_for_ner": True,
        "notes": "Fits 6GB VRAM with 4-bit quant. Good JSON following.",
    },
    "phogpt-7b5-instruct": {
        "hf_id": "vinai/PhoGPT-7B5-Instruct",
        "local_path": None,
        "vietnamese_quality": "high",
        "json_format_quality": "medium",
        "recommended_for_ner": True,
        "notes": "BLOOM-based, native Vietnamese. Format following weaker.",
    },
}

# === DEFAULT MODEL ===
# Switch this constant to change default
# II-Medical-8B is NOT used by default — it's an English/Chinese medical reasoning
# model and performs poorly on Vietnamese NER. Vistral-7B-chat has best VN quality.
DEFAULT_MODEL = "vistral-7b-chat"        # best Vietnamese quality for NER
ALT_MODEL = "qwen2.5-7b-instruct"        # strong JSON following, good VN
FALLBACK_MODEL = "qwen2.5-3b-instruct"   # when VRAM constrained (6GB)


def get_model_config(model_key: str = None):
    if model_key is None:
        model_key = DEFAULT_MODEL
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")
    cfg = MODEL_REGISTRY[model_key].copy()
    if cfg["local_path"] and os.path.exists(cfg["local_path"]):
        cfg["load_path"] = cfg["local_path"]
    else:
        cfg["load_path"] = cfg["hf_id"]
    return cfg


# === INFERENCE PARAMETERS ===
INFERENCE_PARAMS = {
    "max_new_tokens": 4096,
    "temperature": 0.0,      # greedy for deterministic NER
    "top_p": 1.0,
    "do_sample": False,
    "repetition_penalty": 1.05,  # mild, to avoid llama-style repetition
}


# === HARDWARE-AWARE LOADING ===
def get_load_kwargs(model_key: str = None, vram_gb: float = 24.0):
    """Return transformers kwargs based on available VRAM."""
    cfg = get_model_config(model_key)
    if vram_gb >= 16:
        return {
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "load_in_4bit": False,
            "load_in_8bit": False,
        }
    elif vram_gb >= 8:
        return {
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "load_in_4bit": False,
            "load_in_8bit": True,   # 8B → ~8GB
        }
    else:
        # 6GB or less: 4-bit required
        return {
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "load_in_4bit": True,
            "load_in_8bit": False,
            "quant_config": {
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            },
        }


if __name__ == "__main__":
    for k, v in MODEL_REGISTRY.items():
        rec = "✓" if v["recommended_for_ner"] else "✗"
        print(f"{rec} {k:25s} | VN:{v['vietnamese_quality']:6s} | JSON:{v['json_format_quality']:6s} | {v['notes']}")