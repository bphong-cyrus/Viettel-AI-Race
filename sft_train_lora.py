# -*- coding: utf-8 -*-
"""
LoRA / QLoRA fine-tuning script for Vietnamese medical NER.

Designed to run on Kaggle (T4x2 16GB VRAM each) or any GPU with 16GB+ VRAM.
Reads bootstrap_gt/*.json, builds instruction-tuning dataset, trains LoRA adapter
on top of base model (Qwen2.5-7B / Vistral-7B / etc), saves adapter weights.

Usage (Kaggle):
    !python sft_train_lora.py \\
        --base_model Qwen/Qwen2.5-7B-Instruct \\
        --data_dir /kaggle/input/vtai-race/bootstrap_gt \\
        --output_dir /kaggle/working/lora_adapter \\
        --epochs 5 --lr 1e-4 --lora_r 16

For 4-bit (QLoRA) on 16GB GPU:
        --qlora

For local 6GB GPU (e.g. RTX 3050), need to use very small batch + gradient checkpointing.
"""
import os
import sys
import json
import argparse
import re
from typing import List, Dict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')


# ============================================================================
# DATA PREP — convert bootstrap_gt/*.json into SFT examples
# ============================================================================
def build_sft_examples(data_dir: str, max_examples: int = None) -> List[Dict]:
    """Convert bootstrap_gt JSONs into instruction/output SFT pairs.

    Each example is: {input: <text>, output: <JSON array string>}

    The output uses the same schema as the inference prompt expects.
    """
    examples = []
    data_path = Path(data_dir)
    for gt_file in sorted(data_path.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 999):
        try:
            gt = json.load(open(gt_file, encoding="utf-8"))
        except Exception as e:
            print(f"Skip {gt_file.name}: {e}")
            continue
        if not gt:
            continue

        # The text and positions are in GT, but we need the original text
        # to verify positions. Since we don't have input dir here, we rely on
        # positions being correct in GT itself.
        # Reconstruct approximate "input" — the input dir is provided separately.
        # For SFT, we just need a placeholder — actual training can use synthetic text
        # OR we need to load the original input text.
        # NOTE: caller's job to pass actual texts via --input_dir.

        # Store raw GT for later
        examples.append({
            "id": gt_file.stem,
            "entities": gt,
        })

    if max_examples:
        examples = examples[:max_examples]
    return examples


def build_sft_with_text(gt_dir: str, input_dir: str) -> List[Dict]:
    """Build SFT examples pairing input text with GT JSON output."""
    gt_path = Path(gt_dir)
    in_path = Path(input_dir)
    examples = []

    for gt_file in sorted(gt_path.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 999):
        sid = gt_file.stem
        in_file = in_path / f"{sid}.txt"
        if not in_file.exists():
            print(f"Skip {sid}: no input file")
            continue
        try:
            text = in_file.read_text(encoding="utf-8")
            gt = json.load(open(gt_file, encoding="utf-8"))
        except Exception as e:
            print(f"Skip {sid}: {e}")
            continue

        if not gt:
            continue

        # Validate positions: keep only entities with valid positions
        valid = []
        for e in gt:
            pos = e.get("position", [0, 0])
            if len(pos) != 2:
                continue
            s, en = pos
            if s < 0 or en > len(text):
                continue
            # Fix text to match slice (sometimes GT has typo)
            actual = text[s:en]
            if actual.strip() != e["text"].strip():
                # Try to find the text
                idx = text.find(e["text"])
                if idx >= 0:
                    e["position"] = [idx, idx + len(e["text"])]
                else:
                    continue
            valid.append(e)

        if not valid:
            continue

        # Build output JSON matching inference schema
        out_entities = []
        for e in valid:
            out_entities.append({
                "text": e["text"],
                "type": e["type"],
                "candidates": e.get("candidates", []),
                "assertions": e.get("assertions", []),
                "position": e["position"],
            })

        examples.append({
            "id": sid,
            "input_text": text,
            "output_json": json.dumps(out_entities, ensure_ascii=False),
        })

    return examples


# ============================================================================
# PROMPT BUILDING — same as inference
# ============================================================================
def build_prompt_and_response(example: Dict, use_few_shot_in_training: bool = False):
    """Convert an SFT example into (prompt, response) strings.

    Prompt = system + user message asking for extraction
    Response = JSON output
    """
    from llm_prompts import SYSTEM_PROMPT, build_user_prompt

    text = example["input_text"]
    response = example["output_json"]

    # Use same prompt format as inference
    user_msg = build_user_prompt(text)
    return user_msg, response


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF model ID or local path")
    ap.add_argument("--gt_dir", default="bootstrap_gt")
    ap.add_argument("--input_dir", default="input/input")
    ap.add_argument("--output_dir", default="output_lora")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--qlora", action="store_true", help="Use 4-bit quantization (QLoRA)")
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--save_steps", type=int, default=20)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---- Imports (after parse to keep --help fast) ----
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from datasets import Dataset
    from transformers import BitsAndBytesConfig

    print(f"=== SFT LoRA Training ===")
    print(f"Base model: {args.base_model}")
    print(f"GT dir: {args.gt_dir}")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"4-bit (QLoRA): {args.qlora}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    # ---- Build dataset ----
    print("\nBuilding SFT examples...")
    examples = build_sft_with_text(args.gt_dir, args.input_dir)
    print(f"Built {len(examples)} examples")

    if len(examples) == 0:
        print("No examples — exiting")
        return

    # ---- Tokenizer + model ----
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Add chat template if needed
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )

    print("\nLoading model...")
    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16, "device_map": "auto"}

    if args.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ---- LoRA config ----
    # Target all linear layers in attention + MLP for best adaptation
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Tokenize dataset ----
    print("\nTokenizing...")
    def tokenize_fn(ex):
        user_msg, response = build_prompt_and_response(ex)
        # Build full chat-format string
        messages = [
            {"role": "system", "content": _get_system_text()},
            {"role": "user", "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = prompt + response + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False, truncation=True,
                            max_length=args.max_seq_len)["input_ids"]

        # Mask prompt tokens in labels
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:len(full_ids)]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }

    def _get_system_text():
        from llm_prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
        examples_str = "\n\nVÍ DỤ MẪU:"
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            examples_str += f"\n\nVăn bản {i}: {ex['text']}\nJSON: {ex['output']}"
        return SYSTEM_PROMPT + examples_str

    ds = Dataset.from_list(examples)
    ds = ds.map(tokenize_fn, remove_columns=ds.column_names, num_proc=1)

    print(f"Tokenized dataset size: {len(ds)}")
    print(f"Sample lengths: {[len(ds[i]['input_ids']) for i in range(min(3, len(ds)))]}")

    # ---- Data collator (pad to longest) ----
    def collator(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids = f["input_ids"]
            lbls = f["labels"]
            am = f["attention_mask"]
            pad = max_len - len(ids)
            input_ids.append(ids + [tokenizer.pad_token_id] * pad)
            labels.append(lbls + [-100] * pad)
            attn.append(am + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    # ---- Training arguments ----
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="epoch",
        eval_strategy="no",
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
        bf16=True,
        report_to="none",
        seed=args.seed,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )

    print("\nStarting training...")
    trainer.train()

    # ---- Save ----
    print(f"\nSaving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()