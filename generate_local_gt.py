# -*- coding: utf-8 -*-
"""
Generate local ground truth from rule-based pipeline.
This is a quick baseline - we'll improve over time.
"""
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')

from pipeline import extract_entities

INPUT_DIR = r'D:\projects\Viettel AI race\input_turn2_vong1\input'
GT_DIR = r'D:\projects\Viettel AI race\local_gt'
os.makedirs(GT_DIR, exist_ok=True)

# Run on all files
total = 0
for i in range(1, 101):
    p = f'{INPUT_DIR}/{i}.txt'
    if not os.path.exists(p):
        continue
    with open(p, 'r', encoding='utf-8') as f:
        text = f.read()
    entities = extract_entities(text)
    out_p = f'{GT_DIR}/{i}.json'
    with open(out_p, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    total += len(entities)
print(f'Total GT entities: {total}')