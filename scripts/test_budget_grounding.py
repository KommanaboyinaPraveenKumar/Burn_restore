import torch
import json
import cv2
import numpy as np
from pathlib import Path
from model import ForensicT5Restorer
from config import Config
from dataset_generator import SideBurnGenerator
from metrics import extract_infill_text, normalize_text

cfg = Config()
model = ForensicT5Restorer(cfg)
ckpt = torch.load('checkpoints/best_checkpoint.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

with open('data/splits.json') as f:
    splits = json.load(f)

test_item = splits['test'][0]
img_path = test_item['image_path']
txt_path = test_item['text_path']

img = cv2.imread(img_path)
img = cv2.resize(img, cfg.page_resize)
h, w = img.shape[:2]

with open(txt_path, 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()]

gen = SideBurnGenerator(cfg)
burn_res = gen.apply_burn(img, lines, seed=42)
severed = burn_res['severed_lines']

tok = model._get_tokenizer()

print(f'=== Testing Decoder-Level Physical Grounding on Test Page ({len(severed)} severed lines) ===')
for idx, s in enumerate(severed[:5]):
    prefix = s['prefix']
    target = s['target']
    b_x = s['b_x']
    missing_px = float(w - b_x)
    budget = missing_px / cfg.avg_char_width_px
    budget_tensor = torch.tensor([budget], dtype=torch.float32)

    prompt = f"Incomplete line: {prefix} <extra_id_0>"
    enc = tok(prompt, return_tensors='pt')
    crop_tensor = torch.zeros(1, 3, 128, 256)

    # Generate with grounding
    gen_grounded = model.generate(
        input_ids=enc['input_ids'],
        attention_mask=enc['attention_mask'],
        line_burn_crop=crop_tensor,
        line_budget=budget_tensor,
        use_physical_grounding=True,
        max_len=30,
        num_beams=1
    )
    # Generate without grounding
    gen_ungrounded = model.generate(
        input_ids=enc['input_ids'],
        attention_mask=enc['attention_mask'],
        line_burn_crop=crop_tensor,
        line_budget=budget_tensor,
        use_physical_grounding=False,
        max_len=30,
        num_beams=1
    )

    txt_g = extract_infill_text(tok.decode(gen_grounded[0], skip_special_tokens=False))
    txt_u = extract_infill_text(tok.decode(gen_ungrounded[0], skip_special_tokens=False))

    print(f'Line {idx}: Budget={budget:.1f} chars | Target: \"{target}\" (len {len(target)})')
    print(f'  Ungrounded: \"{txt_u}\" (len {len(txt_u)}) -> Excess: {len(txt_u) - budget:.1f}')
    print(f'  Grounded:   \"{txt_g}\" (len {len(txt_g)}) -> Excess: {len(txt_g) - budget:.1f}')

