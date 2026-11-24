import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.eval()

# Compare arbitrary custom prompt vs T5 native span prompt
prompts = [
    # 1. Arbitrary prompt (what we currently use)
    "restore line | page: Value of information in games | prefix: Value of information in noncooperative",
    # 2. T5 Native pre-trained span infilling syntax
    "Value of information in noncooperative <extra_id_0>",
    # 3. Native syntax with context
    "Game theory literature. The value of information in noncooperative <extra_id_0>",
    # 4. Another sample
    "Max Planck Institute for Mathematics in the Sciences, <extra_id_0>",
]

for p in prompts:
    inputs = tokenizer(p, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=15)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=False)
    clean_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt:     {p}")
    print(f"Raw Output: {pred}")
    print(f"Clean:      {clean_pred}")
    print("-" * 60)
