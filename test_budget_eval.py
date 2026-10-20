import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import Config
from dataset import SideBurnDocumentDataset
from model import ForensicT5Restorer
from metrics import compute_cer

def safe_str(s):
    if not isinstance(s, str):
        s = str(s)
    return s.encode("ascii", "replace").decode("ascii")

def test_constrained_decoding():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "oprb_train", "img")
    txt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "oprb_train", "cleaned_texts")
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "final_checkpoint.pt")

    dataset = SideBurnDocumentDataset(cfg, img_dir, txt_dir, tokenizer=tokenizer, is_eval=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=SideBurnDocumentDataset.collate_fn)

    model = ForensicT5Restorer(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model.eval()

    unconstrained_cer = []
    constrained_cer = []
    targets_len = []
    unconstrained_len = []
    constrained_len = []

    count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            gen_unconstrained = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                line_burn_crop=batch["line_burn_crop"],
                max_len=cfg.max_target_len,
            )
            preds_uncon = [tokenizer.decode(g, skip_special_tokens=True) for g in gen_unconstrained]

            budgets = batch["line_budget"].cpu().numpy()
            targets = batch["raw_target_text"]
            prefixes = batch["raw_prefix"]

            for i in range(len(preds_uncon)):
                pred_u = preds_uncon[i]
                target = targets[i]
                budget_chars = float(budgets[i])
                
                # Constrain by physical character budget with buffer (+ 20% or +3 chars)
                max_allowed_chars = max(4, int(budget_chars * 1.25) + 3)
                pred_c = pred_u[:max_allowed_chars].rstrip()

                cer_u = compute_cer(pred_u, target)
                cer_c = compute_cer(pred_c, target)

                unconstrained_cer.append(cer_u)
                constrained_cer.append(cer_c)
                targets_len.append(len(target))
                unconstrained_len.append(len(pred_u))
                constrained_len.append(len(pred_c))

                if count < 10:
                    print(f"Sample {count+1}:", flush=True)
                    print(f"  Prefix:      {safe_str(prefixes[i][:45])}...", flush=True)
                    print(f"  Target:      '{safe_str(target)}' (len {len(target)})", flush=True)
                    print(f"  Uncon Pred:  '{safe_str(pred_u)}' (len {len(pred_u)}) | CER: {cer_u:.4f}", flush=True)
                    print(f"  Budget Pred: '{safe_str(pred_c)}' (len {len(pred_c)}) | CER: {cer_c:.4f}", flush=True)
                    print(f"  Char Budget: {budget_chars:.1f} -> Max Allowed: {max_allowed_chars}", flush=True)
                    print("-" * 60, flush=True)
                count += 1

            if count >= 30:
                break

    print(f"\n--- Diagnostic Summary (30 samples) ---", flush=True)
    print(f"Avg Target Length:        {np.mean(targets_len):.1f} chars", flush=True)
    print(f"Avg Unconstrained Length: {np.mean(unconstrained_len):.1f} chars", flush=True)
    print(f"Avg Budget Length:        {np.mean(constrained_len):.1f} chars", flush=True)
    print(f"Mean Unconstrained CER:   {np.mean(unconstrained_cer):.4f}", flush=True)
    print(f"Mean Budget CER:          {np.mean(constrained_cer):.4f}", flush=True)
    gain = (np.mean(unconstrained_cer) - np.mean(constrained_cer)) / np.mean(unconstrained_cer) * 100
    print(f"Relative CER Reduction:   {gain:.2f}%", flush=True)

if __name__ == "__main__":
    test_constrained_decoding()
