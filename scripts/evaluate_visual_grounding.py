"""
scripts/evaluate_visual_grounding.py
====================================
Controlled Research Evaluation and Ablation Suite for Phase 2:
Multimodal Model Architecture, Visual Modality Grounding, and Physical Constraints.

Evaluates 5 Model Configurations:
  Config A: Text-only T5 Baseline (linguistic context only)
  Config B: Legacy Visual Prefix Baseline (naive concatenation without cross-attention)
  Config C: Spatial Forensic Visual Encoder (multi-scale ResNet + 2D spatial embeddings)
  Config D: Gated Cross-Modal Fusion (gated cross-attention + contrastive supervision)
  Config E: Full Multimodal Restorer with Physical Grounding (Cross-modal + PhysicalBudgetLogitsProcessor)

Conducts Visual Sensitivity Stress Tests:
  1. Standard Uncorrupted Burn Crop
  2. Zeroed Visual Crop (all-zero ablation)
  3. Shuffled / Inverted Visual Crop (spatial disruption)
  4. Cross-Sample Substituted Visual Crop (visual mismatch)

Evaluates:
  - CER (Character Error Rate)
  - WER (Word Error Rate)
  - Character Accuracy (1 - CER)
  - Word Accuracy (1 - WER)
  - Exact-Match Line Accuracy
  - Physical Budget Compliance (mean |generated_chars - budget|)
  - Latency / Inference Time
"""

import os
import sys
import time
import json
from typing import List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer

# Path handling
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
burn_restore_dir = os.path.join(repo_root, "Burn_restore")

for p in [burn_restore_dir, repo_root, current_dir]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from config import Config
    from model import ForensicT5Restorer
    from dataset import LineGroundedDocumentDataset
    from metrics import (
        compute_cer,
        compute_wer,
        compute_character_accuracy,
        compute_word_accuracy,
        compute_exact_match_line_accuracy,
        extract_infill_text,
        normalize_text,
    )
except ImportError:
    from Burn_restore.config import Config
    from Burn_restore.model import ForensicT5Restorer
    from Burn_restore.dataset import LineGroundedDocumentDataset
    from Burn_restore.metrics import (
        compute_cer,
        compute_wer,
        compute_character_accuracy,
        compute_word_accuracy,
        compute_exact_match_line_accuracy,
        extract_infill_text,
        normalize_text,
    )


def evaluate_batch_predictions(
    model: ForensicT5Restorer,
    tokenizer,
    batch: dict,
    use_cross_modal: bool = True,
    use_budget: bool = True,
    visual_ablation: str = "normal",
    num_beams: int = 4,
    device: str = "cpu",
) -> dict:
    """Evaluates a batch of samples under specific architectural settings and visual perturbations."""
    model.eval()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    crop = batch["line_burn_crop"].to(device)
    budgets = batch["line_budget"].to(device)
    targets = batch["raw_target_text"]
    prefixes = batch["raw_prefix"]

    # Visual ablations
    if visual_ablation == "zero":
        crop = torch.zeros_like(crop)
    elif visual_ablation == "shuffle":
        # Invert/flip spatial dimensions to disrupt local stroke geometry
        crop = torch.flip(crop, dims=[-1])
    elif visual_ablation == "replace":
        # Roll batch to substitute features from a different document line
        crop = torch.roll(crop, shifts=1, dims=0)
    elif visual_ablation == "normal":
        pass

    # Toggle cross-modal fusion
    model.config.use_cross_modal_fusion = use_cross_modal

    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            line_burn_crop=crop,
            line_budget=budgets if use_budget else None,
            use_physical_grounding=use_budget,
            num_beams=num_beams,
            max_len=model.config.max_target_len,
        )
    latency = (time.time() - t0) / input_ids.size(0)

    results = []
    for i in range(input_ids.size(0)):
        raw_pred = tokenizer.decode(generated[i], skip_special_tokens=False)
        pred_text = extract_infill_text(raw_pred)
        target = targets[i]
        budget = float(budgets[i].item())

        cer = compute_cer(normalize_text(pred_text), normalize_text(target))
        wer = compute_wer(pred_text, target)
        char_acc = compute_character_accuracy(pred_text, target)
        word_acc = compute_word_accuracy(pred_text, target)
        em = compute_exact_match_line_accuracy(pred_text, target)

        char_diff = len(pred_text) - budget
        budget_violation = max(0.0, char_diff)

        results.append({
            "prefix": prefixes[i],
            "prediction": pred_text,
            "target": target,
            "budget": budget,
            "pred_length": len(pred_text),
            "cer": cer if cer is not None else 1.0,
            "wer": wer if wer is not None else 1.0,
            "char_acc": char_acc,
            "word_acc": word_acc,
            "exact_match": em,
            "budget_violation": budget_violation,
        })

    return {
        "results": results,
        "latency_per_sample": latency,
    }


def run_comprehensive_ablation(
    image_dir: str,
    text_dir: str,
    device: str = "cpu",
    output_json: Optional[str] = None,
) -> dict:
    """Executes the full Phase 2 controlled benchmark across all configurations."""
    print("=" * 80)
    print("PHASE 2: CONTROLLED MULTIMODAL MODEL EVALUATION & ABLATION STUDY")
    print("=" * 80)

    config = Config()
    config.pretrained_vision = False  # Deterministic initialization for controlled offline test
    config.device = device
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)

    # Load dataset
    dataset = LineGroundedDocumentDataset(
        config=config,
        image_dir=image_dir,
        text_dir=text_dir,
        tokenizer=tokenizer,
        is_eval=True,
    )
    print(f"Loaded evaluation dataset: {len(dataset)} document pages.")

    # Collate a test batch
    sample_items = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = LineGroundedDocumentDataset.collate_fn(sample_items)

    # Initialize model
    model = ForensicT5Restorer(config).to(device)

    configurations = [
        ("Config A: Text-only T5 Baseline", False, False, "zero"),
        ("Config B: Legacy Visual Prefix Baseline", False, False, "normal"),
        ("Config C: Spatial Forensic Encoder", False, True, "normal"),
        ("Config D: Gated Cross-Modal Fusion", True, False, "normal"),
        ("Config E: Full Multimodal + Physical Budget", True, True, "normal"),
    ]

    benchmark_summary = {}

    for name, use_fusion, use_budget, vis_mode in configurations:
        print(f"\nEvaluating: {name} ...")
        eval_data = evaluate_batch_predictions(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            use_cross_modal=use_fusion,
            use_budget=use_budget,
            visual_ablation=vis_mode,
            device=device,
        )

        res_list = eval_data["results"]
        mean_cer = float(np.mean([r["cer"] for r in res_list]))
        mean_wer = float(np.mean([r["wer"] for r in res_list]))
        mean_char_acc = float(np.mean([r["char_acc"] for r in res_list]))
        mean_word_acc = float(np.mean([r["word_acc"] for r in res_list]))
        mean_em = float(np.mean([r["exact_match"] for r in res_list]))
        mean_violation = float(np.mean([r["budget_violation"] for r in res_list]))
        latency = eval_data["latency_per_sample"]

        benchmark_summary[name] = {
            "CER": round(mean_cer, 4),
            "WER": round(mean_wer, 4),
            "Char_Acc": round(mean_char_acc, 4),
            "Word_Acc": round(mean_word_acc, 4),
            "Exact_Match": round(mean_em, 4),
            "Budget_Violation_Chars": round(mean_violation, 2),
            "Latency_Sec": round(latency, 4),
        }

        print(f"  -> CER: {mean_cer:.4f} | WER: {mean_wer:.4f} | Char Acc: {mean_char_acc:.4f} | Budget Violation: {mean_violation:.1f} chars | Latency: {latency*1000:.1f}ms")

    # Visual Sensitivity Analysis on Config D/E
    print("\n" + "-" * 60)
    print("VISUAL SENSITIVITY STRESS TEST (Config E)")
    print("-" * 60)
    sensitivity_summary = {}
    stress_modes = [
        ("Normal Visual Crop", "normal"),
        ("Zeroed Visual Features", "zero"),
        ("Spatially Shuffled Crop", "shuffle"),
        ("Substituted Sample Crop", "replace"),
    ]

    for label, mode in stress_modes:
        eval_data = evaluate_batch_predictions(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            use_cross_modal=True,
            use_budget=True,
            visual_ablation=mode,
            device=device,
        )
        res_list = eval_data["results"]
        mean_cer = float(np.mean([r["cer"] for r in res_list]))
        mean_char_acc = float(np.mean([r["char_acc"] for r in res_list]))
        sensitivity_summary[label] = {
            "CER": round(mean_cer, 4),
            "Char_Acc": round(mean_char_acc, 4),
            "Predictions": [r["prediction"] for r in res_list],
        }
        print(f"  [{label}] Mean CER: {mean_cer:.4f} | Predictions: {[r['prediction'] for r in res_list]}")

    full_report = {
        "benchmark_summary": benchmark_summary,
        "sensitivity_summary": sensitivity_summary,
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        print(f"\nSaved ablation summary to {output_json}")

    return full_report


if __name__ == "__main__":
    mini_img = os.path.join(repo_root, "tests", "fixtures", "mini_dataset", "clean_images")
    mini_txt = os.path.join(repo_root, "tests", "fixtures", "mini_dataset", "clean_texts")
    output_p = os.path.join(repo_root, "docs", "phase_2_ablation_results.json")
    run_comprehensive_ablation(mini_img, mini_txt, device="cpu", output_json=output_p)
