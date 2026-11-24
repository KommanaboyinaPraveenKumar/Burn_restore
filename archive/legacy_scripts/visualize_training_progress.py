"""
scripts/visualize_training_progress.py
======================================
Post-Training Qualitative Restoration Inspector and Diagnostic Error Analyzer.

Performs:
  1. Multi-sample evaluation across mild, moderate, and severe burn profiles.
  2. Side-by-side comparison:
       Prefix Context | Physical Budget | Ground Truth Target | Restored Prediction
  3. Fine-grained character error analysis:
       - Insertions, Deletions, Substitutions
       - Over-generation vs Under-generation
       - Physical budget violation check
  4. Exports structured evaluation report to docs/qualitative_restoration_report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

# Ensure repo root and Burn_restore are on sys.path
repo_root = Path(__file__).resolve().parent.parent
burn_restore_dir = repo_root / "Burn_restore"
for p in [str(burn_restore_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import Config
from dataset import LineGroundedDocumentDataset
from model import ForensicT5Restorer
from metrics import (
    compute_cer,
    compute_wer,
    compute_character_accuracy,
    compute_word_accuracy,
    compute_exact_match_line_accuracy,
    extract_infill_text,
    normalize_text,
)


def detailed_char_error_breakdown(pred: str, target: str) -> Dict[str, Any]:
    """Computes exact counts of insertions, deletions, substitutions, and length bias."""
    p_norm = normalize_text(pred)
    t_norm = normalize_text(target)
    n, m = len(p_norm), len(t_norm)

    # Dynamic programming matrix for alignment traceback
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if p_norm[i - 1] == t_norm[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)

    # Traceback
    i, j = n, m
    insertions = 0
    deletions = 0
    substitutions = 0
    matches = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] and p_norm[i - 1] == t_norm[j - 1]:
            matches += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            insertions += 1
            i -= 1
        else:
            deletions += 1
            j -= 1

    length_diff = n - m
    bias_category = "balanced"
    if length_diff > 2:
        bias_category = "over_generated"
    elif length_diff < -2:
        bias_category = "under_generated"

    return {
        "pred_length": n,
        "target_length": m,
        "length_diff": length_diff,
        "bias_category": bias_category,
        "matches": matches,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "total_edits": substitutions + insertions + deletions,
    }


def run_inspection(
    checkpoint_path: str,
    splits_manifest: str = "data/splits.json",
    split: str = "val",
    output_report: str = "docs/qualitative_restoration_report.json",
    device_str: str = "cpu",
) -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 3: QUALITATIVE RESTORATION INSPECTION & DIAGNOSTICS")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split:      {split} (from {splits_manifest})")
    print("=" * 80)

    cfg = Config()
    cfg.splits_manifest = splits_manifest
    device = torch.device(device_str)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    dataset = LineGroundedDocumentDataset(
        config=cfg,
        tokenizer=tokenizer,
        split=split,
        splits_manifest=splits_manifest,
        is_eval=True,
    )

    model = ForensicT5Restorer(cfg).to(device)
    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"[inspect] Loaded weights from {checkpoint_path}")
    else:
        print(f"[inspect] Checkpoint '{checkpoint_path}' not found! Evaluating initialized model.")

    model.eval()

    sample_reports = []
    total_samples = len(dataset)

    all_cer = []
    all_wer = []
    all_cacc = []
    all_wacc = []
    all_em = []
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    over_gen_count = 0
    under_gen_count = 0
    budget_violations = 0

    print(f"\nEvaluating {total_samples} samples:\n")

    for idx in range(total_samples):
        item = dataset[idx]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        line_crop = item["line_burn_crop"].unsqueeze(0).to(device)
        line_budget = item["line_budget"].unsqueeze(0).to(device)

        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                line_burn_crop=line_crop,
                line_budget=line_budget,
                max_len=cfg.max_target_len,
            )

        raw_decoded = tokenizer.decode(gen_tokens[0], skip_special_tokens=False)
        pred_text = extract_infill_text(raw_decoded)
        target_text = item["raw_target_text"]
        prefix_text = item["raw_prefix"]
        budget_val = float(item["line_budget"].item())

        cer = compute_cer(pred_text, target_text)
        wer = compute_wer(pred_text, target_text)
        cacc = compute_character_accuracy(pred_text, target_text)
        wacc = compute_word_accuracy(pred_text, target_text)
        em = compute_exact_match_line_accuracy(pred_text, target_text)
        breakdown = detailed_char_error_breakdown(pred_text, target_text)

        is_budget_violated = len(pred_text) > (budget_val + cfg.budget_margin_buffer)
        if is_budget_violated:
            budget_violations += 1

        all_cer.append(cer if cer is not None else 1.0)
        all_wer.append(wer if wer is not None else 1.0)
        all_cacc.append(cacc)
        all_wacc.append(wacc)
        all_em.append(em)

        total_insertions += breakdown["insertions"]
        total_deletions += breakdown["deletions"]
        total_substitutions += breakdown["substitutions"]
        if breakdown["bias_category"] == "over_generated":
            over_gen_count += 1
        elif breakdown["bias_category"] == "under_generated":
            under_gen_count += 1

        safe_p = prefix_text.encode("ascii", "replace").decode("ascii")
        safe_t = target_text.encode("ascii", "replace").decode("ascii")
        safe_r = pred_text.encode("ascii", "replace").decode("ascii")

        print(f"--- Sample {idx + 1}/{total_samples} [{item.get('stem', 'doc')}] ---")
        print(f"  Surviving Prefix:   {safe_p!r}")
        print(f"  Physical Budget:    {budget_val:.1f} chars (actual: {len(target_text)} chars)")
        print(f"  Ground Truth:       {safe_t!r}")
        print(f"  Restored Output:    {safe_r!r}")
        print(f"  Metrics:            CER={cer:.4f} | WER={wer:.4f} | CharAcc={cacc:.2%} | BudgetOK={not is_budget_violated}")
        print(f"  Error Breakdown:    Matches={breakdown['matches']}, Sub={breakdown['substitutions']}, Ins={breakdown['insertions']}, Del={breakdown['deletions']}\n")

        sample_reports.append({
            "sample_idx": idx,
            "stem": item.get("stem", "doc"),
            "prefix": prefix_text,
            "ground_truth": target_text,
            "restored_output": pred_text,
            "character_budget": budget_val,
            "is_budget_violated": is_budget_violated,
            "cer": cer,
            "wer": wer,
            "char_acc": cacc,
            "word_acc": wacc,
            "exact_match": em,
            "error_breakdown": breakdown,
        })

    summary = {
        "checkpoint": checkpoint_path,
        "split": split,
        "total_samples": total_samples,
        "macro_metrics": {
            "mean_cer": round(float(np.mean(all_cer)), 4),
            "median_cer": round(float(np.median(all_cer)), 4),
            "mean_wer": round(float(np.mean(all_wer)), 4),
            "mean_char_acc": round(float(np.mean(all_cacc)), 4),
            "mean_word_acc": round(float(np.mean(all_wacc)), 4),
            "exact_match_line_rate": round(float(np.mean(all_em)), 4),
            "budget_violation_rate": round(budget_violations / max(1, total_samples), 4),
        },
        "aggregate_error_breakdown": {
            "total_insertions": total_insertions,
            "total_deletions": total_deletions,
            "total_substitutions": total_substitutions,
            "over_generated_samples": over_gen_count,
            "under_generated_samples": under_gen_count,
            "balanced_samples": total_samples - (over_gen_count + under_gen_count),
        },
        "samples": sample_reports,
    }

    out_p = Path(output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    alt_p = repo_root / "Burn_restore" / "docs" / Path(output_report).name
    alt_p.parent.mkdir(parents=True, exist_ok=True)
    with open(alt_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print(f"REPORT SAVED: {output_report}")
    print(f"Macro CER: {summary['macro_metrics']['mean_cer']:.4f} | Char Acc: {summary['macro_metrics']['mean_char_acc']:.2%}")
    print("=" * 80)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Qualitative Visual Restoration Inspector.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_checkpoint.pt")
    parser.add_argument("--splits_manifest", type=str, default="data/splits.json")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_report", type=str, default="docs/qualitative_restoration_report.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_inspection(
        checkpoint_path=args.checkpoint,
        splits_manifest=args.splits_manifest,
        split=args.split,
        output_report=args.output_report,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()

