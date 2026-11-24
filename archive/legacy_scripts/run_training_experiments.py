"""
scripts/run_training_experiments.py
===================================
Controlled Experimentation Suite for Phase 3:
Systematic Comparison of Training Strategies and Loss Formulations.

Experiments:
  - Exp A: Generation-Only Baseline (lambda_contrast = 0.0, no staged warmup, uniform LR)
  - Exp B: Staged Training without Contrastive Loss (1 epoch adapter warmup + joint, lambda_contrast = 0.0)
  - Exp C: Staged Training with Fixed Contrastive Loss (1 epoch warmup + joint, lambda_contrast = 0.1 fixed)
  - Exp D: Staged Training with Dynamic Contrastive Annealing (lambda_contrast: 0.2 -> 0.05 linear)
  - Exp E: Full Optimization (Stage 1 Warmup + Stage 2 Differential LRs + Cosine Annealing + Dynamic Contrastive)

Results are recorded in docs/phase_3_training_experiments.json.
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root and Burn_restore are on sys.path
repo_root = Path(__file__).resolve().parent.parent
burn_restore_dir = repo_root / "Burn_restore"
for p in [str(burn_restore_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import Config
from train import train_restorer


EXPERIMENT_CONFIGS = {
    "exp_a": {
        "id": "EXP-A",
        "name": "Generation-Only Baseline",
        "description": "Cross-entropy generation loss only; no adapter warmup; single uniform learning rate; no LR decay.",
        "params": {
            "warmup_epochs": 0,
            "freeze_t5_warmup": False,
            "lambda_contrastive": 0.0,
            "contrastive_sched_type": "fixed",
            "scheduler_type": "none",
            "lr_t5_backbone": 1e-4,
            "lr_vision_backbone": 1e-4,
            "lr_adapters": 1e-4,
            "lr_warmup_adapters": 1e-4,
        },
    },
    "exp_b": {
        "id": "EXP-B",
        "name": "Staged Warmup without Contrastive Loss",
        "description": "Stage 1 adapter warmup (T5 frozen) followed by Stage 2 joint fine-tuning; lambda_contrast = 0.0.",
        "params": {
            "warmup_epochs": 1,
            "freeze_t5_warmup": True,
            "lambda_contrastive": 0.0,
            "contrastive_sched_type": "fixed",
            "scheduler_type": "cosine",
            "lr_t5_backbone": 1e-5,
            "lr_vision_backbone": 1e-5,
            "lr_adapters": 5e-5,
            "lr_warmup_adapters": 1e-4,
        },
    },
    "exp_c": {
        "id": "EXP-C",
        "name": "Staged Training with Fixed Contrastive Loss",
        "description": "Stage 1 warmup + joint fine-tuning with fixed contrastive stroke alignment weight (lambda_contrast = 0.1).",
        "params": {
            "warmup_epochs": 1,
            "freeze_t5_warmup": True,
            "lambda_contrastive": 0.1,
            "contrastive_sched_type": "fixed",
            "scheduler_type": "cosine",
            "lr_t5_backbone": 1e-5,
            "lr_vision_backbone": 1e-5,
            "lr_adapters": 5e-5,
            "lr_warmup_adapters": 1e-4,
        },
    },
    "exp_d": {
        "id": "EXP-D",
        "name": "Staged Training with Dynamic Contrastive Annealing",
        "description": "Stage 1 warmup + joint fine-tuning with linear contrastive decay (0.2 -> 0.05).",
        "params": {
            "warmup_epochs": 1,
            "freeze_t5_warmup": True,
            "lambda_contrastive": 0.1,
            "contrastive_sched_start": 0.2,
            "contrastive_sched_end": 0.05,
            "contrastive_sched_type": "linear",
            "scheduler_type": "cosine",
            "lr_t5_backbone": 1e-5,
            "lr_vision_backbone": 1e-5,
            "lr_adapters": 5e-5,
            "lr_warmup_adapters": 1e-4,
        },
    },
    "exp_e": {
        "id": "EXP-E",
        "name": "Full Multimodal Staged Optimization",
        "description": "Stage 1 Adapter Warmup + Stage 2 Differential LRs + Cosine Annealing + Dynamic Contrastive Stroke Annealing.",
        "params": {
            "warmup_epochs": 1,
            "freeze_t5_warmup": True,
            "lambda_contrastive": 0.1,
            "contrastive_sched_start": 0.2,
            "contrastive_sched_end": 0.05,
            "contrastive_sched_type": "linear",
            "scheduler_type": "cosine",
            "scheduler_warmup_ratio": 0.1,
            "lr_t5_backbone": 1e-5,
            "lr_vision_backbone": 1e-5,
            "lr_adapters": 5e-5,
            "lr_warmup_adapters": 1e-4,
        },
    },
}


def run_experiment(
    exp_key: str,
    epochs: int = 2,
    splits_manifest: str = "data/splits.json",
    output_dir: str = "checkpoints/experiments",
) -> Dict[str, Any]:
    if exp_key not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment '{exp_key}'. Choose from: {list(EXPERIMENT_CONFIGS.keys())}")

    exp_info = EXPERIMENT_CONFIGS[exp_key]
    print("\n" + "=" * 80)
    print(f"RUNNING {exp_info['id']}: {exp_info['name']}")
    print(f"Description: {exp_info['description']}")
    print("=" * 80)

    cfg = Config()
    cfg.epochs = epochs
    cfg.splits_manifest = splits_manifest
    cfg.checkpoint_dir = os.path.join(output_dir, exp_key)
    cfg.log_history_path = os.path.join(cfg.checkpoint_dir, "training_history.jsonl")

    # Apply experiment-specific parameters
    for k, v in exp_info["params"].items():
        setattr(cfg, k, v)

    start_t = time.time()
    results = train_restorer(
        config=cfg,
        splits_manifest=splits_manifest,
    )
    duration = round(time.time() - start_t, 2)

    last_record = results["history"][-1] if results["history"] else {}

    summary = {
        "experiment_id": exp_info["id"],
        "experiment_key": exp_key,
        "name": exp_info["name"],
        "description": exp_info["description"],
        "params": exp_info["params"],
        "epochs": epochs,
        "duration_sec": duration,
        "best_val_cer": results["best_val_cer"],
        "final_val_cer": last_record.get("val_cer", None),
        "final_val_wer": last_record.get("val_wer", None),
        "final_val_char_acc": last_record.get("val_char_acc", None),
        "final_val_word_acc": last_record.get("val_word_acc", None),
        "final_val_line_em": last_record.get("val_line_em", None),
        "final_train_loss": last_record.get("train_loss", None),
        "budget_violation_rate": last_record.get("budget_violation_rate", None),
        "history": results["history"],
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Controlled Training Optimization Experiments.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["exp_a", "exp_b", "exp_c", "exp_d", "exp_e"],
        help="Experiments to run (e.g. exp_a exp_e or all).",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per experiment (default 2 for smoke benchmark).")
    parser.add_argument("--splits_manifest", type=str, default="data/splits.json")
    parser.add_argument("--output_json", type=str, default="docs/phase_3_training_experiments.json")
    args = parser.parse_args()

    all_results = {}
    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing if available
    if out_file.is_file():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    for exp_k in args.experiments:
        summary = run_experiment(
            exp_key=exp_k,
            epochs=args.epochs,
            splits_manifest=args.splits_manifest,
        )
        all_results[exp_k] = summary

        # Write progressive updates
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    # Also sync to Burn_restore/docs
    alt_out = repo_root / "Burn_restore" / "docs" / "phase_3_training_experiments.json"
    alt_out.parent.mkdir(parents=True, exist_ok=True)
    with open(alt_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 3 EXPERIMENT BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<10} | {'Name':<32} | {'Val CER':<10} | {'Val WER':<10} | {'Char Acc':<10} | {'Time (s)':<8}")
    print("-" * 92)
    for k, v in all_results.items():
        cer_str = f"{v.get('best_val_cer', 0.0):.4f}" if v.get('best_val_cer') is not None else "N/A"
        wer_str = f"{v.get('final_val_wer', 0.0):.4f}" if v.get('final_val_wer') is not None else "N/A"
        cacc_str = f"{v.get('final_val_char_acc', 0.0):.4f}" if v.get('final_val_char_acc') is not None else "N/A"
        print(f"{v['experiment_id']:<10} | {v['name'][:32]:<32} | {cer_str:<10} | {wer_str:<10} | {cacc_str:<10} | {v.get('duration_sec', 0.0):<8.1f}")


if __name__ == "__main__":
    main()

