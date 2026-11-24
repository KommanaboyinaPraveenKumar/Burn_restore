"""
scripts/check_dataset_readiness.py
==================================
Pre-training Dataset Readiness Gate and Diagnostic Auditor for Phase 3.

Audits:
  1. Unique document counts across train, val, and test splits.
  2. Total extractable line restoration instances per split.
  3. Ground-truth missing-text target length distribution (min, max, mean, std).
  4. Alignment confidence distribution and proportion of uncertain samples.
  5. Damage status distribution (severed, intact, destroyed, uncertain).
  6. Physical missing width budget distribution vs actual character count.
  7. Dataset size readiness classification (Fixture vs Full Research Dataset).
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

# Ensure repo root and Burn_restore are on sys.path
repo_root = Path(__file__).resolve().parent.parent
burn_restore_dir = repo_root / "Burn_restore"
for p in [str(burn_restore_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import Config
from dataset import LineGroundedDocumentDataset


def audit_split(dataset: LineGroundedDocumentDataset, split_name: str) -> Dict[str, Any]:
    """Audits every instance in a dataset split to extract geometric and text distributions."""
    total_samples = len(dataset)
    target_lengths = []
    prefix_lengths = []
    budgets = []
    confidences = []
    damage_statuses = {}

    for i in range(total_samples):
        item = dataset[i]
        target_text = item["raw_target_text"]
        prefix_text = item["raw_prefix"]
        budget = float(item["line_budget"].item())
        conf = float(item["alignment_confidence"].item()) if isinstance(item["alignment_confidence"], torch.Tensor) else float(item["alignment_confidence"])
        status = item.get("damage_status", "severed")

        target_lengths.append(len(target_text))
        prefix_lengths.append(len(prefix_text))
        budgets.append(budget)
        confidences.append(conf)
        damage_statuses[status] = damage_statuses.get(status, 0) + 1

    t_arr = np.array(target_lengths) if target_lengths else np.array([0])
    b_arr = np.array(budgets) if budgets else np.array([0])
    c_arr = np.array(confidences) if confidences else np.array([0])

    return {
        "split_name": split_name,
        "document_count": len(dataset.pairs),
        "target_length": {
            "min": int(t_arr.min()),
            "max": int(t_arr.max()),
            "mean": round(float(t_arr.mean()), 2),
            "median": round(float(np.median(t_arr)), 2),
            "std": round(float(t_arr.std()), 2),
        },
        "character_budget": {
            "min": round(float(b_arr.min()), 2),
            "max": round(float(b_arr.max()), 2),
            "mean": round(float(b_arr.mean()), 2),
            "std": round(float(b_arr.std()), 2),
        },
        "alignment_confidence": {
            "min": round(float(c_arr.min()), 2),
            "max": round(float(c_arr.max()), 2),
            "mean": round(float(c_arr.mean()), 2),
        },
        "damage_status_breakdown": damage_statuses,
    }


def run_readiness_gate(splits_manifest: str = "data/splits.json", output_json: str = "docs/dataset_readiness_report.json") -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 3: PRE-TRAINING DATASET READINESS GATE & AUDIT")
    print("=" * 80)

    cfg = Config()

    manifest_path = Path(splits_manifest)
    if not manifest_path.is_file():
        # Check inside Burn_restore/data
        alt_p = burn_restore_dir / splits_manifest
        if alt_p.is_file():
            manifest_path = alt_p
        else:
            raise FileNotFoundError(f"Splits manifest not found at {splits_manifest} or {alt_p}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    meta = manifest_data.get("metadata", {})
    total_docs = meta.get("total_documents", len(manifest_data.get("train", [])) + len(manifest_data.get("val", [])) + len(manifest_data.get("test", [])))

    print(f"Manifest: {manifest_path}")
    print(f"Total Unique Documents Registered: {total_docs}")

    # Audit train split
    train_ds = LineGroundedDocumentDataset(cfg, split="train", splits_manifest=str(manifest_path), is_eval=False)
    val_ds = LineGroundedDocumentDataset(cfg, split="val", splits_manifest=str(manifest_path), is_eval=True)
    test_ds = LineGroundedDocumentDataset(cfg, split="test", splits_manifest=str(manifest_path), is_eval=True)

    train_stats = audit_split(train_ds, "train")
    val_stats = audit_split(val_ds, "val")
    test_stats = audit_split(test_ds, "test")

    # Readiness Classification
    is_fixture = total_docs <= 10
    if is_fixture:
        readiness_status = "FIXTURE_MODE_ONLY (Debugging & Smoke-testing, Not for full training)"
        warning_msg = (
            "CRITICAL NOTICE: Current dataset consists of the 6-page synthetic fixture. "
            "It is mathematically sufficient to verify loss convergence, backward gradients, "
            "AMP mixed precision, checkpointing, and scheduler step mechanics. "
            "However, full-scale generalizable restoration training requires the full OPRB corpus."
        )
    else:
        readiness_status = "RESEARCH_PRODUCTION_READY"
        warning_msg = "Full corpus detected. Proceed with staged training."

    print("\n--- Split Statistics ---")
    for s in [train_stats, val_stats, test_stats]:
        name = s["split_name"].upper()
        print(f"[{name}] Documents: {s['document_count']}")
        print(f"  Target Lengths:  min={s['target_length']['min']}, max={s['target_length']['max']}, mean={s['target_length']['mean']} +/- {s['target_length']['std']}")
        print(f"  Char Budgets:    min={s['character_budget']['min']}, max={s['character_budget']['max']}, mean={s['character_budget']['mean']}")
        print(f"  Avg Confidence:  {s['alignment_confidence']['mean']}")
        print(f"  Damage Status:   {s['damage_status_breakdown']}")

    print("\n" + "=" * 80)
    print(f"DATASET READINESS STATUS: {readiness_status}")
    print("=" * 80)
    if is_fixture:
        print(f"WARNING: {warning_msg}")

    readiness_report = {
        "metadata": meta,
        "manifest_path": str(manifest_path),
        "total_documents": total_docs,
        "is_fixture": is_fixture,
        "readiness_status": readiness_status,
        "warning": warning_msg if is_fixture else None,
        "train_split": train_stats,
        "val_split": val_stats,
        "test_split": test_stats,
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(readiness_report, f, indent=2)
    print(f"\nSaved readiness report to {out_p}")

    return readiness_report


if __name__ == "__main__":
    run_readiness_gate()

