"""
dataset_split.py
================
Page-Level Document Splitting & Data Leakage Prevention Engine.

Ensures that dataset partitioning (train / validation / test) is strictly
performed at the *original document / page level* BEFORE any burn corruption
variants are synthesized.

Guarantees:
  - Zero document overlap: set(train) & set(val) & set(test) == empty set.
  - Deterministic and reproducible splits with configured random seeds.
  - Generates and saves standardized JSON manifest to 'data/splits.json'.
"""

import argparse
from datetime import datetime
import json
import os
import random
from typing import Dict, List, Optional, Set, Tuple


def get_paired_stems(image_dir: str, text_dir: str) -> List[Tuple[str, str, str]]:
    """Discovers matching (image_path, text_path, stem) tuples by stem."""
    text_files_map = {}
    for root, _, files in os.walk(text_dir):
        for fname in files:
            if fname.lower().endswith(".txt"):
                stem = os.path.splitext(fname)[0]
                text_files_map[stem] = os.path.join(root, fname)

    valid_pairs = []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                stem = os.path.splitext(fname)[0]
                text_path = text_files_map.get(stem)
                if text_path and os.path.isfile(text_path):
                    valid_pairs.append((os.path.join(root, fname), text_path, stem))

    # Sort deterministically by stem
    valid_pairs.sort(key=lambda x: x[2])
    return valid_pairs


def create_dataset_splits(
    image_dir: str,
    text_dir: str,
    output_manifest: str = "data/splits.json",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """Partitions document pages into train, val, and test splits with zero page leakage."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-4, "Split ratios must sum to 1.0"
    pairs = get_paired_stems(image_dir, text_dir)
    if not pairs:
        raise ValueError(f"No matching image-transcript pairs found between '{image_dir}' and '{text_dir}'.")

    stems = [p[2] for p in pairs]
    stem_to_pair = {p[2]: {"image_path": p[0], "text_path": p[1], "stem": p[2]} for p in pairs}

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled_stems = list(stems)
    rng.shuffle(shuffled_stems)

    n = len(shuffled_stems)
    if n < 3:
        # Mini fixtures: ensure at least 1 in train and 1 in val
        n_train = max(1, n - 1)
        n_val = 1 if n > 1 else 0
        n_test = n - n_train - n_val
    else:
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        # Ensure non-empty test if ratio > 0
        if test_ratio > 0 and n_test == 0 and n_train > 1:
            n_train -= 1
            n_test += 1

    train_stems = shuffled_stems[:n_train]
    val_stems = shuffled_stems[n_train : n_train + n_val]
    test_stems = shuffled_stems[n_train + n_val :]

    # Rigorous leakage verification
    s_train = set(train_stems)
    s_val = set(val_stems)
    s_test = set(test_stems)

    assert len(s_train & s_val) == 0, "DATA LEAKAGE DETECTED: Overlap between train and val!"
    assert len(s_train & s_test) == 0, "DATA LEAKAGE DETECTED: Overlap between train and test!"
    assert len(s_val & s_test) == 0, "DATA LEAKAGE DETECTED: Overlap between val and test!"

    manifest = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "total_documents": n,
            "counts": {
                "train": len(train_stems),
                "val": len(val_stems),
                "test": len(test_stems),
            },
        },
        "train": [stem_to_pair[s] for s in train_stems],
        "val": [stem_to_pair[s] for s in val_stems],
        "test": [stem_to_pair[s] for s in test_stems],
    }

    os.makedirs(os.path.dirname(output_manifest) or ".", exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[dataset_split] Created zero-leakage split manifest -> {output_manifest}")
    print(f"                Train: {len(train_stems)} | Val: {len(val_stems)} | Test: {len(test_stems)}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create page-level train/val/test splits.")
    parser.add_argument("--images", required=True, help="Path to clean source images.")
    parser.add_argument("--texts", required=True, help="Path to clean transcripts.")
    parser.add_argument("--out", default="data/splits.json", help="Path to output manifest.")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_dataset_splits(
        args.images,
        args.texts,
        output_manifest=args.out,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

