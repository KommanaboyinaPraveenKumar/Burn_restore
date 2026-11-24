"""
scripts/add_estimated_budget.py
===============================
Adds ``budget_est`` to every record of a line cache: the character-budget point
estimate computed with the *predicted* cut position (CutOffsetRegressor,
scripts/train_cut_estimator.py) instead of the fixed GHOST_CUT_PX offset.
Prompt style "v3e" puts this number in the ``length:`` hint.

The original records.jsonl is copied to records_pre_budget_est.jsonl first
(only if that backup does not exist yet).

Usage:
    python scripts/add_estimated_budget.py --data_dir data/lines_v3
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from line_data import estimate_budget  # noqa: E402
from model import load_cut_estimator, predict_cut_offsets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "lines_v3"))
    ap.add_argument("--cut_ckpt", default=str(ROOT / "checkpoints" / "cut_estimator.pt"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_cut_estimator(args.cut_ckpt, dev)
    if model is None:
        sys.exit(f"cut estimator not found: {args.cut_ckpt} (run scripts/train_cut_estimator.py)")

    for split in args.splits:
        d = Path(args.data_dir) / split
        rec_path = d / "records.jsonl"
        backup = d / "records_pre_budget_est.jsonl"
        if not backup.exists():
            shutil.copy2(rec_path, backup)
        recs = [json.loads(l) for l in open(rec_path, encoding="utf-8")]
        crops = np.load(d / "crops.npy", mmap_mode="r")
        assert len(recs) == len(crops), f"{split}: records / crops mismatch"
        off = predict_cut_offsets(model, crops, dev)
        L = np.array([len(r["target"].strip()) for r in recs], float)
        for r, o in zip(recs, off):
            r["cut_offset_pred"] = round(float(o), 2)
            r["budget_est"] = estimate_budget(r["b_x"], r["ink_x0"], r["page_w"], len(r["prefix"]),
                                              ghost_cut_px=float(o))
        tmp = rec_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        tmp.replace(rec_path)
        old = np.array([r["budget"] for r in recs]); new = np.array([r["budget_est"] for r in recs])
        print(f"{split}: {len(recs)} records | length-hint MAE vs truth: fixed cut {np.abs(L - old).mean():.3f} "
              f"-> predicted cut {np.abs(L - new).mean():.3f} chars")


if __name__ == "__main__":
    main()
