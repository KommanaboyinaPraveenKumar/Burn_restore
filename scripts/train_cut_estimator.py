"""
scripts/train_cut_estimator.py
==============================
Trains ``CutOffsetRegressor`` (model.py) to predict, from the burn-edge crop,
how far left of the burn boundary the legible text ends (the 50%-opacity cut).
Then calibrates the *upper* character budget on the validation split:

    estimate = estimate_budget(b_x, ink_x0, page_w, len(prefix), ghost_cut_px=predicted offset)
    upper    = ub_ratio * estimate + ub_add

with ub_ratio chosen so that ``--coverage`` (default 99%) of the true missing
endings fit.  The point estimate is a median-type guess (glyph widths of the
missing text are unknown), so it must not be used as a hard ceiling; the upper
bound is what decoding enforces and what "budget violation" is measured against.

Usage:
    python scripts/train_cut_estimator.py --data_dir data/lines_v3 --out checkpoints/cut_estimator.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from line_data import GHOST_CUT_PX, alpha_x, crop_to_tensor, estimate_budget  # noqa: E402
from model import CutOffsetRegressor, predict_cut_offsets  # noqa: E402


def load_split(data_dir: Path, split: str):
    recs = [json.loads(l) for l in open(data_dir / split / "records.jsonl", encoding="utf-8")]
    return recs, data_dir / split / "crops.npy"


def true_offset(r: dict) -> float:
    if "burn" in r and r["burn"]:
        return float(r["b_x"] - alpha_x(r["b_x"], r["burn"]["T"], r["burn"]["gamma"], 0.5))
    return float(GHOST_CUT_PX)  # v2: fixed linear fade


class CropOffsets(Dataset):
    def __init__(self, crops_path, offsets, idx):
        self.path, self.off, self.idx, self.crops = str(crops_path), offsets, idx, None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.crops is None:
            self.crops = np.load(self.path, mmap_mode="r")
        j = self.idx[i]
        return crop_to_tensor(np.asarray(self.crops[j])), torch.tensor(self.off[j], dtype=torch.float32)


def budgets(recs, offsets):
    return np.array([estimate_budget(r["b_x"], r["ink_x0"], r["page_w"], len(r["prefix"]), ghost_cut_px=o)
                     for r, o in zip(recs, offsets)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "lines_v3"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "cut_estimator.pt"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--coverage", type=float, default=0.99)
    ap.add_argument("--ub_add", type=float, default=1.0)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Path(args.data_dir)

    tr, tr_crops = load_split(d, "train")
    y = np.array([true_offset(r) for r in tr], np.float32)
    print(f"train {len(tr)} crops; true cut offset mean {y.mean():.1f}px  sd {y.std():.1f}  "
          f"(fixed GHOST_CUT_PX = {GHOST_CUT_PX})")
    dl = DataLoader(CropOffsets(tr_crops, y, np.arange(len(tr))), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)
    model = CutOffsetRegressor().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl))
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for step, (x, t) in enumerate(dl, 1):
            x, t = x.to(dev, non_blocking=True), t.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                pred = model(x)
            loss = F.smooth_l1_loss(pred.float(), t, beta=2.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
            if step % 200 == 0:
                print(f"  ep {ep + 1} step {step}/{len(dl)} loss {run / 200:.3f}  ({time.time() - t0:.0f}s)", flush=True)
                run = 0.0

    report = {}
    model.eval()
    ub_ratio = None
    for split in ("val", "test"):
        recs, cp = load_split(d, split)
        crops = np.load(cp, mmap_mode="r")
        off_true = np.array([true_offset(r) for r in recs])
        off_pred = predict_cut_offsets(model, crops, dev)
        L = np.array([len(r["target"].strip()) for r in recs], float)
        b_nom, b_pred, b_true = budgets(recs, [GHOST_CUT_PX] * len(recs)), budgets(recs, off_pred), budgets(recs, off_true)
        if split == "val":
            # smallest ratio such that `coverage` of true endings fit the upper bound
            ub_ratio = float(np.quantile((L - args.ub_add) / np.maximum(b_pred, 1.0), args.coverage))
            ub_ratio_nom = float(np.quantile((L - args.ub_add) / np.maximum(b_nom, 1.0), args.coverage))
        ub = ub_ratio * b_pred + args.ub_add
        ub_nom = ub_ratio_nom * b_nom + args.ub_add
        report[split] = {
            "n": len(recs),
            "offset_mae_px": {"fixed": round(float(np.abs(off_true - GHOST_CUT_PX).mean()), 2),
                              "predicted": round(float(np.abs(off_true - off_pred).mean()), 2)},
            "estimate_mae_chars": {"fixed_cut": round(float(np.abs(L - b_nom).mean()), 3),
                                   "predicted_cut": round(float(np.abs(L - b_pred).mean()), 3),
                                   "oracle_cut": round(float(np.abs(L - b_true).mean()), 3)},
            "truth_exceeds_old_rule(b+2)": round(float((L > b_nom + 2).mean()), 4),
            "truth_within_upper": round(float((L <= ub).mean()), 4),
            "mean_upper_slack_chars": round(float((ub - L).mean()), 2),
            "nominal_cut_upper_slack_chars": round(float((ub_nom - L).mean()), 2),
        }
        print(split, json.dumps(report[split]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "ub_ratio": ub_ratio, "ub_add": args.ub_add,
                "coverage": args.coverage, "data_dir": str(args.data_dir), "report": report}, out)
    (out.with_suffix(".json")).write_text(json.dumps({"ub_ratio": ub_ratio, "ub_add": args.ub_add,
                                                      "coverage": args.coverage, "report": report}, indent=2))
    print(f"ub_ratio {ub_ratio:.3f} (nominal-cut equivalent {ub_ratio_nom:.3f}); saved {out}")


if __name__ == "__main__":
    main()
