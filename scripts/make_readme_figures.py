"""
scripts/make_readme_figures.py
==============================
Renders the example figure used in the README: randomly drawn held-out test
lines (not cherry-picked; fixed seed) with the burn-edge crop, the true ending
split into READ / INFERRED, the full model's top-3 suggestions and the
text-only baseline's top-1. Decoding uses the calibrated upper budget in hard
mode, exactly as in scripts/evaluate_lines.py.

Also copies the reliability plot into assets/figures/.

Usage:
    python scripts/make_readme_figures.py                     # CPU is fine
    python scripts/make_readme_figures.py --full_ckpt checkpoints/v3_full/best_checkpoint.pt
"""

import argparse
import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset import LineCacheDataset  # noqa: E402
from evaluate_lines import load_model  # noqa: E402
from line_data import estimate_budget, extract_infill, split_visible  # noqa: E402
from model import load_cut_estimator, predict_cut_offsets, upper_budget  # noqa: E402


def topk(model, ds, idxs, ub, k, device):
    batch = LineCacheDataset.make_collate(ds.tok.pad_token_id)([ds[i] for i in idxs])
    lb = torch.tensor([float(ub[i]) for i in idxs], device=device)
    with torch.no_grad():
        out = model.generate_topk(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                  batch["line_burn_crop"].to(device), line_budget=lb, top_k=k,
                                  max_len=model.config.max_target_len, budget_mode="hard")
    return [[extract_infill(ds.tok.decode(o, skip_special_tokens=False)) for o in row] for row in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "lines_v3"))
    ap.add_argument("--full_ckpt", default=str(ROOT / "checkpoints" / "v3_base_full" / "best_checkpoint.pt"))
    ap.add_argument("--text_only_ckpt", default=str(ROOT / "checkpoints" / "v3_base_text_only" / "best_checkpoint.pt"))
    ap.add_argument("--cut_ckpt", default=str(ROOT / "checkpoints" / "cut_estimator.pt"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "assets" / "figures" / "examples.png"))
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    full = load_model(args.full_ckpt, dev)
    text = load_model(args.text_only_ckpt, dev)
    tok = AutoTokenizer.from_pretrained(full.config.text_model_name)
    style_full = full.config.prompt_style
    style_text = text.config.prompt_style
    ds_full = LineCacheDataset(args.data_dir, "test", tok, full.config.max_context_len, full.config.max_target_len,
                               augment=False, prompt_style=style_full)
    ds_text = LineCacheDataset(args.data_dir, "test", tok, text.config.max_context_len, text.config.max_target_len,
                               augment=False, prompt_style=style_text)
    ds_full.tok = ds_text.tok = tok
    recs = ds_full.records

    # Random draw among lines that have at least one READ character (so both parts show).
    rng = np.random.default_rng(args.seed)
    pool = [i for i, r in enumerate(recs) if int(r.get("visible_chars", 0)) >= 1]
    idxs = sorted(rng.choice(pool, size=args.n, replace=False).tolist())

    cut, meta = load_cut_estimator(args.cut_ckpt, dev)
    crops = np.asarray(ds_full.crops[idxs])
    off = predict_cut_offsets(cut, crops, dev)
    ub = {i: upper_budget(estimate_budget(recs[i]["b_x"], recs[i]["ink_x0"], recs[i]["page_w"],
                                          len(recs[i]["prefix"]), ghost_cut_px=o), meta)
          for i, o in zip(idxs, off)}

    # different token paths can decode to the same string: take 8 beams, keep 3 distinct
    c_full = [list(dict.fromkeys(c))[:3] for c in topk(full, ds_full, idxs, ub, 8, dev)]
    c_text = topk(text, ds_text, idxs, ub, 1, dev)

    fig = plt.figure(figsize=(11, 2.55 * args.n))
    gs = fig.add_gridspec(args.n, 2, width_ratios=[1, 2.3], wspace=0.04, hspace=0.35, top=0.965)
    for row, i in enumerate(idxs):
        r = recs[i]
        read, inferred = split_visible(r["target"], int(r.get("visible_chars", 0)))
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(crops[row])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"test line {i} (burn-edge crop)", fontsize=8, loc="left")
        tx = fig.add_subplot(gs[row, 1]); tx.axis("off")
        prefix = "…" + r["prefix"][-48:]
        lines = [
            ("surviving text", prefix, "#444444"),
            ("true ending", None, None),
            ("full model top-3", "  |  ".join(f"{k + 1}. {c!r}" for k, c in enumerate(c_full[row])), "#1f4e9c"),
            ("text-only top-1", repr(c_text[row][0]), "#666666"),
            ("budget", f"upper bound {ub[i]:.1f} chars (true length {len(r['target'].strip())})", "#666666"),
        ]
        y = 0.95
        for label, val, col in lines:
            tx.text(0.0, y, label, fontsize=8.5, color="#888888", va="top", family="sans-serif")
            if val is None:
                t1 = tx.text(0.2, y, read, fontsize=10, color="#0a7d2c", va="top", family="monospace",
                             fontweight="bold")
                fig.canvas.draw()
                bb = t1.get_window_extent().transformed(tx.transData.inverted())
                tx.text(bb.x1 if read else 0.2, y, inferred, fontsize=10, color="#b5531a", va="top",
                        family="monospace")
            else:
                tx.text(0.2, y, textwrap.shorten(val, 95), fontsize=9.5, color=col, va="top", family="monospace")
            y -= 0.19
    fig.suptitle("Random held-out test lines (seed %d). True ending: green = READ (visible strokes), "
                 "orange = INFERRED (burnt away)" % args.seed, fontsize=10, y=0.995)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")
    for i, cf, ct in zip(idxs, c_full, c_text):
        print(i, repr(recs[i]["target"].strip()), cf, ct)

    rel = ROOT / "results" / "line_eval_v3_base.reliability.png"
    if rel.exists():
        shutil.copy2(rel, out.parent / "reliability_v3_base.png")


if __name__ == "__main__":
    main()
