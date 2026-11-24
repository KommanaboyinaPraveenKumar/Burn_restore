"""
scripts/make_report_figures.py
==============================
Figures for the LaTeX project report (report/figures/). CPU only, no model
inference: qualitative predictions come from the `samples` stored in the
evaluation JSON files.

  fig_pipeline.png        clean page | ground-truth burn mask | burnt page
  fig_line_zoom.png       one line: clean | mask | burnt | model crop
  fig_crops.png           model crops with prefix / READ / INFERRED text
  fig_qualitative.png     test crops with truth, text-only and full predictions
  fig_read_inferred.png   READ accuracy and INFERRED CER per system
  fig_budget.png          overlong endings, old rule vs calibrated bound
  fig_ablation.png        READ accuracy with real / blank / mismatched image
  fig_training.png        validation CER per epoch and training loss
  fig_baseline.png        (if results/line_eval_baseline.json exists) OCR + LLM vs ours

Usage:  python scripts/make_report_figures.py
"""

import json
import random
import re
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "report" / "figures"
RES = ROOT / "results"

# Validated categorical slots (light mode): blue, orange, aqua; neutral gray for "other"
C1, C2, C3, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def bar_labels(ax, bars, fmt="{:.2f}", pad=3):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, pad),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK)


def grouped(ax, groups, series, colors, labels, fmt="{:.2f}", width=0.36):
    x = np.arange(len(groups))
    n = len(series)
    for k, (vals, col, lab) in enumerate(zip(series, colors, labels)):
        off = (k - (n - 1) / 2) * (width + 0.02)
        bars = ax.bar(x + off, vals, width, color=col, label=lab, zorder=2)
        bar_labels(ax, bars, fmt)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)


# --------------------------------------------------------------------------- #
# Qualitative figures from a freshly rendered held-out page
# --------------------------------------------------------------------------- #
def demo_page():
    from build_line_dataset import make_jobs, sample_layout
    from config import Config
    from dataset_generator import SideBurnGenerator
    from line_data import VISIBLE_ALPHA, burn_page_to_records, render_page

    _, words, font, seed, _ = make_jobs("test", 3, ROOT / "data" / "corpus", seed=4242, verbose=False,
                                        version="v3")[2]
    rng = random.Random(seed)
    page, _ = render_page(words, font, rng, layout=sample_layout(rng))
    gen = SideBurnGenerator(Config())
    burnt, recs, crops = burn_page_to_records(page, gen, seed=seed, appearance="v3")

    # Rebuild the exact boundary and fade the generator used (same random draws, same order)
    cfg = gen.config
    h, w = page.image_bgr.shape[:2]
    r = np.random.RandomState(seed)
    pr = random.Random(seed)
    ratio = pr.uniform(cfg.burn_min_ratio, cfg.burn_max_ratio)
    boundary = np.clip(gen._generate_jagged_boundary(h, int(w * (1 - ratio)), int(w * cfg.jagged_amplitude_ratio),
                                                     rng=r), 0, w - 1)
    T, gamma = float(r.uniform(30, 80)), float(r.uniform(0.6, 1.8))
    assert abs(T - recs[0]["burn"]["T"]) < 0.06, "burn replay drifted from the generator"
    x = np.arange(w, dtype=np.float32)[None, :]
    alpha = np.clip((x - (boundary.reshape(-1, 1) - T)) / T, 0, 1) ** gamma
    zones = np.zeros((h, w), np.uint8)              # 0 legible, 1 READ zone, 2 burnt
    zones[alpha >= 0.5] = 1
    zones[alpha >= VISIBLE_ALPHA] = 2
    return page, burnt, recs, crops, zones


def mask_rgb(zones, base_rgb=None):
    col = {0: (252, 252, 251), 1: tuple(int(C2[i:i + 2], 16) for i in (1, 3, 5)), 2: (40, 38, 36)}
    out = np.zeros(zones.shape + (3,), np.uint8)
    for k, c in col.items():
        out[zones == k] = c
    if base_rgb is not None:                         # keep the text visible under the mask
        ink = base_rgb.mean(-1) < 120
        out[ink & (zones == 0)] = (120, 120, 118)
        out[ink & (zones == 1)] = (120, 50, 20)
    return out


def fig_pipeline(page, burnt, zones):
    clean = cv2.cvtColor(page.image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for a, img, t in zip(ax, [clean, mask_rgb(zones, clean), cv2.cvtColor(burnt, cv2.COLOR_BGR2RGB)],
                         ["(a) Original page", "(b) Burn mask (ground truth)", "(c) Burnt page (model input)"]):
        a.imshow(img)
        a.set_title(t, fontsize=10, color=INK)
        a.axis("off")
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(color="#fcfcfb", ec=INK2, label="legible (prefix)"),
                          Patch(color=C2, label="faded: READ zone"),
                          Patch(color="#282624", label="burnt: INFERRED")],
                 loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=7.5)
    fig.savefig(OUT / "fig_pipeline.png")
    plt.close(fig)


def fig_line_zoom(page, burnt, zones, recs, crops):
    r = max(recs, key=lambda q: q["visible_chars"] if len(q["target"]) > 10 else 0)
    y0, y1, b = r["y0"] - 8, r["y1"] + 8, r["b_x"]
    x0, x1 = max(0, b - 420), min(page.image_bgr.shape[1], b + 120)
    clean = cv2.cvtColor(page.image_bgr, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
    strips = [clean, mask_rgb(zones[y0:y1, x0:x1], clean), cv2.cvtColor(burnt, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]]
    fig = plt.figure(figsize=(10, 3.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[3.2, 1], wspace=0.08, hspace=0.35)
    for i, (img, t) in enumerate(zip(strips, ["original line", "burn mask", "burnt line"])):
        a = fig.add_subplot(gs[i, 0])
        a.imshow(img, aspect="auto")
        a.set_title(t, fontsize=9, loc="left", color=INK)
        a.axis("off")
    a = fig.add_subplot(gs[:, 1])
    a.imshow(crops[recs.index(r)])
    a.set_title("model crop (128×256)", fontsize=9, color=INK)
    a.axis("off")
    from line_data import split_visible
    read, inf = split_visible(r["target"], r["visible_chars"])
    fig.text(0.02, -0.04, f"prefix: …{r['prefix'][-34:]!s}    READ: '{read}'    INFERRED: '{inf}'",
             fontsize=8.5, color=INK, family="DejaVu Sans Mono")
    fig.savefig(OUT / "fig_line_zoom.png")
    plt.close(fig)
    return r


def fig_crops(recs, crops, n=4):
    from line_data import split_visible
    pick = [i for i, q in enumerate(recs) if split_visible(q["target"], q["visible_chars"])[0]
            and len(q["target"].strip()) > 8][:n]
    n = len(pick)
    fig, ax = plt.subplots(n, 2, figsize=(10, 1.55 * n), gridspec_kw={"width_ratios": [1, 2.6]})
    for row, i in enumerate(pick):
        q = recs[i]
        rd, inf = split_visible(q["target"], q["visible_chars"])
        ax[row, 0].imshow(crops[i])
        ax[row, 0].axis("off")
        a = ax[row, 1]
        a.axis("off")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.text(0.0, 0.66, "…" + q["prefix"][-30:], fontsize=9, color=INK, family="DejaVu Sans Mono")
        a.text(0.0, 0.30, "READ: ", fontsize=9, color=INK2)
        a.text(0.13, 0.30, rd, fontsize=9.5, color=C2, family="DejaVu Sans Mono", weight="bold")
        a.text(0.42, 0.30, "INFERRED: ", fontsize=9, color=INK2)
        a.text(0.60, 0.30, inf, fontsize=9.5, color=INK, family="DejaVu Sans Mono")
    fig.savefig(OUT / "fig_crops.png")
    plt.close(fig)


def fig_qualitative(n=5):
    d = load("line_eval_v3_base.json")
    if not d:
        return
    crops = np.load(ROOT / "data" / "lines_v3" / "test" / "crops.npy", mmap_mode="r")
    S = [j for j, s in enumerate(d["samples"]) if s["read"] and len(s["target"]) > 8][:n]
    fig, ax = plt.subplots(len(S), 2, figsize=(10.5, 1.75 * len(S)), gridspec_kw={"width_ratios": [1, 2.9]})
    for row, j in enumerate(S):
        s = d["samples"][j]
        ax[row, 0].imshow(crops[j])
        ax[row, 0].axis("off")
        a = ax[row, 1]
        a.axis("off")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        mono = dict(family="DejaVu Sans Mono", fontsize=8.6)
        a.text(0, 0.80, "surviving: …" + s["prefix"][-36:], color=INK2, **mono)
        a.text(0, 0.56, f"truth:     {s['target']}", color=INK, weight="bold", **mono)
        a.text(0, 0.32, f"text-only: {s['text_only'][0]}", color=C1, **mono)
        a.text(0, 0.08, f"ours:      {s['full'][0]}", color=C2, **mono)
    fig.savefig(OUT / "fig_qualitative.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Charts from the evaluation JSON files
# --------------------------------------------------------------------------- #
def fig_read_inferred():
    small, base = load("line_eval_v3.json"), load("line_eval_v3_base.json")
    S, B = small["systems"], base["systems"]
    groups = ["t5-small", "t5-base"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
    grouped(ax[0], groups, [[S["text_only"]["read_acc"], B["text_only"]["read_acc"]],
                            [S["full"]["read_acc"], B["full"]["read_acc"]]], [C1, C2], ["text-only", "text + image (ours)"])
    ax[0].set_title("READ accuracy (higher is better)", fontsize=10, loc="left", color=INK)
    ax[0].set_ylim(0, 1)
    grouped(ax[1], groups, [[S["text_only"]["inferred_cer"], B["text_only"]["inferred_cer"]],
                            [S["full"]["inferred_cer"], B["full"]["inferred_cer"]]], [C1, C2], ["text-only", "text + image (ours)"])
    ax[1].set_title("INFERRED CER (lower is better)", fontsize=10, loc="left", color=INK)
    ax[1].set_ylim(0, 1)
    ax[0].legend(loc="upper left", fontsize=8)
    fig.savefig(OUT / "fig_read_inferred.png")
    plt.close(fig)


def fig_budget():
    small, base = load("line_eval_v3.json"), load("line_eval_v3_base.json")
    groups, old, new = [], [], []
    for lab, d in (("small", small), ("base", base)):
        for sysn, nm in (("text_only", "text-only"), ("full", "ours")):
            groups.append(f"{nm}\n{lab}")
            old.append(100 * d["systems"][sysn + "_soft"]["overlong_vs_truth"])
            new.append(100 * d["systems"][sysn]["overlong_vs_truth"])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    grouped(ax, groups, [old, new], [GRAY, C1], ["old rule (estimate + 2)", "calibrated upper bound"], fmt="{:.1f}%")
    ax.set_ylabel("endings > 3 chars too long (%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, max(old) * 1.25)
    fig.savefig(OUT / "fig_budget.png")
    plt.close(fig)


def fig_ablation():
    small, base = load("line_eval_v3.json"), load("line_eval_v3_base.json")
    groups = ["t5-small", "t5-base"]
    keys = [("full_soft", "real image", C2), ("full_zero_img", "blank image", GRAY), ("full_shuf_img", "mismatched image", C1)]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    grouped(ax, groups, [[small["systems"][k]["read_acc"], base["systems"][k]["read_acc"]] for k, _, _ in keys],
            [c for *_, c in keys], [lab for _, lab, _ in keys], width=0.26)
    ax.set_ylabel("READ accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUT / "fig_ablation.png")
    plt.close(fig)


def parse_log(path):
    val, loss = [], []
    pat_v = re.compile(r"\[val (\d+)/\d+\].*?val_cer: ([\d.]+)")
    pat_l = re.compile(r"\[ep (\d+)/(\d+) \| Stage \d\] step (\d+)/(\d+) \| loss: ([\d.]+)")
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat_v.search(line)
        if m:
            val.append((int(m.group(1)), float(m.group(2))))
        m = pat_l.search(line)
        if m:
            ep, _, st, tot, lo = m.groups()
            loss.append((int(ep) - 1 + int(st) / int(tot), float(lo)))
    return val, loss


def fig_training():
    runs = [("train_v3_full.log", "ours, t5-small", C2, "-"), ("train_v3_text_only.log", "text-only, t5-small", C1, "-"),
            ("train_v3_base_full.log", "ours, t5-base", C2, "--"), ("train_v3_base_text_only.log", "text-only, t5-base", C1, "--")]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.1))
    for f, lab, col, ls in runs:
        p = ROOT / "logs" / f
        if not p.exists():
            continue
        val, loss = parse_log(p)
        if val:
            ax[0].plot(*zip(*val), ls=ls, color=col, lw=2, marker="o", ms=4, label=lab)
        if loss and "full" in f:
            xs, ys = zip(*loss)
            k = 15
            ys_s = np.convolve(ys, np.ones(k) / k, mode="valid") if len(ys) > k else ys
            ax[1].plot(xs[len(xs) - len(ys_s):], ys_s, ls=ls, color=col, lw=1.6, label=lab)
    ax[0].set_title("Validation CER per epoch", fontsize=10, loc="left", color=INK)
    ax[0].set_xlabel("epoch")
    ax[0].legend(fontsize=7.5)
    ax[1].set_title("Training loss, multimodal models (smoothed)", fontsize=10, loc="left", color=INK)
    ax[1].set_xlabel("epoch")
    ax[1].legend(fontsize=7.5)
    fig.savefig(OUT / "fig_training.png")
    plt.close(fig)


def fig_baseline():
    d = load("line_eval_baseline.json")
    if not d:
        return
    S = d["systems"]
    names = [("text_only", "text-only\nt5-base", C1), ("full", "ours\nt5-base", C2),
             ("llm_only", "LLM only\nQwen2.5-1.5B", GRAY), ("ocr_llm", "OCR + LLM\nTrOCR + Qwen", C3)]
    names = [n for n in names if n[0] in S]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.1))
    for a, key, title in ((ax[0], "read_acc", "READ accuracy (higher is better)"),
                          (ax[1], "inferred_cer", "INFERRED CER (lower is better)")):
        bars = a.bar(range(len(names)), [S[k][key] for k, _, _ in names], 0.6, color=[c for *_, c in names], zorder=2)
        bar_labels(a, bars)
        a.set_xticks(range(len(names)), [lab for _, lab, _ in names], fontsize=8)
        a.set_title(title, fontsize=10, loc="left", color=INK)
        a.set_ylim(0, 1.05)
        a.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig_baseline.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    page, burnt, recs, crops, zones = demo_page()
    fig_pipeline(page, burnt, zones)
    fig_line_zoom(page, burnt, zones, recs, crops)
    fig_crops(recs, crops)
    fig_qualitative()
    fig_read_inferred()
    fig_budget()
    fig_ablation()
    fig_training()
    fig_baseline()
    for src, dst in ((RES / "line_eval_v3_base.reliability.png", "fig_reliability.png"),
                     (ROOT / "assets" / "figures" / "examples.png", "fig_examples.png"),
                     (ROOT / "assets" / "figures" / "ui_demo.png", "fig_ui_demo.png")):
        if src.exists():
            shutil.copy(src, OUT / dst)
    print("figures ->", OUT, sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
