"""
scripts/evaluate_lines.py
=========================
Held-out evaluation on the line cache test split (books never seen in training).

The target of every severed line is split into two parts (see
``line_data.split_visible``):

  READ      leading characters whose strokes are still faintly visible in the
            crop -> a recognition problem; vision *should* recover these.
  INFERRED  characters that are completely burnt -> only guessable from
            context; there is no single correct answer, so top-k recall,
            the likelihood the model gives the true text (bits/char) and
            calibration matter more than single-reference CER.

Systems
-------
  empty            predicts "" (CER = 1.0 by definition)
  t5_zero_shot     pretrained t5-small, same prompt, no fine-tuning, no vision
  text_only        fine-tuned text-only checkpoint           (--text_only_ckpt)
  full             fine-tuned multimodal checkpoint          (--full_ckpt)
  full_zero_img    full model, crop replaced by zeros        (vision ablation)
  full_shuf_img    full model, crops shuffled within a batch (mismatched evidence)
  <name>           any extra checkpoint                      (--extra_ckpt NAME=PATH)
  *_ocr            text_only / full with OCR-style noise in the context
                   (--context_noise P, robustness to imperfect prefixes)

Each model is fed the prompt style it was trained with (v2 or v3).  Models
with a CTC reading head also get a vision-only reading score.

Calibration (--calib_n N): the uncalibrated confidence is the model's own
probability of the first word of its top answer.  A logistic (Platt) model on
[log P(first word), top beam score, top-1/top-2 margin] predicting whether the
first word is correct is fitted on N
validation lines and applied to the test split; ECE / Brier / reliability bins
are reported before and after.

Usage:
    python scripts/evaluate_lines.py --data_dir data/lines_v3 \
        --full_ckpt checkpoints/v3_full/best_checkpoint.pt \
        --text_only_ckpt checkpoints/v3_text_only/best_checkpoint.pt --calib_n 1000
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from dataset import LineCacheDataset  # noqa: E402
from line_data import estimate_budget, extract_infill, split_visible  # noqa: E402
from metrics import _levenshtein, compute_wer  # noqa: E402
from model import (ForensicT5Restorer, apply_saved_architecture, load_cut_estimator,  # noqa: E402
                   predict_cut_offsets, upper_budget)


def load_model(ckpt_path: Optional[str], device, text_only: bool = False,
               text_model: Optional[str] = None) -> ForensicT5Restorer:
    cfg = Config()
    cfg.pretrained_vision = False
    ckpt = None
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        apply_saved_architecture(cfg, ckpt.get("config_dict", {}))
    if text_model:
        cfg.text_model_name = text_model
    if text_only:
        cfg.use_vision = False
        cfg.use_cross_modal_fusion = False
        cfg.use_reading_head = False
    model = ForensicT5Restorer(cfg)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.to(device).eval()


def _image(crop: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(crop)
    if mode == "shuffle":
        return crop[torch.roll(torch.arange(crop.size(0)), 1)]
    return crop


@torch.no_grad()
def run_system(model, loader, tok, device, image_mode: str = "real", k: int = 5, ub=None,
               budget_mode: Optional[str] = None) -> Dict[str, list]:
    """Top-k candidates, their beam scores, per-line NLL (nats) of the truth under
    teacher forcing, and (if available) the vision-only CTC reading."""
    cands, scores, nll, ctc, fw_logp = [], [], [], [], []
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    has_head = getattr(model, "reading_head", None) is not None and model.config.use_vision
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        crop = _image(batch["line_burn_crop"].to(device), image_mode)
        labels = batch["labels"].to(device)
        # decoding budget: the calibrated upper bound if available, else the record's estimate
        lb = (torch.tensor([float(ub[i]) for i in batch["record_idx"]]) if ub is not None
              else batch["line_budget"]).to(device)
        with torch.autocast(**amp):
            out, sc = model.generate_topk(ids, mask, crop, line_budget=lb, top_k=k,
                                          max_len=model.config.max_target_len, return_scores=True,
                                          budget_mode=budget_mode)
            _, logits = model(input_ids=ids, attention_mask=mask, line_burn_crop=crop, labels=labels,
                              lambda_contrastive=0.0)
            if has_head:
                ctc.extend(model.read_crop(crop))
        tok_nll = F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
        nll.extend(tok_nll.sum(1).tolist())
        batch_c = []
        for row, srow in zip(out, sc.float().cpu().tolist()):
            batch_c.append([extract_infill(tok.decode(o, skip_special_tokens=False)) for o in row])
            scores.append(srow)
        cands.extend(batch_c)
        fw_logp.extend(first_word_logprob(model, tok, ids, mask, crop, [c[0] for c in batch_c], amp))
    return {"cands": cands, "scores": scores, "nll": nll, "ctc": ctc, "fw_logp": fw_logp}


@torch.no_grad()
def first_word_logprob(model, tok, ids, mask, crop, top1: List[str], amp) -> List[float]:
    """Model's own log-probability of the first word of its top-1 answer
    (teacher-forced, no logits processor) - the uncalibrated confidence."""
    words = [(c.split()[:1] or [""])[0] for c in top1]
    sent = tok.convert_tokens_to_ids("<extra_id_0>")
    seqs = [[sent] + tok(w, add_special_tokens=False)["input_ids"] for w in words]
    L = max(len(q) for q in seqs)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long)
    for i, q in enumerate(seqs):
        lab[i, :len(q)] = torch.tensor(q)
    lab = lab.to(ids.device)
    with torch.autocast(**amp):
        _, logits = model(input_ids=ids, attention_mask=mask, line_burn_crop=crop, labels=lab,
                          lambda_contrastive=0.0)
    lp = -F.cross_entropy(logits.float().transpose(1, 2), lab, ignore_index=-100, reduction="none")
    lp[:, 0] = 0.0  # the sentinel itself is not part of the word
    return lp.sum(1).tolist()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def auroc(conf: np.ndarray, correct: np.ndarray) -> Optional[float]:
    pos, neg = conf[correct > 0], conf[correct == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def conf_features(res: Dict[str, list]) -> np.ndarray:
    """[log P(first word), top beam score, top1-top2 beam margin]"""
    s = np.array(res["scores"], float)
    margin = s[:, 0] - s[:, 1] if s.shape[1] > 1 else np.zeros(len(s))
    return np.stack([np.array(res["fw_logp"], float), s[:, 0], margin], 1)


def fit_platt(X: np.ndarray, y: np.ndarray, iters: int = 50, l2: float = 1e-3) -> np.ndarray:
    """Logistic regression (Newton's method) with standardised features."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + l2 * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return np.r_[w, mu, sd]


def apply_platt(params: np.ndarray, X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    w, mu, sd = params[:d + 1], params[d + 1:2 * d + 1], params[2 * d + 1:]
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    return 1 / (1 + np.exp(-Z @ w))


def calibration_report(p: np.ndarray, y: np.ndarray, bins: int = 10) -> Dict:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece, table = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
            table.append([round(float(edges[b]), 2), int(m.sum()), round(float(p[m].mean()), 4),
                          round(float(y[m].mean()), 4)])
    return {"ece": round(float(ece), 4), "brier": round(float(np.mean((p - y) ** 2)), 4),
            "bins[lo,n,conf,acc]": table}


def first_word_correct(cands: List[List[str]], raw_targets: List[str]) -> np.ndarray:
    return np.array([(c[0].split()[:1] == t.strip().split()[:1]) if t.strip().split() else (not c[0].strip())
                     for c, t in zip(cands, raw_targets)], float)


def line_stats(res: Dict[str, list], raw_targets: List[str], vis: List[int], crop_texts: List[str],
               budgets: List[float], buffer: float, upper: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
    top1 = [c[0] for c in res["cands"]]
    tgt = [t.strip() for t in raw_targets]
    parts = [split_visible(t, v) for t, v in zip(raw_targets, vis)]
    n = len(tgt)
    st: Dict[str, np.ndarray] = {}
    st["edits"] = np.array([_levenshtein(p, t) for p, t in zip(top1, tgt)], float)
    st["lens"] = np.array([max(len(t), 1) for t in tgt], float)
    st["em"] = np.array([p.strip() == t for p, t in zip(top1, tgt)], float)
    st["first_word"] = first_word_correct(res["cands"], raw_targets)
    st["wer"] = np.array([compute_wer(p, t) or 0.0 for p, t in zip(top1, tgt)], float)
    # violation of the calibrated upper bound (if any); the old rule (estimate + buffer) is kept for comparison
    st["viol_old"] = np.array([len(p) > b + buffer for p, b in zip(top1, budgets)], float)
    st["viol"] = (np.array([len(p) > u for p, u in zip(top1, upper)], float) if upper is not None
                  else st["viol_old"])
    st["len_err"] = np.array([len(p) - len(t) for p, t in zip(top1, tgt)], float)

    # READ part: lines whose target starts with >=1 visible character
    st["has_read"] = np.array([len(r) > 0 for r, _ in parts], float)
    st["read_ok"] = np.array([p.startswith(r) if r else 0.0 for p, (r, _) in zip(top1, parts)], float)
    st["read_edits"] = np.array([_levenshtein(p[:len(r)], r) if r else 0.0 for p, (r, _) in zip(top1, parts)], float)
    st["read_lens"] = np.array([len(r) for r, _ in parts], float)
    st["read_ok_at_k"] = np.array([any(c.startswith(r) for c in cs) if r else 0.0
                                   for cs, (r, _) in zip(res["cands"], parts)], float)
    # Vision-only READ via the CTC head: does the transcription end with the READ chars?
    if res.get("ctc"):
        st["ctc_read_ok"] = np.array([ct.rstrip().endswith(r) if r else 0.0
                                      for ct, (r, _) in zip(res["ctc"], parts)], float)
        st["ctc_edits"] = np.array([_levenshtein(ct, gt) for ct, gt in zip(res["ctc"], crop_texts)], float)
        st["ctc_lens"] = np.array([max(len(gt), 1) for gt in crop_texts], float)
    # INFERRED part: positional approximation - prediction after the read span
    st["inf_edits"] = np.array([_levenshtein(p[len(r):].strip(), f) for p, (r, f) in zip(top1, parts)], float)
    st["inf_lens"] = np.array([max(len(f), 1) for _, f in parts], float)
    st["has_inf"] = np.array([len(f) > 0 for _, f in parts], float)

    # Top-k (hypothesis set) metrics
    st["em_at_k"] = np.array([any(c.strip() == t for c in cs) for cs, t in zip(res["cands"], tgt)], float)
    st["oracle_edits"] = np.array([min(_levenshtein(c, t) for c in cs) for cs, t in zip(res["cands"], tgt)], float)
    st["fw_at_k"] = np.array([any(c.split()[:1] == t.split()[:1] for c in cs) if t.split() else 0.0
                              for cs, t in zip(res["cands"], tgt)], float)
    st["nll"] = np.array(res["nll"], float) if res["nll"] else np.full(n, np.nan)
    st["conf"] = np.array(res["fw_logp"], float) if res.get("fw_logp") else np.zeros(n)
    return st


def summarise(st: Dict[str, np.ndarray]) -> Dict:
    hr, hi = st["has_read"] > 0, st["has_inf"] > 0
    out = {
        "corpus_cer": st["edits"].sum() / st["lens"].sum(),
        "corpus_cer_95ci": bootstrap_ratio(st["edits"], st["lens"]),
        "macro_wer": float(np.mean(st["wer"])),
        "exact_match": float(st["em"].mean()),
        "first_word_acc": float(st["first_word"].mean()),
        "budget_violation_rate": float(st["viol"].mean()),
        "budget_violation_rate_old_rule": float(st["viol_old"].mean()),
        "mean_len_error": float(st["len_err"].mean()),
        "overlong_vs_truth": float((st["len_err"] > 3).mean()),
        # READ (visual evidence)
        "read_lines_frac": float(hr.mean()),
        "read_acc": float(st["read_ok"][hr].mean()) if hr.any() else None,
        "read_acc_95ci": bootstrap_ratio(st["read_ok"][hr], np.ones(hr.sum())) if hr.any() else None,
        "read_cer": float(st["read_edits"][hr].sum() / st["read_lens"][hr].sum()) if hr.any() else None,
        "read_acc_at_k": float(st["read_ok_at_k"][hr].mean()) if hr.any() else None,
        "ctc_read_acc": float(st["ctc_read_ok"][hr].mean()) if "ctc_read_ok" in st and hr.any() else None,
        "ctc_cer": float(st["ctc_edits"].sum() / st["ctc_lens"].sum()) if "ctc_edits" in st else None,
        # INFERRED (context only)
        "inferred_cer": float(st["inf_edits"][hi].sum() / st["inf_lens"][hi].sum()) if hi.any() else None,
        # Hypothesis-set view
        "exact_match_at_k": float(st["em_at_k"].mean()),
        "first_word_acc_at_k": float(st["fw_at_k"].mean()),
        "oracle_cer_at_k": float(st["oracle_edits"].sum() / st["lens"].sum()),
        "bits_per_char": (float(np.nansum(st["nll"]) / st["lens"].sum() / math.log(2))
                          if not np.all(np.isnan(st["nll"])) else None),
        "auroc_conf_first_word": auroc(st["conf"], st["first_word"]),
        "first_word_acc_top25pct_conf": _selective(st["conf"], st["first_word"], 0.25),
    }
    return _round(out)


def _selective(conf, correct, coverage):
    if len(conf) == 0 or np.all(conf == 0):
        return None
    k = max(1, int(len(conf) * coverage))
    return float(correct[np.argsort(-conf)[:k]].mean())


def _round(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return x


def bootstrap_ratio(num, den, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(num)
    s = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        s.append(num[idx].sum() / max(den[idx].sum(), 1e-9))
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def paired_bootstrap(a, b, den, B=1000, seed=0):
    """Mean delta of error ratio (a - b) and P(a is NOT better than b)."""
    rng = np.random.default_rng(seed)
    n = len(den)
    worse, deltas = 0, []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        d = (a[idx].sum() - b[idx].sum()) / max(den[idx].sum(), 1e-9)
        deltas.append(d)
        worse += d >= 0
    return round(float(np.mean(deltas)), 4), round(worse / B, 4)


def save_reliability_plot(calib: Dict, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    for name, c in calib.items():
        for kind, style in (("raw", ":"), ("platt", "-")):
            rows = c[kind]["bins[lo,n,conf,acc]"]
            ax.plot([r[2] for r in rows], [r[3] for r in rows], style, marker="o", ms=3,
                    label=f"{name} {kind} (ECE {c[kind]['ece']:.3f})")
    ax.set_xlabel("predicted P(first word correct)")
    ax.set_ylabel("observed accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "lines"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--full_ckpt", default=None)
    ap.add_argument("--text_only_ckpt", default=None)
    ap.add_argument("--extra_ckpt", action="append", default=[],
                    help="Additional checkpoint as NAME=PATH (repeatable)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--context_noise", type=float, default=0.0,
                    help="If > 0, also evaluate text_only/full with this OCR error rate in the context")
    ap.add_argument("--calib_n", type=int, default=0, help="Validation lines used to fit calibration")
    ap.add_argument("--out", default=str(ROOT / "results" / "line_eval.json"))
    ap.add_argument("--skip_zero_shot", action="store_true")
    ap.add_argument("--cut_ckpt", default=str(ROOT / "checkpoints" / "cut_estimator.pt"),
                    help="CutOffsetRegressor checkpoint; enables the calibrated upper budget")
    ap.add_argument("--budget_mode", default="hard", choices=["hard", "soft"],
                    help="hard = never exceed the upper bound (look-ahead mask); soft = legacy EOS bias")
    ap.add_argument("--no_upper_budget", action="store_true", help="legacy: decode with the point estimate")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    tok = AutoTokenizer.from_pretrained(cfg.text_model_name)
    loaders: Dict[tuple, DataLoader] = {}

    def loader_for(split: str, style: str, noise: float = 0.0, limit=None):
        key = (split, style, noise, limit)
        if key not in loaders:
            ds = LineCacheDataset(args.data_dir, split, tok, cfg.max_context_len, cfg.max_target_len,
                                  augment=False, limit=limit, prompt_style=style, context_noise=noise)
            loaders[key] = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                      collate_fn=LineCacheDataset.make_collate(tok.pad_token_id))
        return loaders[key]

    base = loader_for(args.split, "v2", limit=args.limit).dataset
    recs = [base.records[i] for i in base.indices]
    raw_targets = [r["target"] for r in recs]
    vis = [int(r.get("visible_chars", 0)) for r in recs]
    crop_texts = [r.get("crop_text", "") for r in recs]
    budgets = [float(r["budget"]) for r in recs]
    has_v3_fields = "prev2_prefix" in recs[0]

    # Calibrated upper budget per record (same for every system)
    cut_model, cut_meta = (None, None) if args.no_upper_budget else load_cut_estimator(args.cut_ckpt, device)
    budget_mode = args.budget_mode if cut_meta else "soft"
    ub_cache: Dict[str, np.ndarray] = {}

    def split_ub(split: str):
        if cut_meta is None:
            return None
        if split not in ub_cache:
            ds = loader_for(split, "v2").dataset
            off = predict_cut_offsets(cut_model, ds.crops, device)
            ub_cache[split] = np.array([upper_budget(estimate_budget(r["b_x"], r["ink_x0"], r["page_w"],
                                                                     len(r["prefix"]), ghost_cut_px=o), cut_meta)
                                        for r, o in zip(ds.records, off)])
        return ub_cache[split]

    upper = [float(split_ub(args.split)[i]) for i in base.indices] if cut_meta else None
    if cut_meta:
        cover = np.mean([len(t.strip()) <= u for t, u in zip(raw_targets, upper)])
        print(f"Upper budget: ratio {cut_meta['ub_ratio']:.3f} + {cut_meta['ub_add']:.1f}; "
              f"{cover:.1%} of true endings fit; decoding mode '{budget_mode}'")
    print(f"Evaluating {len(recs)} lines from {args.data_dir} split '{args.split}' on {device} (top-{args.top_k})")

    results: Dict[str, Dict[str, list]] = {
        "empty": {"cands": [[""] * args.top_k for _ in recs], "scores": [], "nll": [], "ctc": [], "fw_logp": []}
    }
    styles: Dict[str, str] = {}
    calib_models: Dict[str, tuple] = {}

    def evaluate(name, model, modes=(("", "real"),), noise_variant=False):
        style = getattr(model.config, "prompt_style", "v2")
        if style in ("v3", "v3e") and not has_v3_fields:
            print(f"  {name}: skipped (v3 prompt needs a v3 cache)")
            return
        for spec in modes:
            suffix, mode = spec[:2]
            # "legacy" rows decode with the old rule (point estimate, soft EOS bias): used for the
            # old-vs-new budget comparison and for image ablations - a blank image is
            # out-of-distribution and otherwise rambles until it hits the hard ceiling
            legacy = len(spec) > 2 and spec[2] == "legacy" and cut_meta is not None
            nm = name + suffix
            print(f"  {nm} [{style}{', legacy budget' if legacy else ''}] ...", flush=True)
            results[nm] = run_system(model, loader_for(args.split, style, limit=args.limit), tok, device,
                                     image_mode=mode, k=args.top_k, ub=None if legacy else split_ub(args.split),
                                     budget_mode="soft" if legacy else budget_mode)
            styles[nm] = style
        if noise_variant and args.context_noise > 0:
            nm = name + "_ocr"
            print(f"  {nm} [{style}, context noise {args.context_noise}] ...", flush=True)
            results[nm] = run_system(model, loader_for(args.split, style, args.context_noise, args.limit),
                                     tok, device, k=args.top_k, ub=split_ub(args.split), budget_mode=budget_mode)
            styles[nm] = style
        if args.calib_n > 0:
            vl = loader_for("val", style, limit=args.calib_n)
            r = run_system(model, vl, tok, device, k=args.top_k, ub=split_ub("val"), budget_mode=budget_mode)
            vrec = [vl.dataset.records[i] for i in vl.dataset.indices]
            calib_models[name] = (conf_features(r),
                                  first_word_correct(r["cands"], [x["target"] for x in vrec]))

    if not args.skip_zero_shot:
        m = load_model(None, device, text_only=True)
        m.config.prompt_style = "v2"
        evaluate("t5_zero_shot", m)
        del m
    if args.text_only_ckpt:
        m = load_model(args.text_only_ckpt, device)
        evaluate("text_only", m, modes=(("", "real"), ("_soft", "real", "legacy")), noise_variant=True)
        del m
    if args.full_ckpt:
        m = load_model(args.full_ckpt, device)
        print(f"  full: fusion={m.fusion_style} grid={m.visual_encoder.grid} "
              f"reading_head={m.reading_head is not None} t5={m.config.text_model_name}", flush=True)
        evaluate("full", m, modes=(("", "real"), ("_soft", "real", "legacy"), ("_zero_img", "zero", "legacy"),
                                   ("_shuf_img", "shuffle", "legacy")),
                 noise_variant=True)
        del m
    for spec in args.extra_ckpt:
        name, path = spec.split("=", 1)
        m = load_model(path, device)
        evaluate(name, m)
        del m
        torch.cuda.empty_cache()

    stats = {k: line_stats(v, raw_targets, vis, crop_texts, budgets, cfg.budget_margin_buffer, upper)
             for k, v in results.items()}
    report = {"n_lines": len(recs), "data_dir": args.data_dir, "split": args.split, "top_k": args.top_k,
              "context_noise": args.context_noise, "prompt_styles": styles,
              "systems": {k: summarise(st) for k, st in stats.items()}, "comparisons": {}, "calibration": {},
              "budget": {"mode": budget_mode, "upper": cut_meta,
                         "truth_within_upper": (float(np.mean([len(t.strip()) <= u for t, u in
                                                               zip(raw_targets, upper)])) if upper else None),
                         "truth_violates_old_rule": float(np.mean([len(t.strip()) > b + cfg.budget_margin_buffer
                                                                  for t, b in zip(raw_targets, budgets)]))}}

    # Calibration on the test split, fitted on validation
    for name, (Xv, yv) in calib_models.items():
        if name not in results or not results[name]["scores"]:
            continue
        Xt, yt = conf_features(results[name]), stats[name]["first_word"]
        params = fit_platt(Xv, yv)
        raw = np.exp(np.clip(Xt[:, 0], -50, 0))  # model's own P(first word) - uncalibrated
        report["calibration"][name] = {
            "fit_on_val_lines": int(len(yv)),
            "raw": calibration_report(raw, yt),
            "platt": calibration_report(apply_platt(params, Xt), yt),
        }
    if report["calibration"]:
        save_reliability_plot(report["calibration"], Path(args.out).with_suffix(".reliability.png"))

    abl = "full_soft" if "full_soft" in stats else "full"
    pairs = [("full", "text_only"), (abl, "full_zero_img"), (abl, "full_shuf_img"),
             ("full", "full_soft"), ("text_only", "text_only_soft"),
             ("text_only", "t5_zero_shot"), ("full", "empty"), ("text_only", "empty"),
             ("full_ocr", "text_only_ocr"), ("full_ocr", "full"), ("text_only_ocr", "text_only")]
    pairs += [("full", spec.split("=", 1)[0]) for spec in args.extra_ckpt]
    hr = stats["empty"]["has_read"] > 0
    for a, b in pairs:
        if a in stats and b in stats:
            sa, sb = stats[a], stats[b]
            comp = {"cer": paired_bootstrap(sa["edits"], sb["edits"], sa["lens"])}
            if hr.any():
                comp["read_err"] = paired_bootstrap(1 - sa["read_ok"][hr], 1 - sb["read_ok"][hr],
                                                    np.ones(hr.sum()))
            hi = sa["has_inf"] > 0
            comp["inferred_cer"] = paired_bootstrap(sa["inf_edits"][hi], sb["inf_edits"][hi], sa["inf_lens"][hi])
            if not (np.all(np.isnan(sa["nll"])) or np.all(np.isnan(sb["nll"]))):
                d = (sa["nll"] - sb["nll"]) / math.log(2)
                comp["bits_per_char"] = paired_bootstrap(np.nan_to_num(d), np.zeros_like(d), sa["lens"])
            report["comparisons"][f"{a}_vs_{b}"] = comp

    # Breakdown for the full model: paragraph-final vs running lines, hyphenated lines
    if "full" in stats and "para_end" in recs[0]:
        brk = {}
        for label, mask in (("para_end", [r.get("para_end", False) for r in recs]),
                            ("hyphenated", [r.get("hyphenated", False) for r in recs])):
            mk = np.array(mask, bool)
            for sysname in ("text_only", "full"):
                if sysname in stats and mk.any():
                    st = stats[sysname]
                    brk[f"{sysname}_{label}"] = _round({
                        "n": int(mk.sum()),
                        "cer": float(st["edits"][mk].sum() / st["lens"][mk].sum()),
                        "cer_rest": float(st["edits"][~mk].sum() / st["lens"][~mk].sum()),
                        "viol": float(st["viol"][mk].mean()), "viol_rest": float(st["viol"][~mk].mean()),
                    })
        report["breakdown"] = brk

    samples = []
    for j in range(min(30, len(recs))):
        r_part, f_part = split_visible(raw_targets[j], vis[j])
        samples.append({"prefix": recs[j]["prefix"], "target": raw_targets[j].strip(),
                        "read": r_part, "inferred": f_part,
                        **{k: v["cands"][j] for k, v in results.items() if k != "empty"},
                        **({"ctc_full": results["full"]["ctc"][j]} if results.get("full", {}).get("ctc") else {})})
    report["samples"] = samples

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    S = report["systems"]
    f = lambda v: "   -  " if v is None else f"{v:6.3f}"  # noqa: E731
    print(f"\n{'system':<16}{'CER':>7}{'EM':>7}{'1stW':>7}{'READacc':>8}{'CTCread':>8}{'INFcer':>8}"
          f"{'EM@k':>7}{'1stW@k':>7}{'bpc':>7}{'viol':>7}{'overlg':>7}")
    for name, m in S.items():
        print(f"{name:<16}{f(m['corpus_cer'])}{f(m['exact_match'])} {f(m['first_word_acc'])}  {f(m['read_acc'])}  "
              f"{f(m['ctc_read_acc'])}  {f(m['inferred_cer'])} {f(m['exact_match_at_k'])} "
              f"{f(m['first_word_acc_at_k'])} {f(m['bits_per_char'])} {f(m['budget_violation_rate'])} "
              f"{f(m['overlong_vs_truth'])}")
    print("\nPaired bootstrap (delta = first - second; negative = first better; p = P(first not better)):")
    for k, v in report["comparisons"].items():
        print("  " + k.ljust(28) + "  ".join(f"{m}={d:+.4f} (p={p:.3f})" for m, (d, p) in v.items()))
    for name, c in report["calibration"].items():
        print(f"calibration {name}: ECE raw {c['raw']['ece']:.3f} -> platt {c['platt']['ece']:.3f}; "
              f"Brier {c['raw']['brier']:.3f} -> {c['platt']['brier']:.3f}")
    if report.get("breakdown"):
        print("breakdown:", json.dumps(report["breakdown"]))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
