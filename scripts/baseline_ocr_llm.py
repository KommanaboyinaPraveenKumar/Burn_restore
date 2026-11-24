"""
scripts/baseline_ocr_llm.py
===========================
External baseline: off-the-shelf OCR + a larger pretrained language model,
with no training on this task.

  1. OCR   TrOCR (trocr-base-stage1, printed text) reads the line crop (the
           vertical stretch of the cached crop is undone first; Tesseract and
           trocr-large-printed were worse on 150 test crops: CER 0.29 / 0.95
           vs 0.09). The OCR text is aligned to the end of the known
           surviving prefix; whatever OCR reads past it is the baseline's
           READ guess.
  2. LLM   A base causal LM continues  prev2 … prev … prefix + READ guess
           with beam search. Candidates are re-ranked by the joint likelihood
           of  context + candidate + start of the next line,  so the LLM also
           uses the next-line context the T5 models get.
  3. Budget  every candidate is cut to the same calibrated upper bound the
           other systems decode under (no candidate exceeds it).

Two systems are written:
  llm_only   step 2 without OCR (pure language model)
  ocr_llm    steps 1 + 2

Output JSON (consumed by evaluate_lines.py --extra_preds NAME=PATH:SYSTEM):
  {"meta": {...}, "ocr": [...], "ocr_read": [...],
   "systems": {"llm_only": [[c1..ck], ...], "ocr_llm": [[...], ...]}}

Usage:
    python scripts/baseline_ocr_llm.py --data_dir data/lines_v3 --split test \
        --out results/preds_ocr_llm_test.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from line_data import CROP_PAD_Y, estimate_budget  # noqa: E402
from model import load_cut_estimator, predict_cut_offsets, upper_budget  # noqa: E402

SEP = " … "


def align_read(prefix: str, ocr: str) -> str:
    """Returns the part of the OCR string that lies past the end of `prefix`.

    Semi-global alignment: o[:i] is matched against a suffix of the prefix tail
    (free start), and the split point i with the lowest edit cost is chosen
    (ties -> the later split, so a fully matched prefix is not cut short)."""
    if not ocr:
        return ""
    t = prefix[-(len(ocr) + 8):].lower()      # caseless: some OCR models output capitals
    o = ocr.lower()
    n, m = len(o), len(t)
    prev = [0] * (m + 1)                      # D[0][j] = 0: any start in t is free
    best_i, best = 0, prev[m] if m == 0 else m
    col_m = [m]
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        oi = o[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (oi != t[j - 1]))
        col_m.append(cur[m])
        prev = cur
    for i, c in enumerate(col_m):
        if c <= best:
            best, best_i = c, i
    return ocr[best_i:]


def line_image(crop, r) -> np.ndarray:
    """Undo the vertical stretch of the cached crop (the cache resizes the line band to 128 px)."""
    h = max(16, (r["y1"] - r["y0"]) + 2 * CROP_PAD_Y)
    return cv2.resize(np.asarray(crop), (crop.shape[1], h), interpolation=cv2.INTER_AREA)


def run_tesseract(crops, recs, exe, workers=8, scale=3) -> List[str]:
    import subprocess
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    tmp = Path(tempfile.mkdtemp())

    def one(i):
        img = line_image(crops[i], recs[i])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        f = tmp / f"{i}.png"
        cv2.imwrite(str(f), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        res = subprocess.run([exe, str(f), "stdout", "--psm", "7", "-l", "eng"], capture_output=True)
        f.unlink(missing_ok=True)
        return res.stdout.decode("utf-8", "replace").strip()

    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, range(len(recs))))
    return out


def run_ocr(crops, recs, model_name, device, bs=32) -> List[str]:
    if model_name.lower().endswith("tesseract.exe") or model_name.lower() == "tesseract":
        exe = model_name if model_name.lower() != "tesseract" else r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        return run_tesseract(crops, recs, exe)
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device).eval()
    out = []
    for s in range(0, len(recs), bs):
        imgs = [line_image(crops[i], recs[i]) for i in range(s, min(s + bs, len(recs)))]
        px = proc(images=imgs, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(px, num_beams=4, max_new_tokens=48)
        out += [t.strip().removesuffix("...").rstrip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
        if s % (bs * 40) == 0:
            print(f"  ocr {s}/{len(recs)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def context_text(r) -> str:
    parts = [p for p in (r.get("prev2_prefix", ""), r.get("prev_prefix", "")) if p.strip()]
    return SEP.join(parts + [r["prefix"]])


def clean(gen: str) -> str:
    for stop in ("\n", "…"):
        gen = gen.split(stop)[0]
    return gen


@torch.no_grad()
def seq_logprob(model, tok, texts: List[str], device, bs=48) -> List[float]:
    out, side = [], tok.padding_side
    tok.padding_side = "right"          # right padding keeps position ids exact for scoring
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True).to(device)
        logits = model(**enc).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, enc.input_ids[:, 1:, None])[..., 0]
        out += (lp * enc.attention_mask[:, 1:]).sum(1).tolist()
    tok.padding_side = side
    return out


@torch.no_grad()
def run_llm(recs, reads, ub, model, tok, device, k=5, beams=10, bs=8, next_chars=60) -> List[List[str]]:
    cands_all = []
    t0 = time.time()
    for s in range(0, len(recs), bs):
        idx = list(range(s, min(s + bs, len(recs))))
        prompts = [context_text(recs[i]) + reads[i] for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        max_new = max(int(max(ub[i] for i in idx) / 2.5) + 4, 6)
        gen = model.generate(**enc, num_beams=beams, num_return_sequences=beams, max_new_tokens=max_new,
                             do_sample=False, early_stopping=True, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        rerank_texts, owners, endings = [], [], []
        for b, i in enumerate(idx):
            seen, leads = [], []
            for g in texts[b * beams:(b + 1) * beams]:
                raw = reads[i] + clean(g)
                end = raw.lstrip()[: int(ub[i])].rstrip()   # budget counts the stripped ending
                if end and end not in seen:
                    seen.append(end)
                    leads.append(raw[:1].isspace())
            nxt = recs[i].get("next_prefix", "")[:next_chars]
            for end, lead in zip(seen, leads):
                rerank_texts.append(context_text(recs[i]) + _glue(recs[i], end, lead) + "\n" + nxt)
                owners.append(i)
                endings.append(end)
        scores = seq_logprob(model, tok, rerank_texts, device)
        for i in idx:
            ranked = sorted([(sc, e) for o, sc, e in zip(owners, scores, endings) if o == i], key=lambda x: -x[0])
            c = [e for _, e in ranked][:k] or [""]
            cands_all.append(c + [c[-1] if c else ""] * (k - len(c)))
        if s % (bs * 100) == 0:
            print(f"  llm {s}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    return cands_all


def _glue(r, end: str, gen_lead: bool) -> str:
    """Prefix + ending as it would read on the line. Whether the ending starts a new
    word is the candidate's own choice (gen_lead), not the truth's."""
    return r["prefix"] + (" " if gen_lead else "") + end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "lines_v3"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ocr_model", default="microsoft/trocr-base-stage1",
                    help="TrOCR model id, or \"tesseract\" (stage1 had the lowest CER on 150 test crops)")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--cut_ckpt", default=str(ROOT / "checkpoints" / "cut_estimator.pt"))
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--beams", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results" / "preds_ocr_llm_test.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Path(args.data_dir) / args.split
    recs = [json.loads(line) for line in open(d / "records.jsonl", encoding="utf-8")]
    crops = np.load(d / "crops.npy", mmap_mode="r")
    if args.limit:
        recs, crops = recs[: args.limit], crops[: args.limit]

    cut_model, cut_meta = load_cut_estimator(args.cut_ckpt, device)
    off = predict_cut_offsets(cut_model, crops, device)
    ub = [upper_budget(estimate_budget(r["b_x"], r["ink_x0"], r["page_w"], len(r["prefix"]), ghost_cut_px=o),
                       cut_meta) for r, o in zip(recs, off)]
    del cut_model

    print(f"OCR with {args.ocr_model} on {len(recs)} lines ...", flush=True)
    ocr = run_ocr(crops, recs, args.ocr_model, device)
    ocr_read = [align_read(r["prefix"], o) for r, o in zip(recs, ocr)]
    from metrics import _levenshtein
    cer = sum(_levenshtein(o, r.get("crop_text", "").strip()) for o, r in zip(ocr, recs)) / max(
        1, sum(len(r.get("crop_text", "").strip()) for r in recs))
    print(f"  OCR CER vs visible crop text: {cer:.3f}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.llm)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.float16).to(device).eval()

    systems = {}
    for name, reads in (("llm_only", [""] * len(recs)), ("ocr_llm", ocr_read)):
        print(f"{name} with {args.llm} ...", flush=True)
        systems[name] = run_llm(recs, reads, ub, model, tok, device, k=args.top_k, beams=args.beams)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {"data_dir": args.data_dir, "split": args.split, "n": len(recs), "ocr_model": args.ocr_model,
                 "llm": args.llm, "top_k": args.top_k, "beams": args.beams, "budget": "calibrated upper, truncated"},
        "ocr": ocr, "ocr_read": ocr_read, "systems": systems}, indent=1))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
