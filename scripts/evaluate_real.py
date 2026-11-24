"""
scripts/evaluate_real.py
========================
Evaluates a checkpoint on photographed, genuinely damaged pages produced with
``scripts/make_real_test_pages.py`` (see that file for the protocol).

For every annotated line (non-empty "legible"):
  prefix  = legible text            (what a human / OCR could read)
  target  = full[len(legible):]     (known, because we printed it)
  READ    = the "faint" characters  (partly visible after the legible text)

Geometry: the photo is rescaled so that its text line pitch matches the printed
pitch at the 1024-px training scale; text lines and the burn boundary are found
by the burn detector and matched to the annotated lines top-to-bottom.
Reports READ accuracy, INFERRED CER, CER, EM, top-5 recall and vision-only
CTC reading, compared with the text-only checkpoint if given.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.burn_detector import SeveredLineRegion  # noqa: E402
from inference.restoration_engine import RestorationEngine  # noqa: E402
from line_data import CROP_LEFT_PX, CROP_RIGHT_PX, split_visible  # noqa: E402
from metrics import _levenshtein  # noqa: E402


def detect_line_pitch(gray: np.ndarray) -> float:
    """Median distance between text-line centres from a horizontal ink profile."""
    ink = (gray < 128).astype(np.float32)
    left = ink[:, : gray.shape[1] // 2].sum(1)  # left half is undamaged
    rows = left > 0.15 * left.max()
    starts = np.flatnonzero(rows[1:] & ~rows[:-1])
    ends = np.flatnonzero(~rows[1:] & rows[:-1])
    if len(starts) < 2 or len(ends) < 2:
        return 0.0
    n = min(len(starts), len(ends))
    centres = (starts[:n] + ends[:n]) / 2.0
    return float(np.median(np.diff(centres)))


def run(engine: RestorationEngine, image: np.ndarray, page: dict, top_k: int) -> list:
    det = engine.detect_burn_region(image)
    lines = [l for l in page["lines"]]
    # detected text bands (top-to-bottom) from the detector's severed lines + all bands
    bands = sorted([(r.y0, r.y1) for r in det.severed_lines])
    profile = np.array(det.boundary_profile)
    out = []
    ann = [l for l in lines if l["legible"].strip()]
    if len(bands) < len(ann):
        print(f"    warning: detector found {len(bands)} severed lines but {len(ann)} are annotated; "
              f"matching by vertical position")
    expected = [(l["y0_at_1024"] + l["y1_at_1024"]) / 2 for l in ann]
    for l, ey in zip(ann, expected):
        # nearest detected band to the expected centre (pages are roughly registered after rescaling)
        if bands:
            y0, y1 = min(bands, key=lambda b: abs((b[0] + b[1]) / 2 - ey))
        else:
            y0, y1 = int(l["y0_at_1024"]), int(l["y1_at_1024"])
        b_x = int(np.median(profile[y0:y1 + 1])) if len(profile) else image.shape[1] - 1
        region = SeveredLineRegion(line_idx=l["line_idx"], y0=y0, y1=y1, b_x=b_x,
                                   crop_bbox=(b_x - CROP_LEFT_PX, y0, b_x + CROP_RIGHT_PX, y1),
                                   damaged_width_px=image.shape[1] - b_x, intact_width_px=b_x,
                                   line_height_px=y1 - y0, ink_x0=0)
        idx = {x["line_idx"]: x for x in lines}
        ctx = lambda d: idx.get(l["line_idx"] + d, {}).get("legible", "")  # noqa: E731
        legible = l["legible"]
        res = engine.restore_region(image, region, prefix_text=legible, prev_prefix=ctx(-1),
                                    next_prefix=ctx(1), prev2_prefix=ctx(-2), next2_prefix=ctx(2), top_k=top_k)
        target = l["full"][len(legible):] if l["full"].startswith(legible) else l["full"][len(legible):]
        out.append({"line_idx": l["line_idx"], "legible": legible, "faint": l["faint"], "target": target.strip(),
                    "visible_chars": len(target) - len(target.lstrip()) + len(l["faint"]),
                    "raw_target": target, "pred": res["predicted_restoration"],
                    "candidates": res.get("candidates", []), "visual_reading": res.get("visual_reading", "")})
    return out


def score(rows: list) -> dict:
    if not rows:
        return {}
    e = l = re_ok = re_n = ie = il = em = em5 = 0
    for r in rows:
        t, p = r["target"], r["pred"].strip()
        e += _levenshtein(p, t); l += max(len(t), 1)
        em += p == t
        em5 += any(c.strip() == t for c in r["candidates"]) if r["candidates"] else p == t
        rd, inf = split_visible(r["raw_target"], r["visible_chars"])
        if rd:
            re_n += 1; re_ok += p.startswith(rd)
        ie += _levenshtein(p[len(rd):].strip(), inf); il += max(len(inf), 1)
    n = len(rows)
    return {"n_lines": n, "cer": round(e / l, 4), "exact_match": round(em / n, 4),
            "exact_match_at_k": round(em5 / n, 4), "read_lines": re_n,
            "read_acc": round(re_ok / re_n, 4) if re_n else None, "inferred_cer": round(ie / il, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "real_test"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text_only_ckpt", default=None)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()
    d = Path(args.dir)
    ann = json.loads((d / "annotations.json").read_text())
    report = {}
    for name, ck in [("full", args.ckpt), ("text_only", args.text_only_ckpt)]:
        if not ck:
            continue
        engine = RestorationEngine(checkpoint_path=ck)
        engine.load_model()
        rows = []
        for page in ann["pages"]:
            photo = d / page["photo"]
            if not photo.is_file() or not any(l["legible"].strip() for l in page["lines"]):
                continue
            img = cv2.imread(str(photo))
            pitch = detect_line_pitch(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            target_pitch = page.get("line_pitch_px_at_1024") or 0
            if pitch > 0 and target_pitch:
                f = target_pitch / pitch
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
            print(f"  [{name}] page {page['page']}: rescaled x{target_pitch / pitch if pitch else 1:.3f}")
            rows.extend(run(engine, img, page, args.top_k))
        report[name] = {"metrics": score(rows), "lines": rows}
        print(f"  [{name}] {report[name]['metrics']}")
    if not report or not any(r["metrics"] for r in report.values()):
        print("No annotated photos found - see scripts/make_real_test_pages.py for the protocol.")
    out = ROOT / "results" / "real_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
