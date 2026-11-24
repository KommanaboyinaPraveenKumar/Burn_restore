"""
scripts/validate_burn_detector.py
=================================
Checks the page-level BurnDetector (used by the demo / API) against the ground
truth of synthetic held-out pages: for every true severed line, is a detected
line found at that height, and how far is its burn edge b_x from the true one?

Usage:
    python scripts/validate_burn_detector.py --pages 20
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_line_dataset import make_jobs, sample_layout  # noqa: E402
from config import Config  # noqa: E402
from dataset_generator import SideBurnGenerator  # noqa: E402
from inference.burn_detector import BurnDetector  # noqa: E402
from line_data import burn_page_to_records, render_page  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--corpus_dir", default=str(ROOT / "data" / "corpus"))
    args = ap.parse_args()
    det = BurnDetector()
    gen = SideBurnGenerator(Config())
    found, total, false_pos, errs = 0, 0, 0, []
    for p, words, font, seed, version in make_jobs("test", args.pages, Path(args.corpus_dir), seed=args.seed,
                                                   verbose=False, version="v3"):
        rng = random.Random(seed)
        page, _ = render_page(words, font, rng, layout=sample_layout(rng))
        img, recs, _ = burn_page_to_records(page, gen, seed=seed, appearance="v3")
        lines = det.detect(img).severed_lines
        used = set()
        for r in recs:
            total += 1
            yc = (r["y0"] + r["y1"]) / 2
            cand = [l for l in lines if l.y0 <= yc <= l.y1 and l.line_idx not in used]
            if cand:
                l = cand[0]
                used.add(l.line_idx)
                found += 1
                errs.append(l.b_x - r["b_x"])
        false_pos += len(lines) - len(used)
    e = np.array(errs, float)
    print(f"pages {args.pages}: true severed lines {total}, matched {found} ({found / max(1, total):.1%}), "
          f"unmatched detections {false_pos}")
    if len(e):
        print(f"b_x error (detected - true) px: median {np.median(e):+.1f}, MAE {np.abs(e).mean():.1f}, "
              f"|err|<=10px {np.mean(np.abs(e) <= 10):.1%}, |err|<=25px {np.mean(np.abs(e) <= 25):.1%}")


if __name__ == "__main__":
    main()
