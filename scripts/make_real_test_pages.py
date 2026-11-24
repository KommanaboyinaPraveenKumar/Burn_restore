"""
scripts/make_real_test_pages.py
===============================
Generates printable pages for a *real-damage* test: text from the held-out
TEST books, rendered at print resolution, plus an annotation template with the
exact text of every line (the ground truth is known because we print it).

Protocol (see README section "Real-damage test"):
  1. python scripts/make_real_test_pages.py --n_pages 4 --out real_test
  2. Print real_test/page_*.png (fit to page width).
  3. Char / scorch the RIGHT edge of each sheet so that line endings are lost
     and a band of partly-legible ink remains.  Do this outdoors or over a sink,
     with water at hand; never leave burning paper unattended.
  4. Photograph or scan each sheet flat, crop to the sheet (the burnt edge may be
     ragged) and save as real_test/photo_<n>.jpg.
  5. For every damaged line, fill in real_test/annotations.json:
        "legible": the text you can read with confidence (left of the burn)
        "faint":   characters right after it that are only partly visible
     Lines that are not damaged can be left empty.
  6. python scripts/evaluate_real.py --dir real_test --ckpt checkpoints/v3_full/best_checkpoint.pt
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_line_dataset import make_jobs, sample_layout  # noqa: E402
from line_data import render_page  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pages", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "real_test"))
    ap.add_argument("--scale", type=float, default=3.0, help="render scale (3 -> ~300 dpi on a 17 cm column)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs("test", args.n_pages, ROOT / "data" / "corpus", args.seed, verbose=False, version="v3")
    pages = []
    for n, (pid, words, font, seed, _) in enumerate(jobs, start=1):
        rng = random.Random(seed)
        layout = sample_layout(rng)
        layout["justify"] = True
        page, _ = render_page(words, font, rng, layout=layout, scale=args.scale)
        img_path = out / f"page_{n}.png"
        cv2.imwrite(str(img_path), page.image_bgr)
        s = args.scale
        pages.append({
            "page": n,
            "printed_image": img_path.name,
            "photo": f"photo_{n}.jpg",
            "font": Path(font).name,
            "line_pitch_px_at_1024": round((page.bands[1][0] - page.bands[0][0]) / s, 2) if len(page.bands) > 1 else None,
            "lines": [{"line_idx": i, "full": text, "y0_at_1024": round(b[0] / s, 1), "y1_at_1024": round(b[1] / s, 1),
                       "legible": "", "faint": ""}
                      for i, (text, b) in enumerate(zip(page.lines, page.bands))],
        })
        print(f"  page {n}: {len(page.lines)} lines, font {Path(font).name} -> {img_path}")
    (out / "annotations.json").write_text(json.dumps({"pages": pages}, indent=2))
    print(f"Wrote {len(pages)} pages and {out / 'annotations.json'}")


if __name__ == "__main__":
    main()
