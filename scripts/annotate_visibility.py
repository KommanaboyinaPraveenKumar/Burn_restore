"""
scripts/annotate_visibility.py
==============================
Adds ``visible_chars`` to an existing line cache without re-writing the crops.

``visible_chars`` = how many leading characters of the target still have
legible (faded) strokes in the crop.  It separates the target into
  * the READ part     - recoverable from visual evidence, and
  * the INFERRED part - fully burnt, only guessable from context.

The pages are re-rendered from the same deterministic job list used by
build_line_dataset.py; every record is checked (prefix, target, line_idx,
b_x) before the new field is written, so a mismatch aborts instead of
silently corrupting labels.

Usage:
    python scripts/annotate_visibility.py --data_dir data/lines
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_line_dataset import _worker, make_jobs  # noqa: E402


def _records_only(job):
    records, _ = _worker(job)
    return records


def annotate_split(split: str, n_pages: int, seed: int, data_dir: Path, corpus_dir: Path, workers: int):
    path = data_dir / split / "records.jsonl"
    with open(path, encoding="utf-8") as f:
        old = [json.loads(l) for l in f]
    jobs = make_jobs(split, n_pages, corpus_dir, seed, verbose=False)
    new = []
    with mp.Pool(workers) as pool:
        for recs in pool.imap(_records_only, jobs, chunksize=4):
            new.extend(recs)
    if len(new) != len(old):
        raise RuntimeError(f"[{split}] record count mismatch: {len(new)} regenerated vs {len(old)} cached")
    for i, (a, b) in enumerate(zip(old, new)):
        for k in ("prefix", "target", "line_idx", "b_x", "page_id"):
            if a[k] != b[k]:
                raise RuntimeError(f"[{split}] record {i} field {k!r} differs: {a[k]!r} vs {b[k]!r}")
        a["visible_chars"] = b["visible_chars"]
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in old:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    n_vis = [r["visible_chars"] for r in old]
    has = sum(v > 0 for v in n_vis)
    print(f"  [{split}] {len(old):,} records annotated; {has / len(old):.1%} have >=1 visible target char, "
          f"mean {sum(n_vis) / len(n_vis):.2f} visible chars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "lines"))
    ap.add_argument("--corpus_dir", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    margs = meta["args"]
    corpus_dir = Path(args.corpus_dir or margs["corpus_dir"])
    for i, split in enumerate(["train", "val", "test"]):
        n = margs[f"{split}_pages"]
        if n > 0:
            annotate_split(split, n, margs["seed"] + i, data_dir, corpus_dir, args.workers)


if __name__ == "__main__":
    main()
