"""
scripts/build_line_dataset.py
=============================
Builds the line-level training / validation / test cache.

  1. Downloads public-domain books from Project Gutenberg (cached in data/corpus).
  2. Splits *by book* so no test sentence is ever seen in training.
  3. Renders synthetic pages (varied fonts, sizes, margins, spacing, paper tone),
     applies a jagged side burn, and splits every line at character level.
  4. Stores every severed line as one example:
        data/lines/<split>/crops.npy    uint8 memmap (N, 128, 256, 3) RGB
        data/lines/<split>/records.jsonl

Usage:
    python scripts/build_line_dataset.py --train_pages 2000 --val_pages 150 --test_pages 250
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from dataset_generator import SideBurnGenerator  # noqa: E402
from line_data import PARA, CROP_H, CROP_W, burn_page_to_records, normalize_line_text, render_page  # noqa: E402

# Book-level split.  IDs are Project Gutenberg ebook numbers.
BOOKS_V3_EXTRA = {
    "train": [1080, 1184, 2600, 4300, 1399, 3207, 2148, 829, 25344, 2097, 141, 105, 121, 1023,
              766, 580, 967, 1998, 4363],
    "val": [863],
    "test": [599, 2500],
}

BOOKS = {
    "train": [1342, 84, 11, 1661, 98, 2701, 74, 76, 345, 174, 1260, 768, 158, 161,
              1400, 730, 120, 36, 35, 43, 2814, 215, 219, 5200, 205, 1232, 2554, 16328],
    "val": [1952, 244, 2591],
    "test": [55, 514, 46, 1727],
}

FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
FONTS = ["times.ttf", "georgia.ttf", "BOOKOS.TTF", "GARA.TTF", "CENTURY.TTF", "BKANT.TTF",
         "cambria.ttc", "constan.ttf", "arial.ttf", "calibri.ttf", "Candara.ttf",
         "verdana.ttf", "corbel.ttf", "pala.ttf", "timesi.ttf", "georgiai.ttf"]


MIRRORS = ["https://gutenberg.pglaf.org", "https://mirrors.xmission.com/gutenberg"]


def _mirror_urls(book_id: int):
    s = str(book_id)
    sub = "/".join(s[:-1]) if len(s) > 1 else "0"
    for m in MIRRORS:
        for name in (f"{s}-0.txt", f"{s}.txt", f"{s}-8.txt"):
            yield f"{m}/{sub}/{s}/{name}"
    yield f"https://www.gutenberg.org/cache/epub/{s}/pg{s}.txt"


def download_book(book_id: int, corpus_dir: Path) -> str:
    path = corpus_dir / f"pg{book_id}.txt"
    if not path.is_file():
        last_err = None
        for url in _mirror_urls(book_id):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "burn-restore-dataset-builder/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if len(data) > 10_000:
                    path.write_bytes(data)
                    break
            except Exception as e:  # try the next mirror / filename
                last_err = e
        else:
            raise RuntimeError(f"all mirrors failed for {book_id}: {last_err}")
    raw = path.read_text(encoding="utf-8", errors="ignore")
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF", raw)
    body = raw[start.end() if start else 0: end.start() if end else len(raw)]
    return body


def book_words(body: str, paragraphs: bool = False) -> list:
    """Word stream of a book.  Very short paragraphs (headings, chapter numbers)
    are dropped.  With ``paragraphs`` a PARA token separates paragraphs."""
    paras = [p.replace("\n", " ") for p in re.split(r"\n\s*\n", body)]
    paras = [normalize_line_text(p) for p in paras if len(p.split()) >= 8]
    if not paragraphs:
        return " ".join(paras).split()
    out = []
    for p in paras:
        out.extend(p.split())
        out.append(PARA)
    return out


def sample_layout(rng: random.Random) -> dict:
    """Per-page typographic layout (v3)."""
    return {
        "indent": rng.choice([0.0, 0.0, 1.5, 2.0, 3.0]),
        "hyphen_p": rng.choice([0.0, 0.3, 0.6, 0.9]),
        "justify": rng.random() < 0.5,
    }


def _worker(args):
    page_idx, words, font_path, seed = args[:4]
    version = args[4] if len(args) > 4 else "v2"
    cfg = Config()
    gen = SideBurnGenerator(cfg)
    rng = random.Random(seed)
    if version == "v3":
        page, used = render_page(words, font_path, rng, layout=sample_layout(rng))
        _, records, crops = burn_page_to_records(page, gen, seed=seed, appearance="v3")
    else:
        page, used = render_page(words, font_path, rng)
        _, records, crops = burn_page_to_records(page, gen, seed=seed)
    for r in records:
        r["page_id"] = page_idx
        r["font"] = os.path.basename(font_path)
    return records, crops


def make_jobs(split: str, n_pages: int, corpus_dir: Path, seed: int, verbose: bool = True,
              version: str = "v2") -> list:
    """Deterministic page jobs for a split (same seed -> same pages)."""
    words = []
    books = BOOKS[split] + (BOOKS_V3_EXTRA[split] if version == "v3" else [])
    for bid in books:
        try:
            words.extend(book_words(download_book(bid, corpus_dir), paragraphs=(version == "v3")))
            if verbose:
                print(f"  [{split}] book {bid}: total words {len(words):,}")
        except Exception as e:  # network issues etc.
            print(f"  [{split}] WARNING could not load book {bid}: {e}")
    if not words:
        raise RuntimeError(f"No text for split {split}")

    fonts = [str(FONT_DIR / f) for f in FONTS if (FONT_DIR / f).is_file()]
    rng = random.Random(seed)
    jobs = []
    for p in range(n_pages):
        start = rng.randint(0, max(0, len(words) - 600))
        jobs.append((p, words[start:start + 600], rng.choice(fonts), seed * 100003 + p, version))
    return jobs


def build_split(split: str, n_pages: int, corpus_dir: Path, out_dir: Path, seed: int, workers: int,
                version: str = "v2"):
    jobs = make_jobs(split, n_pages, corpus_dir, seed, version=version)

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    # Crops are streamed to a raw temp file so RAM stays flat for large splits
    all_records = []
    raw_path = split_dir / "crops.raw.tmp"
    n_crops = 0
    with mp.Pool(workers) as pool, open(raw_path, "wb") as raw:
        for i, (records, crops) in enumerate(pool.imap(_worker, jobs, chunksize=4)):
            all_records.extend(records)
            for c in crops:
                raw.write(np.ascontiguousarray(c, dtype=np.uint8).tobytes())
            n_crops += len(crops)
            if (i + 1) % 200 == 0:
                print(f"  [{split}] {i + 1}/{n_pages} pages, {len(all_records):,} lines", flush=True)

    src = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=(n_crops, CROP_H, CROP_W, 3))
    mm = np.lib.format.open_memmap(split_dir / "crops.npy", mode="w+", dtype=np.uint8,
                                   shape=(n_crops, CROP_H, CROP_W, 3))
    for s0 in range(0, n_crops, 4096):
        mm[s0:s0 + 4096] = src[s0:s0 + 4096]
    mm.flush()
    del src, mm
    os.remove(raw_path)
    with open(split_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"  [{split}] done: {n_pages} pages -> {len(all_records):,} severed lines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pages", type=int, default=2000)
    ap.add_argument("--val_pages", type=int, default=150)
    ap.add_argument("--test_pages", type=int, default=250)
    ap.add_argument("--out_dir", default=str(ROOT / "data" / "lines"))
    ap.add_argument("--corpus_dir", default=str(ROOT / "data" / "corpus"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--version", choices=["v2", "v3"], default="v2",
                    help="v3 = paragraphs/indent/hyphenation/justification, randomised burn, "
                         "2-line context, CTC crop_text labels, extra books")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    for i, (split, n) in enumerate([("train", args.train_pages), ("val", args.val_pages), ("test", args.test_pages)]):
        if n > 0:
            build_split(split, n, corpus_dir, out_dir, args.seed + i, args.workers, version=args.version)
    books = {k: v + (BOOKS_V3_EXTRA[k] if args.version == "v3" else []) for k, v in BOOKS.items()}
    meta = {"books": books, "fonts": FONTS, "args": vars(args)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
