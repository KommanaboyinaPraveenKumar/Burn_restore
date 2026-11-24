"""The page-level BurnDetector must find the severed lines of a v3 synthetic page
and place the burn edge close to the generator's b_x (text ink must not be
mistaken for char)."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"


@pytest.mark.skipif(not CORPUS.is_dir() or not any(CORPUS.iterdir()), reason="Gutenberg corpus not downloaded")
def test_detector_matches_ground_truth_on_v3_pages():
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_line_dataset import make_jobs, sample_layout
    from config import Config
    from dataset_generator import SideBurnGenerator
    from inference.burn_detector import BurnDetector
    from line_data import burn_page_to_records, render_page

    det, gen = BurnDetector(), SideBurnGenerator(Config())
    found, total, errs = 0, 0, []
    for _, words, font, seed, _ in make_jobs("test", 3, CORPUS, seed=5, verbose=False, version="v3"):
        rng = random.Random(seed)
        page, _ = render_page(words, font, rng, layout=sample_layout(rng))
        img, recs, _ = burn_page_to_records(page, gen, seed=seed, appearance="v3")
        lines = det.detect(img).severed_lines
        for r in recs:
            total += 1
            yc = (r["y0"] + r["y1"]) / 2
            hit = [l for l in lines if l.y0 <= yc <= l.y1]
            if hit:
                found += 1
                errs.append(abs(hit[0].b_x - r["b_x"]))
    assert found / total >= 0.95
    assert np.mean(errs) <= 12


def test_parse_page_text():
    from api.routes.restoration import parse_page_text
    assert parse_page_text(None) is None
    assert parse_page_text("  \n ") is None
    assert parse_page_text("\n\nfirst line  \r\n\nthird\n\n") == ["first line", "", "third"]
