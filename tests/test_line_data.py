"""Tests for line_data: the shared train/eval/inference preprocessing contract."""

import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from dataset_generator import SideBurnGenerator  # noqa: E402
from line_data import (  # noqa: E402
    CROP_H, CROP_W, build_prompt, burn_page_to_records, crop_line, crop_to_tensor,
    estimate_budget, extract_infill, normalize_line_text, render_page, split_at,
)

FONT = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "times.ttf")


class TestLineData(unittest.TestCase):
    def test_extract_infill_stops_at_next_sentinel(self):
        raw = "<pad><extra_id_0> of the night<extra_id_1> garbage after</s>"
        self.assertEqual(extract_infill(raw), "of the night")
        self.assertEqual(extract_infill("<pad><extra_id_0> words</s><pad>"), "words")

    def test_prompt_preserves_midword_cut(self):
        self.assertIn("line: restora<extra_id_0>", build_prompt("restora"))
        self.assertIn("line: the <extra_id_0>", build_prompt("the "))

    def test_normalize_is_ascii(self):
        out = normalize_line_text("\u201cCaf\u00e9\u201d \u2014 na\u00efve")
        self.assertEqual(out, '"Cafe" - naive')
        out.encode("ascii")

    def test_split_at_character_level(self):
        boxes = [(c, i * 10.0, i * 10.0 + 10.0) for i, c in enumerate("hello world")]
        self.assertEqual(split_at(boxes, 33.0), ("hel", "lo world"))

    def test_crop_is_rgb_and_fixed_size(self):
        img = np.zeros((100, 400, 3), dtype=np.uint8)
        img[..., 0] = 255  # pure blue in BGR
        crop = crop_line(img, 40, 60, 390)  # near the right edge -> padding path
        self.assertEqual(crop.shape, (CROP_H, CROP_W, 3))
        self.assertTrue((crop[..., 2] == 255).all() and (crop[..., 0] == 0).all())  # RGB order
        self.assertEqual(tuple(crop_to_tensor(crop).shape), (3, CROP_H, CROP_W))

    def test_budget_monotonic(self):
        far = estimate_budget(b_x=600, ink_x0=100, page_w=1024, prefix_len=40)
        near = estimate_budget(b_x=850, ink_x0=100, page_w=1024, prefix_len=60)
        self.assertGreater(far, near)
        self.assertEqual(estimate_budget(b_x=1000, ink_x0=100, page_w=1024, prefix_len=80), 0.0)

    @unittest.skipUnless(os.path.isfile(FONT), "times.ttf not available")
    def test_render_and_burn_produces_consistent_records(self):
        words = ("the quick brown fox jumps over the lazy dog and keeps running far away " * 40).split()
        page, used = render_page(words, FONT, random.Random(0))
        self.assertGreater(len(page.lines), 5)
        self.assertGreater(used, 50)
        _, records, crops = burn_page_to_records(page, SideBurnGenerator(Config()), seed=1)
        self.assertGreater(len(records), 0)
        self.assertEqual(len(records), len(crops))
        for r in records:
            # prefix + target reconstructs the rendered line exactly
            self.assertEqual(r["prefix"] + r["target"], r["full_line"])
            self.assertGreater(r["budget"], 0)
            self.assertGreaterEqual(r["budget"] + 6, len(r["target"].strip()) * 0.5)

    @unittest.skipUnless(os.path.isfile(FONT), "times.ttf not available")
    def test_line_cache_dataset_roundtrip(self):
        from transformers import AutoTokenizer
        from dataset import LineCacheDataset

        words = ("it was the best of times it was the worst of times " * 60).split()
        page, _ = render_page(words, FONT, random.Random(3))
        _, records, crops = burn_page_to_records(page, SideBurnGenerator(Config()), seed=3)
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "test"
            sd.mkdir()
            np.save(sd / "crops.npy", np.stack(crops))
            (sd / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            tok = AutoTokenizer.from_pretrained("t5-small")
            ds = LineCacheDataset(d, "test", tok)
            batch = LineCacheDataset.make_collate(tok.pad_token_id)([ds[0], ds[1]])
            self.assertEqual(batch["line_burn_crop"].shape[1:], (3, CROP_H, CROP_W))
            self.assertEqual(batch["input_ids"].shape, batch["attention_mask"].shape)
            label0 = [t for t in batch["labels"][0].tolist() if t != -100]
            self.assertEqual(extract_infill(tok.decode(label0, skip_special_tokens=False)),
                             records[0]["target"].strip())
            del ds  # release memmap before temp dir cleanup (Windows)


if __name__ == "__main__":
    unittest.main()
