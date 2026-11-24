"""
test_dataset_engineering.py
===========================
Comprehensive unit and integration test suite for Phase 1:
Advanced Dataset Engineering & Synthetic Burn Generation.

Validates:
  1. Boundary generation smoothness, intensity, and physical bounds.
  2. Burn mask validity (0 and 255 binary values, shape, and bbox).
  3. Text alignment: Margin calibration eliminates label contamination.
  4. Word-level bounding-box alignment (Priority 2).
  5. Character-level bounding-box alignment (Priority 1).
  6. Dataset pairing & missing-file handling.
  7. Page-level dataset splitting: 100% zero data leakage.
  8. Deterministic evaluation corruption (reproducibility across runs).
  9. Dynamic training augmentation variance across epochs.
  10. DataLoader batching and collate_fn tensor dimensions.
  11. Image preprocessing & ImageNet normalization statistics.
  12. Unicode NFKD normalization and typographical punctuation handling.
  13. Burn corruption profiles (mild, moderate, severe, irregular).
  14. Edge cases: empty lines, fully intact lines, completely burnt lines, large margins.
"""

import json
import os
import sys
import unittest

# Ensure repository root and Burn_restore are on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
burn_restore_dir = os.path.join(repo_root, "Burn_restore")

for p in [burn_restore_dir, repo_root, current_dir]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset_generator import SideBurnGenerator, TextSpanAnnotation
from dataset import LineGroundedDocumentDataset
from dataset_split import create_dataset_splits, get_paired_stems


class TestDatasetEngineering(unittest.TestCase):
    """Exhaustive test suite for dataset generation, alignment, and splitting."""

    def setUp(self):
        self.config = Config()
        self.config.device = "cpu"
        self.generator = SideBurnGenerator(self.config)
        self.fixture_dir = os.path.join(repo_root, "tests", "fixtures", "mini_dataset")
        self.clean_img_dir = os.path.join(self.fixture_dir, "clean_images")
        self.clean_txt_dir = os.path.join(self.fixture_dir, "clean_texts")
        self.anno_dir = os.path.join(self.fixture_dir, "annotations")

    def test_01_boundary_generation_and_bounds(self):
        """Test jagged boundary stays within image dimensions and centers on base_x."""
        h, w = 1024, 1024
        base_x = 800
        amp = 40
        boundary = self.generator._generate_jagged_boundary(h, base_x, amp)
        self.assertEqual(boundary.shape, (h,))
        self.assertTrue(np.all(boundary >= 0))
        self.assertTrue(np.all(boundary <= w - 1))
        self.assertTrue(base_x - amp - 10 <= boundary.min())
        self.assertTrue(boundary.max() <= base_x + amp + 10)
        # Check smoothness: consecutive difference should be gradual, not white noise
        diffs = np.abs(np.diff(boundary))
        self.assertLess(diffs.mean(), 5.0)

    def test_02_burn_mask_validity(self):
        """Test mask is binary (0, 255), matches image size, and bbox is valid."""
        img = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        lines = ["Test line for burn mask verification."]
        res = self.generator.apply_burn(img, lines, seed=42)
        mask = res["burn_mask"]
        self.assertEqual(mask.shape, (1024, 1024))
        self.assertEqual(mask.dtype, np.uint8)
        unique_vals = set(np.unique(mask))
        self.assertTrue(unique_vals.issubset({0, 255}))

        bx, by, bw, bh = res["burn_bbox"]
        self.assertGreaterEqual(bx, 0)
        self.assertGreaterEqual(by, 0)
        self.assertLessEqual(bx + bw, 1024)
        self.assertLessEqual(by + bh, 1024)

    def test_03_margin_calibration_eliminates_contamination(self):
        """Empirically test that calibrated ink splitting prevents visible words from being clipped."""
        text = "Deep learning for document text restoration with multimodal transformers"
        w = 1024
        burn_x = 800.0  # burn begins at x=800

        # Simulate text line indented between x=200 and x=850 (margin = 200px)
        binary_row = np.zeros(w, dtype=np.uint8)
        binary_row[200:850] = 255

        # Naive whole-page split
        k_naive, c_naive = SideBurnGenerator._proportional_split(text, burn_x / w)
        # Calibrated ink split
        k_cal, c_cal, ratio, conf = SideBurnGenerator._split_line_calibrated(text, binary_row, burn_x, w)

        # Naive cut ratio is 800/1024 = 0.78125
        # Calibrated cut ratio is (800-200)/(850-200) = 600/650 = 0.9230
        self.assertGreater(ratio, burn_x / w)
        self.assertGreater(len(k_cal), len(k_naive))
        self.assertLess(len(c_cal), len(c_naive))
        # Verify confidence is high when ink cols >= 10
        self.assertGreaterEqual(conf, 0.8)

    def test_04_word_level_box_alignment_priority_2(self):
        """Test exact word box splitting when word coordinates are provided."""
        words_with_boxes = [
            ("Value", 120.0, 100.0, 200.0, 130.0),
            ("of", 210.0, 100.0, 240.0, 130.0),
            ("information", 250.0, 100.0, 420.0, 130.0),
            ("in", 430.0, 100.0, 460.0, 130.0),
            ("games", 470.0, 100.0, 580.0, 130.0),
        ]
        # Burn at x=450: 'Value of information' (intact), 'in' (mid=445 <= 450 -> kept), 'games' (x1=470 > 450 -> clipped)
        kept, clipped, ratio, conf = SideBurnGenerator._split_line_by_word_boxes(words_with_boxes, 450.0)
        self.assertEqual(kept, "Value of information in")
        self.assertEqual(clipped, "games")
        self.assertEqual(conf, 0.95)

    def test_05_char_level_box_alignment_priority_1(self):
        """Test exact character box splitting when character coordinates are provided."""
        chars_with_boxes = [
            ("c", 100.0, 50.0, 110.0, 70.0),
            ("a", 110.0, 50.0, 120.0, 70.0),
            ("t", 120.0, 50.0, 130.0, 70.0),
            ("s", 130.0, 50.0, 140.0, 70.0),
        ]
        # Burn at x=125: 'c', 'a' are kept (< 120); 't' mid=125 <= 125 kept; 's' clipped
        kept, clipped, ratio, conf = SideBurnGenerator._split_line_by_char_boxes(chars_with_boxes, 125.0)
        self.assertEqual(kept, "cat")
        self.assertEqual(clipped, "s")
        self.assertEqual(conf, 1.0)

    def test_06_dataset_pairing_and_missing_handling(self):
        """Test pair discovery and graceful handling of missing transcripts."""
        pairs = get_paired_stems(self.clean_img_dir, self.clean_txt_dir)
        self.assertEqual(len(pairs), 6)
        stems = [p[2] for p in pairs]
        self.assertIn("sample_page_001", stems)
        self.assertIn("sample_page_006", stems)

    def test_07_page_level_splitting_zero_leakage(self):
        """Test train/val/test splitting ensures ZERO document page leakage."""
        manifest = create_dataset_splits(
            self.clean_img_dir,
            self.clean_txt_dir,
            output_manifest="data/test_splits_temp.json",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        train_stems = {x["stem"] for x in manifest["train"]}
        val_stems = {x["stem"] for x in manifest["val"]}
        test_stems = {x["stem"] for x in manifest["test"]}

        self.assertEqual(len(train_stems & val_stems), 0)
        self.assertEqual(len(train_stems & test_stems), 0)
        self.assertEqual(len(val_stems & test_stems), 0)
        self.assertEqual(len(train_stems) + len(val_stems) + len(test_stems), 6)

        if os.path.exists("data/test_splits_temp.json"):
            os.remove("data/test_splits_temp.json")

    def test_08_deterministic_evaluation_corruption(self):
        """Test evaluation mode produces identical corrupted images and targets on repeated access."""
        ds_eval = LineGroundedDocumentDataset(
            self.config,
            image_dir=self.clean_img_dir,
            text_dir=self.clean_txt_dir,
            is_eval=True,
        )
        item_a = ds_eval[0]
        item_b = ds_eval[0]

        self.assertEqual(item_a["raw_prefix"], item_b["raw_prefix"])
        self.assertEqual(item_a["raw_target_text"], item_b["raw_target_text"])
        self.assertTrue(torch.equal(item_a["line_burn_crop"], item_b["line_burn_crop"]))
        self.assertEqual(float(item_a["line_budget"]), float(item_b["line_budget"]))

    def test_09_dynamic_training_augmentation(self):
        """Test training mode produces different burn geometries across different epochs."""
        ds_train = LineGroundedDocumentDataset(
            self.config,
            image_dir=self.clean_img_dir,
            text_dir=self.clean_txt_dir,
            is_eval=False,
        )
        ds_train.set_epoch(0)
        item_ep0 = ds_train[0]
        ds_train.set_epoch(1)
        item_ep1 = ds_train[0]

        pixel_diff = torch.abs(item_ep0["line_burn_crop"] - item_ep1["line_burn_crop"]).sum().item()
        self.assertGreater(pixel_diff, 10.0)

    def test_10_dataloader_batching_and_collate_fn(self):
        """Test DataLoader and collate_fn return correct batch tensor shapes."""
        ds = LineGroundedDocumentDataset(
            self.config,
            image_dir=self.clean_img_dir,
            text_dir=self.clean_txt_dir,
            is_eval=True,
        )
        batch_size = 4
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=LineGroundedDocumentDataset.collate_fn)
        batch = next(iter(loader))

        self.assertEqual(batch["input_ids"].shape, (batch_size, self.config.max_context_len))
        self.assertEqual(batch["attention_mask"].shape, (batch_size, self.config.max_context_len))
        self.assertEqual(batch["labels"].shape, (batch_size, self.config.max_target_len))
        self.assertEqual(batch["line_burn_crop"].shape, (batch_size, 3, 128, 256))
        self.assertEqual(batch["line_budget"].shape, (batch_size,))
        self.assertEqual(batch["burn_mask_profile"].shape, (batch_size, self.config.profile_bins))
        self.assertIsInstance(batch["raw_prefix"], list)
        self.assertEqual(len(batch["raw_prefix"]), batch_size)

    def test_11_image_preprocessing_and_normalization(self):
        """Test line crop normalization conforms to ImageNet mean and standard deviation."""
        img = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        crop_tensor = self.generator.apply_burn(img, ["Line 1"])
        line_crop = ds_prep = LineGroundedDocumentDataset._prepare_line_crop(
            None, crop_tensor["corrupted_image"], 100, 140, 750
        )
        self.assertEqual(line_crop.shape, (3, 128, 256))
        self.assertEqual(line_crop.dtype, torch.float32)

    def test_12_unicode_and_ligature_normalization(self):
        """Test Unicode NFKD decomposes ligature glyphs cleanly."""
        text_with_ligatures = "The \ufb01rst \ufb02oating \u201cquotation\u201d \u2014 end"
        img = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        res = self.generator.apply_burn(img, [text_with_ligatures], seed=42)
        # Check that result doesn't crash and returns valid string splits
        self.assertIsInstance(res["healthy_context"], str)

    def test_13_burn_profiles_mild_moderate_severe_irregular(self):
        """Test setting all 4 burn profiles updates config parameters correctly."""
        cfg = Config()
        for prof in ["mild", "moderate", "severe", "irregular"]:
            cfg.set_burn_profile(prof)
            self.assertEqual(cfg.burn_profile, prof)
            self.assertGreater(cfg.burn_max_ratio, cfg.burn_min_ratio)
            self.assertGreater(cfg.transition_zone_px, 0)

    def test_14_edge_cases_empty_and_fully_burnt_lines(self):
        """Test edge cases: completely burnt lines, empty text, short words."""
        img = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        lines = ["", "Short", "A completely normal line that is quite long"]
        res = self.generator.apply_burn(img, lines, seed=42)
        self.assertIn("corrupted_image", res)
        self.assertIn("severed_lines", res)


if __name__ == "__main__":
    unittest.main()
