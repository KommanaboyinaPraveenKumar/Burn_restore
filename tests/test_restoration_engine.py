"""
tests/test_restoration_engine.py
================================
Unit tests for the central RestorationEngine inference component.
"""

import os
import unittest
import numpy as np
import torch
import cv2

from inference.restoration_engine import RestorationEngine
from inference.burn_detector import SeveredLineRegion
from config import Config


class TestRestorationEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = "checkpoints/best_checkpoint.pt"
        cls.has_ckpt = os.path.isfile(cls.ckpt_path)
        cls.cfg = Config()
        cls.cfg.batch_size = 1

    def test_checkpoint_validation_nonexistent_file(self):
        engine = RestorationEngine(checkpoint_path=None, lazy_load=True)
        is_valid, msg, meta = engine.validate_checkpoint("non_existent_ckpt.pt")
        self.assertFalse(is_valid)
        self.assertIn("does not exist", msg)

    def test_checkpoint_validation_valid_file(self):
        if not self.has_ckpt:
            self.skipTest("checkpoints/best_checkpoint.pt not available")
        engine = RestorationEngine(checkpoint_path=None, lazy_load=True)
        is_valid, msg, meta = engine.validate_checkpoint(self.ckpt_path)
        self.assertTrue(is_valid)
        self.assertIn("file_size_mb", meta)
        self.assertGreater(meta["file_size_mb"], 0)

    def test_engine_initialization_and_status(self):
        engine = RestorationEngine(
            checkpoint_path=self.ckpt_path if self.has_ckpt else None,
            device="cpu",
            config=self.cfg,
        )
        self.assertTrue(engine.is_loaded)
        status = engine.get_model_status()
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["device"], "cpu")
        self.assertIn("ForensicT5Restorer", status["model_architecture"])
        self.assertGreater(status["total_parameters"], 50_000_000)

    def test_preprocess_image_resizing(self):
        engine = RestorationEngine(checkpoint_path=None, lazy_load=True, config=self.cfg)
        raw_img = np.full((600, 400, 3), 255, dtype=np.uint8)
        processed = engine.preprocess_image(raw_img)
        expected_w, expected_h = self.cfg.page_resize
        self.assertEqual(processed.shape, (expected_h, expected_w, 3))

    def test_extract_forensic_crop_tensor_shape(self):
        engine = RestorationEngine(checkpoint_path=None, lazy_load=True, config=self.cfg)
        img = np.full((1024, 1024, 3), 200, dtype=np.uint8)
        crop_bbox = (800, 100, 1024, 130)
        resized, tensor = engine.extract_forensic_crop(img, crop_bbox)
        target_h, target_w = self.cfg.line_crop_size
        self.assertEqual(resized.shape, (target_h, target_w, 3))
        self.assertEqual(tensor.shape, (1, 3, target_h, target_w))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_single_region_restoration(self):
        engine = RestorationEngine(
            checkpoint_path=self.ckpt_path if self.has_ckpt else None,
            device="cpu",
            config=self.cfg,
        )
        img = np.full((1024, 1024, 3), 240, dtype=np.uint8)
        region = SeveredLineRegion(
            line_idx=0,
            y0=100,
            y1=130,
            b_x=850,
            crop_bbox=(670, 94, 1024, 136),
            damaged_width_px=174,
            intact_width_px=850,
            line_height_px=30,
        )
        res = engine.restore_region(
            image=img,
            region=region,
            prefix_text="Scientific analysis of neural",
            budget=12.0,
            max_new_tokens=10,
        )
        self.assertIn("predicted_restoration", res)
        self.assertIn("physical_budget", res)
        self.assertIn("inference_time_ms", res)
        self.assertGreaterEqual(res["inference_time_ms"], 0.0)

    def test_full_page_restoration(self):
        engine = RestorationEngine(
            checkpoint_path=self.ckpt_path if self.has_ckpt else None,
            device="cpu",
            config=self.cfg,
        )
        # Create page with dark right margin
        img = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        for y in [100, 200]:
            cv2.line(img, (50, y), (850, y), (0, 0, 0), 3)
        img[:, 900:] = (20, 15, 12)  # Dark charcoal margin

        res = engine.restore_page(img, max_lines=2)
        self.assertIn("burn_detection", res)
        self.assertIn("restorations", res)
        self.assertIn("total_page_inference_ms", res)


if __name__ == "__main__":
    unittest.main()

