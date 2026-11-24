"""
tests/test_inference_integration.py
===================================
End-to-end integration tests for document restoration CLI and real fixture inference.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

from inference.restoration_engine import RestorationEngine
from config import Config

_repo_root = Path(__file__).resolve().parent.parent


class TestInferenceIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_img_path = str(_repo_root / "tests" / "fixtures" / "mini_dataset" / "clean_images" / "sample_page_006.png")
        cls.test_txt_path = str(_repo_root / "tests" / "fixtures" / "mini_dataset" / "clean_texts" / "sample_page_006.txt")
        cls.output_dir = str(_repo_root / "results" / "test_integration_output")
        os.makedirs(cls.output_dir, exist_ok=True)

        cls.has_fixture = os.path.isfile(cls.test_img_path)
        cls.ckpt_path = str(_repo_root / "checkpoints" / "best_checkpoint.pt")
        cls.has_ckpt = os.path.isfile(cls.ckpt_path)

    def test_fixture_page_restoration(self):
        if not self.has_fixture:
            self.skipTest(f"Fixture image not found at {self.test_img_path}")

        cfg = Config()
        cfg.batch_size = 1
        engine = RestorationEngine(
            checkpoint_path=self.ckpt_path if self.has_ckpt else None,
            device="cpu",
            config=cfg,
        )

        overlay_p = os.path.join(self.output_dir, "sample_page_006_overlay.png")
        results = engine.restore_page(
            image=self.test_img_path,
            max_lines=2,
            max_new_tokens=10,
            save_overlay_path=overlay_p,
        )

        self.assertIn("burn_detection", results)
        self.assertIn("restorations", results)
        self.assertGreaterEqual(results["restored_lines_count"], 1)

        first_restoration = results["restorations"][0]
        self.assertIn("surviving_prefix", first_restoration)
        self.assertIn("predicted_restoration", first_restoration)
        self.assertIn("physical_budget", first_restoration)

        # Verify overlay was created
        self.assertTrue(os.path.isfile(overlay_p))
        self.assertGreater(os.path.getsize(overlay_p), 1000)

    def test_cli_execution_end_to_end(self):
        if not self.has_fixture:
            self.skipTest(f"Fixture image not found at {self.test_img_path}")

        cli_script = str(_repo_root / "scripts" / "restore_document.py")
        cli_out = os.path.join(self.output_dir, "cli_run")

        cmd = [
            sys.executable,
            cli_script,
            "--input", self.test_img_path,
            "--output", cli_out,
            "--max-lines", "2",
            "--max-new-tokens", "10",
            "--save-overlay",
            "--save-json",
        ]

        # If checkpoint exists, pass it
        if self.has_ckpt:
            cmd.extend(["--checkpoint", self.ckpt_path])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, f"CLI failed with error:\n{result.stderr}")

        # Check console output
        self.assertIn("GROUNDED SIDE-BURNT DOCUMENT TEXT RESTORATION CLI", result.stdout)
        self.assertIn("RESTORATION SUMMARY", result.stdout)

        # Check output JSON file
        expected_json = os.path.join(cli_out, "sample_page_006_restoration_report.json")
        self.assertTrue(os.path.isfile(expected_json))

        with open(expected_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("burn_detection", data)
        self.assertIn("restorations", data)


if __name__ == "__main__":
    unittest.main()

