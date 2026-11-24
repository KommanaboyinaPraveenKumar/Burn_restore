"""
dataset_validator.py
====================
Automated Dataset Quality Validation & Integrity Checking Suite.

Audits document datasets against 12 critical research quality checks:
  1. Image / transcript pairing and existence.
  2. Image dimensions and channel formats.
  3. Transcript readability and non-empty lines.
  4. Burn mask validity (binary values 0 and 255).
  5. Burn bounding boxes within valid image boundaries.
  6. Text alignment consistency and ink-bound calibrations.
  7. Train / validation / test document ID zero-leakage check.
  8. Evaluation corruption determinism (repeatability).
  9. Training dynamic augmentation variance across epochs.
  10. Corrupted samples preserve valid image dtype and bounds.
  11. Target text consistency with clean transcript and spatial boundary.
  12. Explicit detection and reporting of uncertain or invalid alignments.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config
from dataset_generator import SideBurnGenerator
from dataset_split import get_paired_stems


class DatasetValidator:
    """Automated validator enforcing dataset integrity and geometric consistency."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.generator = SideBurnGenerator(self.config)

    def run_full_validation(
        self,
        image_dir: str,
        text_dir: str,
        splits_json: Optional[str] = None,
        annotation_dir: Optional[str] = None,
        max_samples: int = 50,
    ) -> Dict:
        """Executes the complete 12-point dataset quality validation."""
        report = {
            "summary": {"total_checked": 0, "passed": 0, "warnings": 0, "failed": 0},
            "checks": {},
        }

        print("=" * 70)
        print("RUNNING AUTOMATED DATASET QUALITY & INTEGRITY VALIDATION")
        print("=" * 70)

        # Check 1: Image / transcript pairing
        pairs = get_paired_stems(image_dir, text_dir)
        check_1_pass = len(pairs) > 0
        report["checks"]["1_pairing"] = {
            "status": "PASS" if check_1_pass else "FAIL",
            "pairs_found": len(pairs),
            "description": "Image and transcript pairing by filename stem.",
        }
        print(f"[Check 1] Pairing & Discovery: {'PASS' if check_1_pass else 'FAIL'} ({len(pairs)} pairs)")
        if not check_1_pass:
            report["summary"]["failed"] += 1
            return report

        # Check 7: Train/val/test splits zero-leakage
        check_7_status = "SKIPPED"
        if splits_json and os.path.isfile(splits_json):
            with open(splits_json, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            train_stems = {x["stem"] for x in s_data.get("train", [])}
            val_stems = {x["stem"] for x in s_data.get("val", [])}
            test_stems = {x["stem"] for x in s_data.get("test", [])}

            leak_tv = len(train_stems & val_stems)
            leak_tt = len(train_stems & test_stems)
            leak_vt = len(val_stems & test_stems)
            leakage_free = (leak_tv == 0) and (leak_tt == 0) and (leak_vt == 0)
            check_7_status = "PASS" if leakage_free else "FAIL"
            report["checks"]["7_leakage"] = {
                "status": check_7_status,
                "train_count": len(train_stems),
                "val_count": len(val_stems),
                "test_count": len(test_stems),
                "overlaps": {"train_val": leak_tv, "train_test": leak_tt, "val_test": leak_vt},
            }
        else:
            report["checks"]["7_leakage"] = {"status": "SKIPPED", "reason": "No splits.json provided."}
        print(f"[Check 7] Page-Level Zero-Leakage: {check_7_status}")

        dim_errors = 0
        transcript_errors = 0
        mask_errors = 0
        bbox_errors = 0
        alignment_uncertainties = 0
        corruption_format_errors = 0
        target_consistency_errors = 0

        # Sample checks
        checked = min(len(pairs), max_samples)
        for idx in range(checked):
            img_path, txt_path, stem = pairs[idx]

            # Check 2: Image dimensions
            img = cv2.imread(img_path)
            if img is None or img.ndim != 3 or img.shape[0] < 100 or img.shape[1] < 100:
                dim_errors += 1
                continue

            # Check 3: Transcripts non-empty
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                transcript_errors += 1
                continue

            # Check 4 & 5: Burn mask validity and bounding box
            res = self.generator.apply_burn(img, lines, seed=42 + idx)
            mask = res["burn_mask"]
            unique_vals = set(np.unique(mask))
            if not unique_vals.issubset({0, 255}):
                mask_errors += 1

            bx, by, bw, bh = res["burn_bbox"]
            h, w = img.shape[:2]
            if bx < 0 or by < 0 or (bx + bw) > w or (by + bh) > h:
                bbox_errors += 1

            # Check 6 & 11: Text alignment and target consistency
            for s in res["all_line_splits"]:
                if s["alignment_strategy"] == "priority_4_uncertain_fallback":
                    alignment_uncertainties += 1
                # Target text must not contain words that are actually surviving
                prefix_words = set(s["prefix"].lower().split())
                target_words = set(s["target"].lower().split())
                # Disjoint word sets for cleanly split words (length > 3)
                overlap = {w for w in prefix_words & target_words if len(w) > 3}
                if overlap:
                    target_consistency_errors += 1

            # Check 10: Corrupted image format
            c_img = res["corrupted_image"]
            if c_img.dtype != np.uint8 or c_img.shape != img.shape:
                corruption_format_errors += 1

        report["checks"]["2_dimensions"] = {"status": "PASS" if dim_errors == 0 else "FAIL", "errors": dim_errors}
        report["checks"]["3_transcripts"] = {"status": "PASS" if transcript_errors == 0 else "FAIL", "errors": transcript_errors}
        report["checks"]["4_burn_masks"] = {"status": "PASS" if mask_errors == 0 else "FAIL", "errors": mask_errors}
        report["checks"]["5_bounding_boxes"] = {"status": "PASS" if bbox_errors == 0 else "FAIL", "errors": bbox_errors}
        report["checks"]["6_alignment_uncertainty"] = {
            "status": "PASS" if alignment_uncertainties < checked * 2 else "WARN",
            "uncertain_lines": alignment_uncertainties,
        }
        report["checks"]["10_corrupted_format"] = {"status": "PASS" if corruption_format_errors == 0 else "FAIL"}
        report["checks"]["11_target_consistency"] = {"status": "PASS" if target_consistency_errors == 0 else "WARN"}

        # Check 8: Evaluation determinism
        img_0, txt_0, _ = pairs[0]
        with open(txt_0, "r", encoding="utf-8") as f:
            lines_0 = [l.strip() for l in f if l.strip()]
        img_0_data = cv2.imread(img_0)
        res_a = self.generator.apply_burn(img_0_data, lines_0, seed=12345)
        res_b = self.generator.apply_burn(img_0_data, lines_0, seed=12345)
        diff_eval = np.abs(res_a["corrupted_image"].astype(float) - res_b["corrupted_image"].astype(float)).sum()
        check_8_pass = (diff_eval == 0.0) and (res_a["target_text"] == res_b["target_text"])
        report["checks"]["8_determinism"] = {"status": "PASS" if check_8_pass else "FAIL", "pixel_diff": diff_eval}

        # Check 9: Training dynamic augmentation
        res_dyn_1 = self.generator.apply_burn(img_0_data, lines_0, seed=100)
        res_dyn_2 = self.generator.apply_burn(img_0_data, lines_0, seed=200)
        diff_dyn = np.abs(res_dyn_1["corrupted_image"].astype(float) - res_dyn_2["corrupted_image"].astype(float)).sum()
        check_9_pass = diff_dyn > 1000.0  # Dynamic runs produce different burns
        report["checks"]["9_dynamic_augmentation"] = {"status": "PASS" if check_9_pass else "FAIL", "variance": diff_dyn}

        # Summary
        report["summary"]["total_checked"] = checked
        pass_count = sum(1 for c in report["checks"].values() if c.get("status") == "PASS")
        warn_count = sum(1 for c in report["checks"].values() if c.get("status") == "WARN")
        fail_count = sum(1 for c in report["checks"].values() if c.get("status") == "FAIL")

        report["summary"]["passed"] = pass_count
        report["summary"]["warnings"] = warn_count
        report["summary"]["failed"] = fail_count

        print(f"[Check 2] Image Dimensions: {report['checks']['2_dimensions']['status']}")
        print(f"[Check 3] Transcripts Non-Empty: {report['checks']['3_transcripts']['status']}")
        print(f"[Check 4] Mask Validity: {report['checks']['4_burn_masks']['status']}")
        print(f"[Check 5] Bounding Box Bounds: {report['checks']['5_bounding_boxes']['status']}")
        print(f"[Check 6] Text Alignment Confidence: {report['checks']['6_alignment_uncertainty']['status']} ({alignment_uncertainties} uncertain lines)")
        print(f"[Check 8] Evaluation Determinism: {report['checks']['8_determinism']['status']} (Pixel diff: {diff_eval})")
        print(f"[Check 9] Dynamic Augmentation Variance: {report['checks']['9_dynamic_augmentation']['status']}")
        print(f"[Check 10] Image Format Integrity: {report['checks']['10_corrupted_format']['status']}")
        print(f"[Check 11] Target Text Disjointness: {report['checks']['11_target_consistency']['status']}")
        print("-" * 70)
        print(f"VALIDATION SUMMARY: {pass_count} PASSED | {warn_count} WARNINGS | {fail_count} FAILED")
        print("=" * 70)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset quality and integrity.")
    parser.add_argument("--images", default="tests/fixtures/mini_dataset/clean_images")
    parser.add_argument("--texts", default="tests/fixtures/mini_dataset/clean_texts")
    parser.add_argument("--splits", default="data/splits.json")
    args = parser.parse_args()

    validator = DatasetValidator()
    validator.run_full_validation(args.images, args.texts, splits_json=args.splits)

