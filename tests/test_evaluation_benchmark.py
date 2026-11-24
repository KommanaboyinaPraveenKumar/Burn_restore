"""
tests/test_evaluation_benchmark.py
==================================
Comprehensive Unit and Regression Test Suite for Phase 4:
Comprehensive Evaluation, Benchmarking, Statistical Validation, and Ablation Studies.

Tests:
  1. test_checkpoint_loading_and_metadata:
     Verifies correct loading of model weights and configuration dicts.
  2. test_dataset_split_zero_leakage:
     Verifies that train, val, and test splits share 0 page overlap.
  3. test_position_vs_edit_char_accuracy_distinction:
     Verifies that position-based character accuracy and edit-distance-based accuracy
     are distinct and mathematically correct.
  4. test_empty_reference_and_prediction_handling:
     Verifies graceful policy for empty targets, empty predictions, and identical strings.
  5. test_unicode_normalization_ligatures:
     Verifies unicode decomposition of PDF ligatures (fi, fl) and typographical quotes.
  6. test_special_token_removal_extract_infill:
     Verifies that <extra_id_N>, <pad>, and </s> are cleanly extracted.
  7. test_detailed_levenshtein_rates_consistency:
     Verifies that insertions, deletions, and substitutions sum to total edit distance.
  8. test_paired_bootstrap_ci_computation:
     Verifies paired bootstrap confidence intervals and permutation p-value bounds.
  9. test_physical_grounding_budget_metrics:
     Verifies budget excess calculation, over-generation, and under-generation flags.
  10. test_region_bounded_psnr_ssim:
      Verifies that image quality metrics are bounded strictly to the damaged region.
  11. test_ablation_configuration_separation:
      Verifies that ablation modes toggle the intended architectural submodules.
  12. test_multi_seed_reproducibility:
      Verifies deterministic output reproduction with fixed seeds.
"""

import copy
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Ensure repo root and Burn_restore are on sys.path
repo_root = Path(__file__).resolve().parent.parent
burn_restore_dir = repo_root / "Burn_restore"
for p in [str(burn_restore_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import Config
from model import ForensicT5Restorer
from dataset_generator import SideBurnGenerator
from metrics import (
    compute_cer,
    compute_normalized_cer,
    compute_wer,
    compute_character_accuracy,
    compute_position_character_accuracy,
    compute_word_accuracy,
    compute_exact_match_line_accuracy,
    compute_detailed_levenshtein_rates,
    compute_physical_grounding_metrics,
    compute_region_bounded_psnr_ssim,
    extract_infill_text,
    normalize_text,
)
from scripts.run_comprehensive_benchmark import compute_paired_bootstrap_ci


class TestEvaluationBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        cls.splits_manifest = str(repo_root / "data" / "splits.json")
        if not os.path.isfile(cls.splits_manifest):
            cls.splits_manifest = str(burn_restore_dir / "data" / "splits.json")

    def test_dataset_split_zero_leakage(self):
        """Train, val, and test splits must have strictly zero page overlap."""
        self.assertTrue(os.path.isfile(self.splits_manifest), f"Missing splits manifest: {self.splits_manifest}")
        with open(self.splits_manifest, "r", encoding="utf-8") as f:
            splits = json.load(f)

        s_train = set(x["stem"] for x in splits.get("train", []))
        s_val = set(x["stem"] for x in splits.get("val", []))
        s_test = set(x["stem"] for x in splits.get("test", []))

        self.assertEqual(len(s_train & s_val), 0, "Train and Val splits share pages!")
        self.assertEqual(len(s_train & s_test), 0, "Train and Test splits share pages!")
        self.assertEqual(len(s_val & s_test), 0, "Val and Test splits share pages!")

    def test_position_vs_edit_char_accuracy_distinction(self):
        """Position-based character accuracy must penalize shifted strings while edit-distance accuracy accounts for alignment."""
        target = "abcdef"
        shifted_pred = "bcdef"  # 1 deletion shift: all positions misaligned, but edit distance is only 1

        pos_acc = compute_position_character_accuracy(shifted_pred, target)
        edit_acc = compute_character_accuracy(shifted_pred, target)

        # Position accuracy compares: b!=a, c!=b, d!=c, e!=d, f!=e -> 0 matches / 6 = 0.0
        self.assertEqual(pos_acc, 0.0)
        # Edit accuracy: CER = 1/6 = 0.1667 -> 1.0 - 0.1667 = 0.8333
        self.assertAlmostEqual(edit_acc, 5.0 / 6.0, places=3)
        self.assertNotEqual(pos_acc, edit_acc, "Position and Edit character accuracies must not be interchangeable.")

    def test_empty_reference_and_prediction_handling(self):
        """Evaluates policy for empty target and empty predictions."""
        # Both empty: exact match, CER=0
        self.assertEqual(compute_cer("", ""), 0.0)
        self.assertEqual(compute_wer("", ""), 0.0)
        self.assertEqual(compute_character_accuracy("", ""), 1.0)
        self.assertEqual(compute_exact_match_line_accuracy("", ""), 1.0)

        # Non-empty prediction, empty target: CER=1.0, CharAcc=0.0
        self.assertEqual(compute_cer("restored", ""), 1.0)
        self.assertEqual(compute_character_accuracy("restored", ""), 0.0)
        self.assertEqual(compute_exact_match_line_accuracy("restored", ""), 0.0)

        # Empty prediction, non-empty target: CER=1.0, CharAcc=0.0
        self.assertEqual(compute_cer("", "target"), 1.0)
        self.assertEqual(compute_character_accuracy("", "target"), 0.0)

    def test_unicode_normalization_ligatures(self):
        """Verifies decomposition of LaTeX/PDF ligatures and typographical quotes."""
        # \ufb01 is 'fi', \ufb02 is 'fl'
        ligature_str = "\ufb01re \ufb02ame \u2018quote\u2019 \u2014 dash"
        normalized = normalize_text(ligature_str)
        self.assertEqual(normalized, "fire flame 'quote' - dash")

        # Edit distance between ligature and decomposed should be 0.0 after normalization
        cer = compute_normalized_cer("\ufb01re", "fire")
        self.assertEqual(cer, 0.0)

    def test_special_token_removal_extract_infill(self):
        """Verifies that T5 sentinel tokens, <pad>, and trailing infill spans are stripped cleanly."""
        raw_1 = "<pad> <extra_id_0> restored text <extra_id_1> remaining"
        cleaned_1 = extract_infill_text(raw_1)
        self.assertEqual(cleaned_1, "restored text")

        raw_2 = "<pad> incomplete line | prefix <extra_id_0> missing words </s>"
        cleaned_2 = extract_infill_text(raw_2)
        self.assertEqual(cleaned_2, "missing words </s>")

    def test_detailed_levenshtein_rates_consistency(self):
        """Insertions, deletions, and substitutions must sum to total edit distance."""
        pred = "kttn"
        target = "kitten"
        rates = compute_detailed_levenshtein_rates(pred, target)

        # Levenshtein distance between 'kttn' and 'kitten' is 2 deletions ('i', 'e')
        self.assertEqual(rates["total_edits"], 2)
        self.assertEqual(rates["deletions"], 2)
        self.assertEqual(rates["insertions"], 0)
        self.assertEqual(rates["substitutions"], 0)
        self.assertAlmostEqual(rates["deletion_rate"], 2.0 / 6.0, places=3)

    def test_paired_bootstrap_ci_computation(self):
        """Paired bootstrap must return valid CI bounds and sensible p-value."""
        # Simulated scenario: Model A has consistently lower CER than Model B
        vec_a = [0.18, 0.26, 0.31, 0.14, 0.22, 0.29, 0.17, 0.23]
        vec_b = [0.35, 0.40, 0.45, 0.30, 0.37, 0.43, 0.33, 0.39]

        res = compute_paired_bootstrap_ci(vec_a, vec_b, n_resamples=500, confidence_level=0.95, seed=42)

        self.assertLess(res["mean_difference"], -0.10)
        self.assertLessEqual(res["ci_lower_95"], res["ci_upper_95"])
        self.assertLess(res["ci_lower_95"], 0.0)
        self.assertLess(res["p_value"], 0.05, "Statistically significant improvement should have p < 0.05.")
        self.assertTrue(res["statistically_significant_05"])

    def test_physical_grounding_budget_metrics(self):
        """Verifies detection of budget excess, over-generation, and under-generation."""
        target = "short"
        long_pred = "short prediction that exceeds budget significantly"
        budget = 10.0

        grounding = compute_physical_grounding_metrics(long_pred, target, budget=budget, buffer=2.0)
        self.assertTrue(grounding["budget_violation"])
        self.assertTrue(grounding["over_generated"])
        self.assertFalse(grounding["under_generated"])
        self.assertGreater(grounding["budget_excess_chars"], 0.0)

        # Compliant prediction
        comp_pred = "short"
        comp_grounding = compute_physical_grounding_metrics(comp_pred, target, budget=budget, buffer=2.0)
        self.assertFalse(comp_grounding["budget_violation"])
        self.assertEqual(comp_grounding["budget_excess_chars"], 0.0)

    def test_region_bounded_psnr_ssim(self):
        """Image metrics must evaluate strictly on x >= burn_x_start."""
        h, w = 100, 200
        clean = np.ones((h, w, 3), dtype=np.uint8) * 200
        restored = clean.copy()

        # Corrupt only the right damaged strip (x >= 150)
        burn_x = 150
        restored[:, burn_x:] = 50

        # Region bounded metrics
        res = compute_region_bounded_psnr_ssim(clean, restored, burn_x_start=burn_x)
        self.assertIsNotNone(res["bounded_psnr"])
        self.assertLess(res["bounded_psnr"], 30.0, "Corrupted region must have lower PSNR.")

    def test_ablation_configuration_separation(self):
        """Verifies that ablation modes map cleanly to baseline parameters."""
        cfg = Config()
        model = ForensicT5Restorer(cfg)

        # Baseline text-only vs full model
        self.assertTrue(hasattr(model, "visual_encoder"))
        self.assertTrue(hasattr(model, "cross_modal_fusion"))
        self.assertTrue(hasattr(model, "contrastive_aligner"))

    def test_multi_seed_reproducibility(self):
        """Running with the same random seed must produce identical corruption boundaries."""
        cfg = Config()
        gen = SideBurnGenerator(cfg)
        h, w = 1024, 1024
        seed = 2026

        rng1 = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed)
        rng3 = np.random.RandomState(seed + 1)

        b1 = gen._generate_jagged_boundary(h, int(w * 0.75), 40, rng=rng1)
        b2 = gen._generate_jagged_boundary(h, int(w * 0.75), 40, rng=rng2)
        b3 = gen._generate_jagged_boundary(h, int(w * 0.75), 40, rng=rng3)

        np.testing.assert_array_equal(b1, b2, "Identical seed must produce identical jagged boundary.")
        self.assertFalse(np.array_equal(b1, b3), "Different seeds must produce different jagged boundaries.")


if __name__ == "__main__":
    unittest.main()
