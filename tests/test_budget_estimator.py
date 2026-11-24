"""
tests/test_budget_estimator.py
==============================
Unit tests for physical character budget estimation, compliance auditing, and word snapping.
"""

import unittest
from inference.budget_estimator import BudgetEstimator, BudgetComplianceReport


class TestBudgetEstimator(unittest.TestCase):

    def setUp(self):
        self.estimator = BudgetEstimator(
            default_char_width_px=14.0,
            margin_buffer=2.0,
        )

    def test_default_geometric_budget_calculation(self):
        # 140 px damaged width with default 14 px char width = 10.0 chars
        budget = self.estimator.estimate_budget(damaged_width_px=140)
        self.assertAlmostEqual(budget, 10.0, places=2)

    def test_zero_or_negative_damaged_width(self):
        self.assertEqual(self.estimator.estimate_budget(damaged_width_px=0), 0.0)
        self.assertEqual(self.estimator.estimate_budget(damaged_width_px=-50), 0.0)

    def test_dynamic_font_scale_from_surviving_text(self):
        # 20 characters in 200px width = 10 px per character
        # 150 px damaged width / 10 px = 15.0 characters
        prefix = "This is twenty chars"
        budget = self.estimator.estimate_budget(
            damaged_width_px=150,
            intact_width_px=200,
            visible_text=prefix,
        )
        self.assertAlmostEqual(budget, 15.0, places=1)

    def test_line_height_font_scale_fallback(self):
        # Line height 20 px * 0.58 = 11.6 px char width
        # 116 px damaged width / 11.6 = 10.0 chars
        budget = self.estimator.estimate_budget(
            damaged_width_px=116,
            line_height_px=20,
        )
        self.assertAlmostEqual(budget, 10.0, places=1)

    def test_clean_infill_text_strips_sentinels_and_special_tokens(self):
        raw = "<pad> <extra_id_0> restore this text </s> | prefix"
        cleaned = self.estimator.clean_infill_text(raw)
        self.assertEqual(cleaned, "restore this text")

    def test_compliance_within_budget(self):
        budget = 10.0
        text = "Hello"  # 5 characters <= 10.0 + 2.0 buffer
        rep = self.estimator.evaluate_compliance(text, budget)
        self.assertIsInstance(rep, BudgetComplianceReport)
        self.assertFalse(rep.budget_violation)
        self.assertEqual(rep.excess_characters, 0)
        self.assertEqual(rep.generated_character_count, 5)
        self.assertFalse(rep.was_snapped)

    def test_compliance_exceeding_budget_and_word_snapping(self):
        budget = 10.0  # max allowed with buffer = 12.0 chars
        text = "Deep learning is wonderful"  # 26 chars > 12
        rep = self.estimator.evaluate_compliance(text, budget, enable_word_snapping=True)
        self.assertTrue(rep.budget_violation)
        self.assertGreater(rep.excess_characters, 0)
        self.assertTrue(rep.was_snapped)
        # Word snapped text should be <= max_allowed (12 chars) and snap at space
        self.assertLessEqual(len(rep.word_snapped_text), 12)
        self.assertEqual(rep.word_snapped_text, "Deep")

    def test_report_serialization_to_dict(self):
        rep = self.estimator.evaluate_compliance("sample prediction", 15.0)
        d = rep.to_dict()
        self.assertIn("estimated_budget", d)
        self.assertIn("budget_violation", d)
        self.assertIn("excess_characters", d)
        self.assertIn("cleaned_text", d)


if __name__ == "__main__":
    unittest.main()

