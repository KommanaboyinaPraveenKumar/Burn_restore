"""
tests/test_burn_detector.py
===========================
Unit tests for OpenCV-based page-level burn boundary detection and text-line segmentation.
"""

import unittest
import numpy as np
import cv2

from inference.burn_detector import BurnDetector, BurnDetectionResult, SeveredLineRegion


class TestBurnDetector(unittest.TestCase):

    def setUp(self):
        self.detector = BurnDetector(
            charcoal_threshold=70,
            edge_search_ratio=0.50,
            min_ink_row_density=0.015,
            line_gap_merge_px=4,
        )
        # Create a synthetic 400x300 document page image
        # White background (255) with 5 horizontal text lines (black pixels, 0)
        self.h = 400
        self.w = 300
        self.image = np.full((self.h, self.w, 3), 255, dtype=np.uint8)

        # Draw 5 horizontal text lines across y=50, 100, 150, 200, 250
        for y in [50, 100, 150, 200, 250]:
            cv2.line(self.image, (30, y), (270, y), (0, 0, 0), 3)

        # Simulate a dark charred side burn on the right edge (x >= 220)
        for y in range(self.h):
            cut_x = 220 + int(10 * np.sin(y / 20.0))
            self.image[y, cut_x:] = (15, 12, 10)  # Dark charcoal color

    def test_auto_burn_detection_returns_valid_result(self):
        result = self.detector.detect(self.image)
        self.assertIsInstance(result, BurnDetectionResult)
        self.assertEqual(result.image_shape, (self.h, self.w, 3))
        self.assertFalse(result.is_manual)
        self.assertGreater(result.detection_confidence, 0.0)
        self.assertLessEqual(result.detection_confidence, 1.0)

    def test_damage_mask_dimensions_and_values(self):
        result = self.detector.detect(self.image)
        mask = result.damage_mask
        self.assertEqual(mask.shape, (self.h, self.w))
        self.assertEqual(mask.dtype, np.uint8)
        # Verify right edge pixels are labeled damaged (255)
        self.assertTrue(np.all(mask[:, -5:] == 255))
        # Verify left edge intact pixels are 0
        self.assertTrue(np.all(mask[:, :50] == 0))

    def test_severed_lines_detection(self):
        result = self.detector.detect(self.image)
        self.assertGreater(len(result.severed_lines), 0)
        for s in result.severed_lines:
            self.assertIsInstance(s, SeveredLineRegion)
            self.assertGreater(s.y1, s.y0)
            self.assertGreater(s.damaged_width_px, 0)
            self.assertGreater(s.intact_width_px, 0)
            # Crop bbox should be [x0, y0, x1, y1] within image dimensions
            x0, y0, x1, y1 = s.crop_bbox
            self.assertGreaterEqual(x0, 0)
            self.assertLessEqual(x1, self.w)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(y1, self.h)

    def test_manual_bounding_box_override(self):
        manual_box = (200, 40, 290, 360)
        result = self.detector.detect(self.image, manual_bbox=manual_box)
        self.assertTrue(result.is_manual)
        self.assertEqual(result.burn_bbox, manual_box)
        self.assertEqual(result.detection_confidence, 1.0)
        self.assertEqual(result.confidence_method, "manual_user_specification")
        # Mask inside the manual box should be 255
        self.assertEqual(result.damage_mask[100, 250], 255)
        # Mask outside the manual box should be 0
        self.assertEqual(result.damage_mask[10, 50], 0)

    def test_empty_or_invalid_image_raises_error(self):
        with self.assertRaises(ValueError):
            self.detector.detect(None)

        with self.assertRaises(ValueError):
            self.detector.detect(np.array([]))

    def test_diagnostic_visualization_generation(self):
        result = self.detector.detect(self.image)
        vis = self.detector.generate_diagnostic_visualization(self.image, result)
        self.assertEqual(vis.shape, (self.h, self.w, 3))
        self.assertEqual(vis.dtype, np.uint8)
        # Should not be identical to original image due to annotations
        self.assertFalse(np.array_equal(vis, self.image))

    def test_result_to_dict_serialization(self):
        result = self.detector.detect(self.image)
        d = result.to_dict()
        self.assertIn("image_height", d)
        self.assertIn("burn_bbox", d)
        self.assertIn("detection_confidence", d)
        self.assertIn("severed_lines", d)
        self.assertEqual(d["image_height"], self.h)
        self.assertEqual(d["image_width"], self.w)


if __name__ == "__main__":
    unittest.main()

