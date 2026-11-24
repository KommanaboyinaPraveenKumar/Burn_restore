"""
tests/test_api.py
=================
Unit and endpoint tests for FastAPI document restoration endpoints.
"""

import io
import unittest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import get_restoration_engine, set_restoration_engine
from inference.restoration_engine import RestorationEngine
from config import Config


class TestFastAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Configure lightweight engine on CPU for API tests
        cfg = Config()
        cfg.batch_size = 1
        cls.engine = RestorationEngine(checkpoint_path="checkpoints/best_checkpoint.pt", device="cpu", config=cfg)
        set_restoration_engine(cls.engine)
        cls.client = TestClient(app)

        # Create a small valid test image in memory (PNG bytes)
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        # Add text line and dark right edge
        cv2.line(img, (20, 100), (220, 100), (0, 0, 0), 2)
        cv2.line(img, (20, 180), (220, 180), (0, 0, 0), 2)
        img[:, 240:] = (20, 15, 12)  # Charcoal burn strip

        success, buf = cv2.imencode(".png", img)
        cls.test_png_bytes = buf.tobytes()

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("device", data)
        self.assertTrue(data["model_loaded"])

    def test_model_info_endpoint(self):
        resp = self.client.get("/model/info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("model_architecture", data)
        self.assertIn("total_parameters", data)
        self.assertIn("config", data)
        self.assertGreater(data["total_parameters"], 0)

    def test_interactive_ui_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        self.assertIn("Grounded Document Restoration", resp.text)

    def test_restore_image_valid(self):
        files = {"file": ("test_doc.png", io.BytesIO(self.test_png_bytes), "image/png")}
        resp = self.client.post("/restore/image", files=files, data={"max_lines": 2})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("document_shape", data)
        self.assertIn("burn_detection", data)
        self.assertIn("restorations", data)
        self.assertIn("total_page_inference_ms", data)
        self.assertIn("diagnostic_overlay_base64", data)
        self.assertIsNotNone(data["diagnostic_overlay_base64"])

    def test_restore_image_empty_file_returns_400(self):
        files = {"file": ("empty.png", io.BytesIO(b""), "image/png")}
        resp = self.client.post("/restore/image", files=files)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("empty", resp.json()["detail"].lower())

    def test_restore_image_manual_bbox(self):
        files = {"file": ("test_doc.png", io.BytesIO(self.test_png_bytes), "image/png")}
        data = {"manual_bbox": "200,50,290,250", "max_lines": 2}
        resp = self.client.post("/restore/image", files=files, data=data)
        self.assertEqual(resp.status_code, 200)
        res_data = resp.json()
        self.assertTrue(res_data["burn_detection"]["is_manual"])

    def test_restore_region_endpoint(self):
        files = {"file": ("test_doc.png", io.BytesIO(self.test_png_bytes), "image/png")}
        form_data = {
            "x1": 220,
            "y1": 90,
            "x2": 290,
            "y2": 115,
            "prefix_text": "Deep learning project",
            "max_new_tokens": 8,
        }
        resp = self.client.post("/restore/region", files=files, data=form_data)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("predicted_restoration", data)
        self.assertIn("physical_budget", data)
        self.assertIn("inference_time_ms", data)

    def test_restore_batch_endpoint(self):
        files = [
            ("files", ("doc1.png", io.BytesIO(self.test_png_bytes), "image/png")),
            ("files", ("doc2.png", io.BytesIO(self.test_png_bytes), "image/png")),
        ]
        resp = self.client.post("/restore/batch", files=files, data={"max_lines_per_doc": 1})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_documents_processed"], 2)
        self.assertEqual(data["successful_count"], 2)
        self.assertEqual(data["failed_count"], 0)

    def test_restore_page_alias(self):
        files = {"file": ("test_doc.png", io.BytesIO(self.test_png_bytes), "image/png")}
        resp = self.client.post("/restore/page", files=files, data={"max_lines": 1})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("restorations", data)

    def test_invalid_image_format_returns_400(self):
        files = {"file": ("corrupt.txt", io.BytesIO(b"Not an image at all"), "text/plain")}
        resp = self.client.post("/restore/image", files=files)
        self.assertEqual(resp.status_code, 400)

    def test_invalid_manual_bbox_returns_400(self):
        files = {"file": ("test_doc.png", io.BytesIO(self.test_png_bytes), "image/png")}
        resp = self.client.post("/restore/image", files=files, data={"manual_bbox": "invalid,bbox"})
        self.assertEqual(resp.status_code, 400)

    def test_batch_exceeding_limit_returns_400(self):
        files = [("files", (f"doc{i}.png", io.BytesIO(self.test_png_bytes), "image/png")) for i in range(12)]
        resp = self.client.post("/restore/batch", files=files)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("exceeds maximum limit", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()

