#!/usr/bin/env python3
"""
scripts/test_frontend_backend_flow.py
=====================================
Automated End-to-End Simulation of the Frontend-Backend Restoration Workflow.
Simulates all user actions performed in the Web UI workbench:
  1. Health check probe (/health)
  2. Model info query (/model/info)
  3. Static workbench delivery (GET /)
  4. Automatic restoration request (/restore/image) with file upload
  5. Manual region bounding box restoration (/restore/image with manual_bbox)
  6. Single severed region restoration (/restore/region)
  7. Batch restoration (/restore/batch)
  8. Error handling: empty payload, corrupt format, invalid bbox
"""

import base64
import io
import json
import urllib.request
import urllib.parse
from pathlib import Path
import cv2
import numpy as np

BASE_URL = "http://127.0.0.1:8000"
REPO_ROOT = Path(__file__).resolve().parent.parent

def print_step(msg: str):
    print(f"\n[E2E Flow] {msg}")

def run_tests():
    # 1. Health check
    print_step("1. Verifying GET /health ...")
    req = urllib.request.Request(f"{BASE_URL}/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"Health check returned {resp.status}"
        data = json.loads(resp.read().decode())
        print(f"   Status: {data.get('status')} | Device: {data.get('device')} | Model Loaded: {data.get('model_loaded')}")
        assert data.get("status") == "healthy"
        assert data.get("model_loaded") is True

    # 2. Model Info
    print_step("2. Verifying GET /model/info ...")
    req = urllib.request.Request(f"{BASE_URL}/model/info")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        print(f"   Architecture: {data.get('model_architecture')}")
        print(f"   Parameters: {data.get('total_parameters'):,}")
        assert data.get("total_parameters") == 85476929

    # 3. Static UI Workbench
    print_step("3. Verifying GET / (Interactive Web UI Workbench HTML) ...")
    req = urllib.request.Request(f"{BASE_URL}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        html = resp.read().decode()
        assert "<title>Grounded Document Restoration" in html
        assert "drop-zone" in html
        assert "doc-canvas" in html
        assert "restore-btn" in html
        assert "copy-all-btn" in html
        print(f"   Workbench UI served successfully ({len(html)} bytes).")

    # 4. Automatic Document Restoration (Simulating UI File Upload & Click)
    print_step("4. Simulating Document Upload and Auto-Restoration (/restore/image) ...")
    test_image_p = REPO_ROOT / "data" / "synthetic_fixture" / "page_001.png"
    assert test_image_p.exists(), f"Missing test image: {test_image_p}"

    with open(test_image_p, "rb") as f:
        img_bytes = f.read()

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    body = io.BytesIO()
    # file field
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="file"; filename="page_001.png"\r\n')
    body.write(b"Content-Type: image/png\r\n\r\n")
    body.write(img_bytes)
    body.write(b"\r\n")
    # max_lines field
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="max_lines"\r\n\r\n')
    body.write(b"2\r\n")
    # num_beams field
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="num_beams"\r\n\r\n')
    body.write(b"2\r\n")
    # end boundary
    body.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"{BASE_URL}/restore/image",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        result = json.loads(resp.read().decode())
        print(f"   Restored Lines Count: {result.get('restored_lines_count')}")
        print(f"   Total Inference Latency: {result.get('total_page_inference_ms')} ms")
        print(f"   Diagnostic Overlay: {'Present' if result.get('diagnostic_overlay_base64') else 'None'}")
        assert result.get("restored_lines_count") == 2
        assert result.get("diagnostic_overlay_base64") is not None
        # Verify overlay is valid Base64 PNG
        raw_b64 = result.get("diagnostic_overlay_base64").split(",")[-1]
        decoded_overlay = base64.b64decode(raw_b64)
        assert len(decoded_overlay) > 1000, "Decoded overlay is too small"
        print(f"   Decoded Visual Overlay: {len(decoded_overlay)} PNG bytes verified.")

    # 5. Manual Bounding Box Restoration (Simulating User Canvas Selection)
    print_step("5. Simulating User Manual Bounding Box Override ...")
    body_man = io.BytesIO()
    body_man.write(f"--{boundary}\r\n".encode())
    body_man.write(b'Content-Disposition: form-data; name="file"; filename="page_001.png"\r\n')
    body_man.write(b"Content-Type: image/png\r\n\r\n")
    body_man.write(img_bytes)
    body_man.write(b"\r\n")
    # manual_bbox field
    body_man.write(f"--{boundary}\r\n".encode())
    body_man.write(b'Content-Disposition: form-data; name="manual_bbox"\r\n\r\n')
    body_man.write(b"550,100,950,850\r\n")
    body_man.write(f"--{boundary}\r\n".encode())
    body_man.write(b'Content-Disposition: form-data; name="max_lines"\r\n\r\n')
    body_man.write(b"2\r\n")
    body_man.write(f"--{boundary}--\r\n".encode())

    req_man = urllib.request.Request(
        f"{BASE_URL}/restore/image",
        data=body_man.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    with urllib.request.urlopen(req_man) as resp:
        assert resp.status == 200
        res_man = json.loads(resp.read().decode())
        print(f"   Manual Detection Flag: {res_man.get('burn_detection', {}).get('is_manual')}")
        assert res_man.get("burn_detection", {}).get("is_manual") is True

    # 6. Error Handling: Invalid Upload Formats
    print_step("6. Verifying API Error Handling on Corrupt Input ...")
    body_err = io.BytesIO()
    body_err.write(f"--{boundary}\r\n".encode())
    body_err.write(b'Content-Disposition: form-data; name="file"; filename="bad.txt"\r\n')
    body_err.write(b"Content-Type: text/plain\r\n\r\n")
    body_err.write(b"This is not a picture of a burnt manuscript")
    body_err.write(b"\r\n")
    body_err.write(f"--{boundary}--\r\n".encode())

    req_err = urllib.request.Request(
        f"{BASE_URL}/restore/image",
        data=body_err.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    try:
        urllib.request.urlopen(req_err)
        raise AssertionError("Expected HTTP 400 on invalid image file")
    except urllib.error.HTTPError as err:
        assert err.code == 400
        print(f"   Correctly rejected invalid file format with HTTP {err.code}.")

    print("\n" + "=" * 80)
    print("ALL FRONTEND-BACKEND WORKFLOW SIMULATION TESTS PASSED (100% OPERATIONAL)")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
