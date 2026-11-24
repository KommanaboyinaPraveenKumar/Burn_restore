#!/usr/bin/env python3
"""
Master Validation Suite: Grounded Side-Burnt Document Text Restoration
Phase 6: Comprehensive Repository Integrity, Regression, and Smoke Testing

Executes:
  1. Artifact & Checkpoint Verification (SHA-256 checks)
  2. Full 80-Test Regression Suite (unittest discover)
  3. Restoration Engine Checkpoint Smoke Test
  4. Command-Line Interface (CLI) Smoke Test
  5. FastAPI REST API Endpoint Verification
"""

import sys
import os
import time
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def step_1_verify_artifacts() -> bool:
    print_banner("STEP 1: REPOSITORY ARTIFACTS & CHECKPOINT INTEGRITY")
    script = REPO_ROOT / "scripts" / "verify_artifacts.py"
    res = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("[FAIL] Step 1 failed artifact verification.")
        return False
    print("[PASS] Step 1: All artifacts, checksums, and manifests verified.")
    return True

def step_2_run_test_suite() -> bool:
    print_banner("STEP 2: FULL REGRESSION TEST SUITE (84 TESTS)")
    start_time = time.time()
    tests_dir = REPO_ROOT / "tests"
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(tests_dir), pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    duration = time.time() - start_time
    
    print(f"\nRan {result.testsRun} tests in {duration:.2f}s")
    if not result.wasSuccessful():
        print(f"[FAIL] Step 2 failed: {len(result.failures)} failures, {len(result.errors)} errors.")
        return False
    print(f"[PASS] Step 2: 100% test pass rate ({result.testsRun}/{result.testsRun} passed).")
    return True

def step_3_restoration_engine_smoke_test() -> bool:
    print_banner("STEP 3: RESTORATION ENGINE CHECKPOINT SMOKE TEST")
    try:
        from inference.restoration_engine import RestorationEngine
        checkpoint_path = REPO_ROOT / "checkpoints" / "best_checkpoint.pt"
        if not checkpoint_path.exists():
            print(f"[FAIL] Checkpoint not found at {checkpoint_path}")
            return False
            
        print("Initializing RestorationEngine (device='cpu')...")
        engine = RestorationEngine(checkpoint_path=str(checkpoint_path), device="cpu")
        model_info = engine.get_model_status()
        print(f"Loaded architecture: {model_info.get('model_architecture')}")
        print(f"Total parameters: {model_info.get('total_parameters'):,}")
        
        test_image = REPO_ROOT / "data" / "synthetic_fixture" / "page_001.png"
        if not test_image.exists():
            print(f"[FAIL] Test image not found at {test_image}")
            return False
            
        print(f"Restoring test page: {test_image.name}...")
        result = engine.restore_page(str(test_image), max_lines=2, num_beams=2)
        burn_info = result.get("burn_detection", {})
        print(f"Burn detected: {burn_info.get('severed_lines_count', 0)} severed lines (confidence: {burn_info.get('detection_confidence', 0.0)})")
        print(f"Restored {result.get('restored_lines_count', 0)} lines successfully.")
        print("[PASS] Step 3: Restoration engine successfully executed single-page inference.")
        return True
    except Exception as exc:
        print(f"[FAIL] Step 3 encountered an exception: {exc}")
        import traceback
        traceback.print_exc()
        return False

def step_4_cli_smoke_test() -> bool:
    print_banner("STEP 4: COMMAND-LINE INTERFACE (CLI) SMOKE TEST")
    cli_script = REPO_ROOT / "scripts" / "restore_document.py"
    input_img = REPO_ROOT / "data" / "synthetic_fixture" / "page_001.png"
    out_dir = REPO_ROOT / "outputs" / "validation_cli_test"
    
    cmd = [
        sys.executable,
        str(cli_script),
        "--input", str(input_img),
        "--checkpoint", str(REPO_ROOT / "checkpoints" / "best_checkpoint.pt"),
        "--output", str(out_dir),
        "--max-lines", "2",
        "--device", "cpu"
    ]
    
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print("[FAIL] Step 4 CLI script failed:")
        print(res.stderr)
        return False
    print("[PASS] Step 4: CLI executed successfully and generated output JSON and transcript.")
    return True

def step_5_api_smoke_test() -> bool:
    print_banner("STEP 5: FASTAPI REST API SMOKE TEST")
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        # Test /health
        res_health = client.get("/health")
        assert res_health.status_code == 200, f"Health check returned {res_health.status_code}"
        health_data = res_health.json()
        print(f"Health check status: {health_data.get('status')}")
        
        # Test /model/info
        res_info = client.get("/model/info")
        assert res_info.status_code == 200, f"Model info returned {res_info.status_code}"
        info_data = res_info.json()
        print(f"Model info total parameters: {info_data.get('total_parameters'):,}")
        
        print("[PASS] Step 5: FastAPI server endpoints verified via TestClient.")
        return True
    except Exception as exc:
        print(f"[FAIL] Step 5 API test encountered an exception: {exc}")
        import traceback
        traceback.print_exc()
        return False

def main():
    start_all = time.time()
    print_banner("GROUNDED SIDE-BURNT DOCUMENT RESTORATION: FULL VALIDATION")
    print(f"Working Directory: {REPO_ROOT}")
    print(f"Python Executable: {sys.executable}")
    
    results = [
        ("Artifact Integrity", step_1_verify_artifacts()),
        ("Regression Test Suite", step_2_run_test_suite()),
        ("Restoration Engine Inference", step_3_restoration_engine_smoke_test()),
        ("CLI Smoke Test", step_4_cli_smoke_test()),
        ("FastAPI Server Smoke Test", step_5_api_smoke_test()),
    ]
    
    total_time = time.time() - start_all
    print_banner("FINAL VALIDATION SUMMARY")
    all_passed = True
    for name, passed in results:
        status_str = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        print(f"  {name:<35} {status_str}")
        if not passed:
            all_passed = False
            
    print("-" * 80)
    print(f"Total Validation Time: {total_time:.2f}s")
    if all_passed:
        print("[SUCCESS] ALL VALIDATION CHECKS PASSED (100% OPERATIONAL)")
        sys.exit(0)
    else:
        print("[FAILURE] ONE OR MORE VALIDATION CHECKS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()

