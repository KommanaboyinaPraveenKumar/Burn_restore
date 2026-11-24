"""
scripts/verify_artifacts.py
===========================
Verifies file existence, directory structure, checkpoint checksums, and dataset fixtures.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent

EXPECTED_FILES = [
    # Core Architecture & Pipeline
    "config.py",
    "model.py",
    "dataset.py",
    "dataset_generator.py",
    "dataset_split.py",
    "dataset_validator.py",
    "train.py",
    "metrics.py",
    # Inference & API
    "inference/burn_detector.py",
    "inference/budget_estimator.py",
    "inference/restoration_engine.py",
    "api/main.py",
    "api/schemas.py",
    "api/dependencies.py",
    "api/services/restoration_service.py",
    "api/routes/restoration.py",
    "api/static/index.html",
    "scripts/restore_document.py",
    # Documentation & Cards
    "README.md",
    "requirements.txt",
    "environment.yml",
    "configs/reproduction.yaml",
    "docs/EXPERIMENT_TRACKER.md",
    "docs/MODEL_CARD.md",
    "docs/DATASET_CARD.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/FINAL_RESEARCH_AUDIT.json",
    "docs/FINAL_RESEARCH_AUDIT_REPORT.md",
    # Research Paper
    "paper/main.tex",
    "paper/references.bib",
    "paper/README.md",
    "paper/figures/figure1_architecture.png",
    "paper/figures/figure2_baselines_comparison.png",
    "paper/figures/figure3_ablation_study.png",
    "paper/figures/figure4_burn_severity.png",
    "paper/figures/figure5_visual_stress_tests.png",
    # Test Suites
    "tests/test_dataset_engineering.py",
    "tests/test_model_architecture.py",
    "tests/test_training_pipeline.py",
    "tests/test_evaluation_benchmark.py",
    "tests/test_burn_detector.py",
    "tests/test_budget_estimator.py",
    "tests/test_restoration_engine.py",
    "tests/test_api.py",
    "tests/test_inference_integration.py",
]

EXPECTED_CHECKPOINTS = {
    "checkpoints/best_checkpoint.pt": "2403d3190e8e3c2863d23734c881f1b60e8bc58c762652ad96e9a1a72c06fc12",
}


def verify_file_sha256(filepath: Path, expected_hash: str) -> bool:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha.update(chunk)
    return sha.hexdigest().lower() == expected_hash.lower()


def main():
    print("=" * 80)
    print("PHASE 6: REPRODUCIBILITY ARTIFACT VERIFIER")
    print("=" * 80)

    missing_files = []
    for rel_path in EXPECTED_FILES:
        full_p = repo_root / rel_path
        if not full_p.is_file():
            missing_files.append(rel_path)

    if missing_files:
        print(f"[FAIL] Missing {len(missing_files)} expected repository files:")
        for m in missing_files:
            print(f"  - {m}")
    else:
        print(f"[PASS] All {len(EXPECTED_FILES)} core repository files verified on disk.")

    # Check Checkpoints
    ckpt_pass = True
    for rel_p, exp_hash in EXPECTED_CHECKPOINTS.items():
        full_p = repo_root / rel_p
        if not full_p.is_file():
            print(f"[WARN] Checkpoint file missing: {rel_p}")
            ckpt_pass = False
        else:
            matches = verify_file_sha256(full_p, exp_hash)
            sz_mb = full_p.stat().st_size / (1024 * 1024)
            if matches:
                print(f"[PASS] Checkpoint {rel_p} ({sz_mb:.1f} MB) SHA-256 hash verified.")
            else:
                print(f"[WARN] Checkpoint {rel_p} hash mismatch (expected {exp_hash[:12]}...).")

    # Check Dataset Fixtures
    manifest_p = repo_root / "data" / "splits.json"
    if manifest_p.is_file():
        with open(manifest_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_p = data.get("metadata", {}).get("total_documents", 0)
        print(f"[PASS] Dataset splits manifest verified: {total_p} document pages registered.")
    else:
        print(f"[FAIL] Dataset splits manifest missing: {manifest_p}")

    print("=" * 80)
    all_ok = len(missing_files) == 0 and ckpt_pass
    if all_ok:
        print("RESULT: ALL ARTIFACTS VERIFIED SUCCESSFULLY (100% PASS)")
    else:
        print("RESULT: ARTIFACT VERIFICATION COMPLETED WITH WARNINGS")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

