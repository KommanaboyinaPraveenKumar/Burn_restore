# Final Reproducibility Audit: Grounded Side-Burnt Document Text Restoration

**Project:** Grounded Side-Burnt Document Text Restoration  
**Date:** November 2026  
**Document Status:** AUTHORITATIVE REPRODUCIBILITY AUDIT  
**Protocol Version:** v1.0.0-final  

---

## 1. System Environment & Hardware Configuration

The framework has been audited and validated on the following experimental platform:

| Parameter | Tested Environment | Supported Alternatives |
|---|---|---|
| **OS** | Windows 11 Enterprise (x86_64) | Ubuntu 20.04/22.04 LTS, macOS 12+ |
| **Python** | Python 3.11.9 (CPython) | Python 3.10 – 3.11 |
| **PyTorch** | PyTorch 2.10.0+cpu | PyTorch >= 2.0.0 (CUDA 11.8/12.1+ supported) |
| **Transformers** | Hugging Face Transformers 4.57.6 | Transformers >= 4.30.0 |
| **OpenCV** | OpenCV-Python 4.11.0.86 | OpenCV >= 4.7.0 |
| **FastAPI** | FastAPI 0.111.0 | FastAPI >= 0.100.0 |
| **Uvicorn** | Uvicorn 0.30.1 | Uvicorn >= 0.22.0 |

---

## 2. Checkpoint Provenance & Cryptographic Verification

All published baseline evaluations, inferences, and API demonstrations utilize the primary trained checkpoint:

| Property | Value |
|---|---|
| **File Path** | `checkpoints/best_checkpoint.pt` |
| **File Size** | 1,026,233,181 bytes ($978.68\text{ MB}$) |
| **SHA-256 Digest** | `2403d3190e8e3c2863d23734c881f1b60e8bc58c762652ad96e9a1a72c06fc12` |
| **Training Epoch** | Epoch 3 (Global Step 6) |
| **State Keys Count** | 627 weight and bias tensors |
| **Architecture** | `ForensicT5Restorer` (Flan-T5-small + ResNet-34) |
| **Total Parameters** | **85,476,929** |

*Integrity Check:*
```bash
python scripts/verify_artifacts.py
```

---

## 3. Dataset Splits & Deterministic Seeding Audit

Partitioning across the 6 synthetic document pages is strictly enforced at the page level in `data/splits.json`:
- **Training (4 pages):** `sample_page_002`, `sample_page_003`, `sample_page_004`, `sample_page_005`
- **Validation (1 page):** `sample_page_001`
- **Test (1 page):** `sample_page_006`

*Seeding Audit:*
In `Burn_restore/dataset.py:197`, Python's randomly-salted `hash()` function was replaced with `hashlib.sha256(stem.encode("utf-8"))`. As a result, evaluation samples generate 100% bit-for-bit identical scorch contours, severed lines, and physical budgets across all Python sessions and processes.

---

## 4. Authoritative Benchmark Reproduction Commands

### A. Run Canonical Evaluation Suite
Executes all 8 controlled baselines and 6 visual modality stress tests on the held-out test split:
```bash
python scripts/run_canonical_evaluation.py
```
Outputs generated:
- `docs/FINAL_METRICS.json`
- `docs/FINAL_METRICS_REPORT.md`
- `docs/FINAL_EXPERIMENT_COMPARISON.csv`

### B. Run Complete Automated Regression Test Suite
Executes all unit and integration test modules:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### C. Run Command-Line Interface (CLI)
Restores missing line endings on a damaged document page and exports JSON metadata:
```bash
python scripts/restore_document.py \
    --input data/synthetic_fixture/page_001.png \
    --checkpoint checkpoints/best_checkpoint.pt \
    --output outputs/cli_audit_test \
    --device cpu \
    --max-lines 5
```

### D. Run REST API & Interactive Web Workbench
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
- Interactive Web Workbench: `http://127.0.0.1:8000/`
- Swagger Interactive Documentation: `http://127.0.0.1:8000/docs`

---

## 5. Staged Model Retraining Protocol

If retraining from scratch:

### Stage 1: Adapter & Gated Cross-Attention Warmup
```bash
python train.py \
    --stage 1 \
    --images data/synthetic_fixture \
    --texts data/synthetic_fixture \
    --epochs 5 \
    --batch_size 2 \
    --lr 1e-3 \
    --output_dir checkpoints/stage1
```

### Stage 2: End-to-End Multimodal Fine-Tuning
```bash
python train.py \
    --stage 2 \
    --images data/synthetic_fixture \
    --texts data/synthetic_fixture \
    --init_checkpoint checkpoints/stage1/best_checkpoint.pt \
    --epochs 10 \
    --batch_size 2 \
    --lr 5e-5 \
    --output_dir checkpoints/stage2
```

