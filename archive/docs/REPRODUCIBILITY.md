# Reproducibility Guide: Grounded Side-Burnt Document Text Restoration

**Project Repository:** `https://github.com/KommanaboyinaPraveenKumar/Burn_restore.git`  
**Model Architecture:** `ForensicT5Restorer` (Flan-T5-small text transformer + ResNet-34 multi-scale visual encoder)  
**Total Parameters:** 85,476,929 (T5: 60,506,624 | ResNet-34: 21,284,672 | Adapters & Cross-Modal: 3,685,633)  
**Protocol Version:** v1.0.0-research-prototype  
**Reference Checkpoint:** `checkpoints/best_checkpoint.pt` (`SHA-256: 2403d3190e8e3c2863d23734c881f1b60e8bc58c762652ad96e9a1a72c06fc12`)

---

## 1. System Requirements & Hardware Specification

The framework supports cross-platform execution on Windows, Linux, and macOS. Both CPU-only and GPU-accelerated (NVIDIA CUDA 11.8 / 12.1+) backends are fully supported.

| Specification | Minimum Requirement | Tested Benchmark Configuration |
|---|---|---|
| **Operating System** | Windows 10/11, Ubuntu 20.04/22.04, macOS 12+ | Windows 11 Enterprise (x86_64) |
| **Python Version** | Python 3.10 – 3.11 | Python 3.11.9 |
| **RAM** | 8 GB RAM | 16 GB DDR5 |
| **Storage** | 3 GB free disk space | SSD with NVMe |
| **Accelerator** | Optional (CPU supported) | Intel Core i7 / NVIDIA RTX (CUDA compatible) |
| **PyTorch Version** | PyTorch >= 2.0.0 | PyTorch 2.10.0+cpu |
| **Transformers** | Hugging Face Transformers >= 4.30.0 | Transformers 4.57.6 |

---

## 2. Environment Setup & Dependency Installation

### Option A: Standard Virtual Environment (`venv` + `pip`)

```bash
# 1. Clone repository
git clone https://github.com/KommanaboyinaPraveenKumar/Burn_restore.git
cd Burn_restore

# 2. Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS (Bash):
source .venv/bin/activate

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Option B: Conda / Mamba Environment

```bash
# 1. Create conda environment from specification file
conda env create -f environment.yml

# 2. Activate conda environment
conda activate burn-restore
```

---

## 3. Repository Artifact & Checkpoint Verification

Before running experiments, execute the automated checksum and integrity verifier to ensure all model checkpoints, configs, figures, and dataset manifests match the publication SHA-256 hashes.

```bash
python scripts/verify_artifacts.py
```

**Expected Console Output:**
```
================================================================================
PHASE 6: REPRODUCIBILITY ARTIFACT VERIFIER
================================================================================
[PASS] All 15 required repository files verified present.
[PASS] Checkpoint checkpoints/best_checkpoint.pt (978.7 MB) SHA-256 hash verified:
       2403d3190e8e3c2863d23734c881f1b60e8bc58c762652ad96e9a1a72c06fc12
[PASS] Dataset splits manifest verified: 6 document pages registered.
================================================================================
RESULT: ALL ARTIFACTS AND CHECKSUMS VERIFIED SUCCESSFULLY (100% PASS)
```

---

## 4. Dataset Splits & Ground-Truth Alignment

The deterministic synthetic document fixture is located at `data/synthetic_fixture/`. The canonical page-level splits are tracked in `docs/dataset_readiness_report.json` with zero page overlap:

- **Training Split (4 pages, 66.7%):** `page_001.png`, `page_002.png`, `page_004.png`, `page_005.png`
- **Validation Split (1 page, 16.7%):** `page_003.png`
- **Test Split (1 page, 16.7%):** `page_006.png`

Each page image is paired with a UTF-8 ground-truth transcript containing exactly one text line per physical document line, accompanied by spatial bounding boxes, pixel width allocations, and character budget calibrations.

---

## 5. Executing the Regression Test Suite

The test suite contains **80 independent unit and integration tests** verifying all mathematical, tensor, dataset, model, training, inference, burn detection, budget estimation, and API components.

```bash
# Run all 80 unit and integration tests
python -m unittest discover -s tests -p "test_*.py" -v
```

**Expected Result:**
```
Ran 80 tests in ~115s
OK
```

### Key Subsystem Test Targets:
1. `tests/test_model.py`: Forward pass, multi-scale visual tensor shapes `(B, 49, 512)`, gated cross-attention, contrastive stroke loss, autoregressive text generation.
2. `tests/test_dataset.py`: Multi-scale visual crop extraction `(3, 224, 224)`, coordinate normalization, char budget calculation, deterministic masking.
3. `tests/test_burn_detector.py`: Edge boundary contour extraction, severity classification, severed line detection.
4. `tests/test_budget_estimator.py`: Proportional pixel budget calculation, word-boundary snapping, character budget enforcement.
5. `tests/test_api.py`: FastAPI endpoints (`/health`, `/model/info`, `/restore/page`, `/restore/batch`).

---

## 6. Reproducing Benchmark & Ablation Tables

### A. Run Full Scientific Audit Engine

To reproduce the exact audit numbers, parameter counts, and zero-leakage split audits:

```bash
python scripts/final_research_audit.py
```
Output artifact written to `docs/FINAL_RESEARCH_AUDIT.json`.

### B. Reproduce Benchmark Baselines (EXP-A through EXP-G)

To run the complete benchmark evaluation across all 7 model configurations:

```bash
python test_evaluate.py --config configs/reproduction.yaml --checkpoint checkpoints/best_checkpoint.pt
```

**Verified Benchmark Performance Table:**

| Model ID | Method / Architecture | Raw CER ↓ | Norm CER ↓ | WER ↓ | BLEU-4 ↑ | ROUGE-L ↑ | Exact Match |
|---|---|---|---|---|---|---|---|
| **EXP-A** | T5-small Text-Only Baseline | 2.1592 | 0.9429 | 5.0633 | 0.0000 | 0.0825 | 0.0000 |
| **EXP-B** | Naive Multi-Modal Concatenation | 2.1592 | 0.9429 | 5.0633 | 0.0000 | 0.0825 | 0.0000 |
| **EXP-C** | Unconstrained Greedy Restorer | 2.1592 | 0.9429 | 5.0633 | 0.0000 | 0.0825 | 0.0000 |
| **EXP-D** | Beam Search Restorer ($k=4$) | 2.1804 | 0.9512 | 5.1200 | 0.0000 | 0.0801 | 0.0000 |
| **EXP-E** | Physical Budget Constrained Decoding | 2.1592 | 0.9429 | 5.0633 | 0.0000 | 0.0825 | 0.0000 |
| **EXP-F** | ForensicT5 Full Multimodal Model | 2.1592 | 0.9429 | 5.0633 | 0.0000 | 0.0825 | 0.0000 |
| **EXP-G** | Full Multimodal + Post-Hoc Snapping | 1.8420 | 0.8910 | 4.3120 | 0.0000 | 0.1140 | 0.0000 |

*Scientific Note on Raw CER:* Raw Character Error Rate exceeds 1.0 because standard Levenshtein edit distance includes unconstrained autoregressive insertions ($I$) that exceed the target ground-truth character length ($N$). Normalized CER ($1 - \text{Accuracy}$) is bounded in $[0.89, 0.95]$.

### C. Reproduce Visual Modality Stress Tests

To assess whether the model relies on visual stroke features or primarily language context:

| Visual Modality Condition | Test CER | Relative Change vs Clean | Empirical Finding |
|---|---|---|---|
| **Clean Burn Crop** | 2.1592 | 0.00% (Baseline) | Standard visual representation |
| **Zeroed Crop (Black)** | 2.1547 | -0.21% | Complete visual ablation does not degrade CER |
| **Spatially Shuffled Crop** | 2.1592 | 0.00% | Scrambled spatial layout does not degrade CER |
| **Inverted Pixel Crop** | 2.1592 | 0.00% | Inverted contrast does not alter prediction |
| **Gaussian Blurred Crop** | 2.1592 | 0.00% | Loss of high-frequency strokes produces identical output |

*Conclusion:* The language prior of T5-small dominates text generation on this training distribution; microscopic character-stroke visual grounding remains an open research frontier.

---

## 7. Running the End-to-End Restoration Pipeline

### A. Command-Line Interface (CLI)

Restore a single side-burnt document page from the terminal:

```bash
# Basic single-page restoration
python scripts/restore_document.py \
    --input-image data/synthetic_fixture/page_001.png \
    --transcript data/synthetic_fixture/page_001.txt \
    --checkpoint checkpoints/best_checkpoint.pt \
    --output-dir outputs/cli_restoration \
    --device cpu

# High-resolution rendering with burn mask export
python scripts/restore_document.py \
    --input-image data/synthetic_fixture/page_006.png \
    --checkpoint checkpoints/best_checkpoint.pt \
    --output-dir outputs/cli_page6 \
    --save-visualizations \
    --beam-size 4
```

### B. Running the FastAPI REST Server & Interactive Workbench

```bash
# Start FastAPI application with live interactive UI
python scripts/run_api.py --host 127.0.0.1 --port 8000
```

Once running:
- **Interactive Web UI Workbench:** Navigate your browser to `http://127.0.0.1:8000/`
- **Swagger Interactive API Documentation:** Navigate to `http://127.0.0.1:8000/docs`
- **OpenAPI JSON Specification:** Navigate to `http://127.0.0.1:8000/openapi.json`

#### Example REST API Query (`curl`):

```bash
curl -X POST "http://127.0.0.1:8000/restore/page" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/synthetic_fixture/page_001.png" \
  -F "max_lines=15" \
  -F "beam_size=4"
```

---

## 8. Retraining From Scratch (Two-Stage Protocol)

If you wish to re-train the model weights from initialization:

### Stage 1: Adapter & Cross-Attention Warmup
Freezes T5-small backbone and ResNet-34 trunk; trains only cross-modal adapters, projection heads, and gated cross-attention:

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
Unfreezes top ResNet stages and T5 text layers with differential learning rates:

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

---

## 9. Academic Paper Compilation (LaTeX)

The complete academic manuscript source is in `paper/main.tex` with figures in `paper/figures/` and citations in `paper/references.bib`.

```bash
cd paper

# Using latexmk:
latexmk -pdf -bibtex main.tex

# Or manual compilation cycle:
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

For users without local TeX installations, the `paper/` folder can be zipped and uploaded directly to [Overleaf](https://www.overleaf.com) for 1-click cloud rendering.

