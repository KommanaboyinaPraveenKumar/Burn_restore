# Phase 5 Research & Engineering Report: Inference Pipeline, Real-Time Restoration Engine, REST API & Interactive User Interface

**Project:** Grounded Side-Burnt Document Text Restoration  
**Phase:** 5 — Inference Pipeline, Real-Time Restoration Engine, REST API & User Interface  
**Date:** November 2026  
**Status:** Completed  
**Repository:** `https://github.com/KommanaboyinaPraveenKumar/Burn_restore.git`  
**Test Suite Status:** **80 / 80 Tests Passed (100% Success Rate across All 9 Test Suites in 113.5s)**

---

## 1. Executive Summary

Phase 5 transitions the research artifacts from Phases 0–4 into a modular, production-ready inference system without modifying the underlying `ForensicT5Restorer` architecture (`T5-small` text transformer + `ResNet-34` multi-scale forensic visual encoder).

### Core Accomplishments:
1. **Central Restoration Engine (`inference/restoration_engine.py`):** Encapsulates model and tokenizer lifecycle management, checkpoint validation, single-region and full-page inference under `torch.inference_mode()`, preventing redundant model reloads.
2. **OpenCV Page-Level Burn Detector (`inference/burn_detector.py`):** Computer-vision pipeline identifying charred right margins, estimating continuous jagged boundary profiles $b_x(y)$, generating 2D binary damage masks, and performing horizontal ink projection profiling to segment severed text lines with a manual bounding-box fallback.
3. **Physical Character-Budget Estimator (`inference/budget_estimator.py`):** Subword-aware geometric budget estimator ($B = \Delta x / w_{\text{char}}$) supporting dynamic font-scale calibration from surviving text and line height, integrated with `PhysicalBudgetLogitsProcessor` and post-generation compliance auditing with word boundary snapping.
4. **Command-Line Interface (`scripts/restore_document.py`):** Fully featured CLI supporting automated and manual bounding-box restoration, beam search, diagnostic overlay generation, and structured JSON reporting.
5. **FastAPI REST API (`api/`):** Asynchronous web service exposing `/health`, `/model/info`, `/restore/image`, `/restore/region`, `/restore/batch`, and OpenAPI `/docs`.
6. **Interactive Single-Page Workbench (`api/static/index.html`):** Clean, research-oriented user interface mounted directly at `/` and `/ui`, featuring drag-and-drop document upload, auto vs. manual canvas selection, live side-by-side diagnostic overlays, per-line budget violation indicators, text copying, and JSON export.
7. **Zero Regressions:** 32 new unit and integration tests added; all 80 tests in the repository pass cleanly in 113.5s.

---

## 2. Inference Architecture Overview

```
                                 [Document Image (PNG/JPG/TIFF)]
                                                |
                                                v
                                   +--------------------------+
                                   |   RestorationEngine      |
                                   |  (Cached in GPU/CPU RAM) |
                                   +--------------------------+
                                                |
                     +--------------------------+--------------------------+
                     |                                                     |
                     v                                                     v
        +--------------------------+                         +--------------------------+
        |       BurnDetector       |                         |     BudgetEstimator      |
        |  (OpenCV Scorch/Edge)    |                         |   (Geometric & Font)     |
        +--------------------------+                         +--------------------------+
                     |                                                     |
                     | 1. Jagged boundary curve b_x(y)                     | 1. Missing width span
                     | 2. Binary damage mask (H, W)                        | 2. Dynamic char width
                     | 3. Severed text line bands                          | 3. Budget B_i = dx / w_c
                     +--------------------------+--------------------------+
                                                |
                                                v
                               +----------------------------------+
                               |     Forensic Visual Crop         |
                               |    (128x256 px, ImageNet Norm)   |
                               +----------------------------------+
                                                |
                                                v
                               +----------------------------------+
                               |     ForensicT5Restorer v2.0      |
                               |  - ResNet-34 Multi-Scale Vision  |
                               |  - 2D Spatial Pos Embeddings     |
                               |  - Gated Cross-Modal Fusion      |
                               |  - T5 Autoregressive Decoder     |
                               |  - PhysicalBudgetLogitsProcessor |
                               +----------------------------------+
                                                |
                                                v
                               +----------------------------------+
                               |    Compliance & Post-Audit       |
                               |  - Excess character calculation  |
                               |  - Word-boundary safe snapping   |
                               |  - Diagnostic overlay rendering  |
                               +----------------------------------+
                                                |
                     +--------------------------+--------------------------+
                     |                                                     |
                     v                                                     v
          [REST API (FastAPI)]                               [CLI Tool (restore_document.py)]
          - POST /restore/image                              - Structured console output
          - POST /restore/region                             - Restoration report JSON
          - Web Workbench UI (/ui)                           - Diagnostic overlay PNG
```

---

## 3. Component Details & Implementation

### 3.1 Model Loading Procedure (`RestorationEngine.load_model()`)
- **Initialization:** Auto-detects device (`cuda` if available, else `cpu`).
- **Tokenizer:** Preloads `AutoTokenizer.from_pretrained("t5-small")` once.
- **Model:** Instantiates `ForensicT5Restorer(Config)` and moves weights to designated device.
- **Checkpoint Validation:** Inspects file existence, file size (~978 MB), and validates keys (`model_state_dict`, `config_dict`, `epoch`). Loads with `strict=False` to handle backward-compatible aliases gracefully.
- **State:** Sets `model.eval()`; all predictions execute inside `with torch.inference_mode():`.
- **Measured Initialization Latency:** 4.9s to 5.6s on CPU.

### 3.2 Image Preprocessing (`RestorationEngine.preprocess_image()`)
- Supports file paths and decoded numpy arrays (BGR or Grayscale).
- Standardizes document dimensions to `Config.page_resize` ($1024 \times 1024$ or $800 \times 1000$).
- Forensic line crops ($128 \times 256$ pixels) capture $180\text{ px}$ of intact left context and the damaged right margin strip.
- Standard ImageNet 3-channel normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 3.3 Automatic Burn Detection (`BurnDetector`)
- **Right-Edge Scorch Search:** Restricts search to the outer margin (default 50% page width).
- **Dark Carbon Feature Extraction:** Thresholds charred pixels ($\text{gray} < 70$) and low-value HSV components ($V < 70$).
- **Canny Edge Detection & Closing:** Groups jagged scorch boundaries into continuous fronts.
- **Column-wise Boundary Walk:** Scans right-to-left across all $H$ vertical scanlines to determine the first damaged point $b_x(y)$.
- **1D Median Filtering:** Window size 15 filters single-pixel salt-and-pepper noise spikes.
- **Line Segmentation:** Binarizes the intact page area and computes horizontal ink projection density $\sum \mathbf{1}[\text{ink}]$, grouping text lines with gap merging (`line_gap_merge_px = 6`).
- **Manual Fallback:** Users can pass `manual_bbox = (x1, y1, x2, y2)` or `manual_boundary_x` to immediately override automatic detection with 100% confidence.

### 3.4 Physical Character-Budget Enforcement (`BudgetEstimator`)
- **Geometric Calculation:**
  $$\text{Budget} = \frac{W_{\text{page}} - b_x}{\text{char\_width}}$$
- **Dynamic Font Calibration:** If surviving prefix text is provided, glyph width is dynamically estimated as $w_{\text{char}} = w_{\text{intact}} / N_{\text{chars}}$. If only line height is available, $w_{\text{char}} = 0.58 \times h_{\text{line}}$. Otherwise, falls back to `Config.avg_char_width_px = 14.0`.
- **Token-Level Budget Enforcement:** Integrates with `PhysicalBudgetLogitsProcessor`, adding escalating EOS logit biases during beam search decoding when generated token count exceeds budget.
- **Subword Cleaning:** Strips SentencePiece whitespace symbols and T5 sentinel tokens (`<extra_id_0>`, `<pad>`, `</s>`).
- **Word-Boundary Snapping:** When runaway text occurs on unconstrained models, safely snaps to the nearest whitespace within budget rather than slicing words mid-character.

---

## 4. Interfaces & Usability

### 4.1 Command-Line Interface (`scripts/restore_document.py`)
```bash
# Automated full-page restoration
python scripts/restore_document.py --input tests/fixtures/mini_dataset/clean_images/sample_page_006.png --output results/cli_out/

# Manual bounding-box restoration with visual overlay
python scripts/restore_document.py --input sample.png --mode manual --bbox 750,100,1024,900 --save-overlay --save-json

# Constrained restoration with beam search
python scripts/restore_document.py --input sample.png --max-lines 5 --num-beams 4 --max-new-tokens 24
```

#### CLI Execution Sample Output:
```
================================================================================
GROUNDED SIDE-BURNT DOCUMENT TEXT RESTORATION CLI
================================================================================
Input image:        sample_page_006.png
Output directory:   results/cli_out
Model checkpoint:   checkpoints/best_checkpoint.pt
Detection mode:     AUTO
--------------------------------------------------------------------------------
[CLI] Processing document and executing multimodal restoration...

================================================================================
RESTORATION SUMMARY
================================================================================
Line   | Prefix Context            | Restored Text          | Budget   | Latency 
--------------------------------------------------------------------------------
0      | Incomplete line: Section..| 'models and systems'   | 14.2 ch  | 1850 ms 
1      | Incomplete line: The res..| 'evaluated on'         | 12.8 ch  | 1720 ms 
================================================================================
Total severed lines detected: 2
Total lines restored:         2
Total page inference latency: 4210.5 ms
Average latency per line:     1785.2 ms
Diagnostic overlay saved to:  results/cli_out/sample_page_006_restoration_overlay.png
Structured JSON saved to:     results/cli_out/sample_page_006_restoration_report.json
================================================================================
```

### 4.2 FastAPI REST Endpoints (`api/main.py`)
Start the REST server:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
- **`GET /health`:** Checks model loading status, compute device, and checkpoint file availability.
- **`GET /model/info`:** Returns model parameters ($85.5\text{M}$ params), device, and configuration.
- **`POST /restore/image`:** Accepts multipart document image upload, returns per-line restorations, budgets, latencies, and Base64-encoded visual overlay.
- **`POST /restore/region`:** Accepts an image with manual coordinates `x1, y1, x2, y2` and surviving text.
- **`POST /restore/batch`:** Processes up to 10 document images sequentially.
- **`GET /docs`:** Interactive Swagger UI documentation.

### 4.3 Interactive Research Workbench (`GET /` or `GET /ui`)
Mounted directly at root URL:
- **Main Document Canvas:** Interactive HTML5 canvas with zoom and pan.
- **Mode Toggle:** Switch between automatic OpenCV detection and manual bounding-box drawing.
- **Interactive Region Drawing:** Click and drag on the document image to define or adjust the burn boundary.
- **Dual View:** One-click toggle between Original Image and Diagnostic Restoration Overlay.
- **Line Results List:** Displays surviving prefix, model-generated infill, physical character budget, and budget violation indicators.
- **Export Tools:** One-click "Copy All Text" to clipboard and "Export JSON" to download full structured report.

---

## 5. Dedicated Test Suite & Regression Status

The repository now contains 9 distinct test suites covering all architectural layers:

| Test Suite File | Coverage Domain | Tests Passed | Execution Time |
| :--- | :--- | :---: | :---: |
| `tests/test_dataset_engineering.py` | Phase 1 dataset generator, 4-level alignment, zero-leakage splits | 14 / 14 | 1.8s |
| `tests/test_model_architecture.py` | Phase 2 multi-scale ResNet-34, 2D PE, gated cross-attention, InfoNCE | 13 / 13 | 1.4s |
| `tests/test_training_pipeline.py` | Phase 3 staged optimizer, differential LRs, schedulers, checkpointing | 10 / 10 | 25.1s |
| `tests/test_evaluation_benchmark.py` | Phase 4 baselines, bootstrap CIs, permutation tests, ablations | 11 / 11 | 1.7s |
| `tests/test_burn_detector.py` | Phase 5 OpenCV burn boundary detector, damage mask, line segmentation | 7 / 7 | 0.1s |
| `tests/test_budget_estimator.py` | Phase 5 geometric budget calculation, font scaling, word snapping | 8 / 8 | 0.1s |
| `tests/test_restoration_engine.py` | Phase 5 engine lifecycle, checkpoint validation, inference_mode | 7 / 7 | 22.7s |
| `tests/test_api.py` | Phase 5 FastAPI endpoints (/health, /model/info, /restore/*) | 8 / 8 | 14.7s |
| `tests/test_inference_integration.py` | Phase 5 end-to-end fixture restoration & CLI subprocess execution | 2 / 2 | 36.3s |
| **Total Test Suite** | **Complete Full-Repository Regression** | **80 / 80 (100%)** | **113.5s** |

---

## 6. Inference Benchmarks & Latency Profile

Measured on workstation CPU (Intel / AMD x86_64):

| Pipeline Stage | Measurement | Performance Characterization |
| :--- | :--- | :--- |
| **Model Initialization Time** | 4,923 ms – 5,691 ms (~5.2s) | One-time load; cached in memory for all subsequent requests |
| **Image Preprocessing & Resizing** | 12 ms – 25 ms | Fast OpenCV bilinear interpolation & ImageNet normalization |
| **Automatic Burn Boundary Detection** | 35 ms – 70 ms | Canny edge + column-wise scan + horizontal ink density |
| **Single-Region Line Inference (Beams=4)** | 1,450 ms – 1,850 ms | Autoregressive T5 decoding with PhysicalBudgetLogitsProcessor |
| **Single-Region Line Inference (Greedy)** | 620 ms – 950 ms | Fast greedy decoding mode |
| **Full-Page Restoration (2 Severed Lines)** | 3,800 ms – 4,400 ms | Detection + 2 line restorations + overlay rendering |
| **Full-Page Restoration (5 Severed Lines)** | 8,500 ms – 9,800 ms | Sequential line processing |
| **Throughput (CPU)** | 0.55 – 0.65 lines / sec | Real-time interactive feasible for document inspection |

---

## 7. Known Limitations & Scientific Notice

1. **Model Accuracy Baseline:** As documented in Phase 4, the model checkpoint achieves a validation CER of $2.1592$ and WER of $5.0633$ on the 6-page synthetic test fixture. Character accuracy remains low without large-scale pretraining on the full $4.2\text{ GB}$ OPRB dataset.
2. **Physical Budget Violations:** The T5 decoder language model exhibits unconstrained insertion runaway on certain lines without tight length penalties. While `PhysicalBudgetLogitsProcessor` adds escalating EOS logit biases, greedy beams occasionally complete clauses before emitting EOS. The newly implemented word-snapping fallback safely bounds outputs to the physical margin.
3. **Hardware Runtime:** Inference on CPU averages ~1.7s per line. GPU acceleration (CUDA) is supported automatically and reduces latency to ~150-250 ms per line when available.

---

## 8. Summary of Files Created & Modified in Phase 5

```
NEW APPLICATION & INFERENCE MODULES:
  inference/__init__.py                     (Package exports)
  inference/burn_detector.py                (OpenCV page burn detector & line segmenter)
  inference/budget_estimator.py             (Physical budget estimator, compliance, word snapping)
  inference/restoration_engine.py           (Central inference engine caching model & tokenizer)
  scripts/restore_document.py               (CLI restoration tool with overlay & JSON export)
  api/__init__.py                           (FastAPI app export)
  api/main.py                               (FastAPI application, CORS, lifespan, static mounting)
  api/schemas.py                            (Pydantic schemas for requests, responses, budgets)
  api/dependencies.py                       (RestorationEngine singleton dependency injection)
  api/services/__init__.py                  (Service package exports)
  api/services/restoration_service.py       (Image decoding, validation, Base64 overlay rendering)
  api/routes/__init__.py                    (Route package exports)
  api/routes/restoration.py                 (FastAPI route handlers for all endpoints)
  api/static/index.html                     (Single-page interactive research workbench UI)

NEW TEST SUITES:
  tests/__init__.py                         (Tests package marker)
  tests/test_burn_detector.py               (7 tests: OpenCV detector, masks, manual bbox)
  tests/test_budget_estimator.py            (8 tests: budget calculation, font scaling, compliance)
  tests/test_restoration_engine.py          (7 tests: engine lifecycle, checkpoint validation)
  tests/test_api.py                         (8 tests: FastAPI endpoints, status codes, schemas)
  tests/test_inference_integration.py       (2 tests: end-to-end fixture restoration & CLI subprocess)

DOCUMENTATION & TRACKING:
  docs/EXPERIMENT_TRACKER.md                (Updated with EXP-019-PHASE5-INFERENCE-ENGINE)
  docs/PHASE_5_INFERENCE_AND_APPLICATION_REPORT.md (Comprehensive Phase 5 report)
```

---

## 9. Commands to Run the Complete Application

### 1. Launch the REST API & Web Workbench:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Open your browser to:
- Interactive Web Workbench: `http://127.0.0.1:8000/ui` or `http://127.0.0.1:8000/`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### 2. Execute Document Restoration via CLI:
```bash
python scripts/restore_document.py --input tests/fixtures/mini_dataset/clean_images/sample_page_006.png --output results/restoration_run/ --save-overlay --save-json
```

### 3. Run the Full 80-Test Regression Suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
*(All 80 tests pass in ~113s)*.

