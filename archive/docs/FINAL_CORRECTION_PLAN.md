# Final Correction Plan: Grounded Side-Burnt Document Text Restoration

**Project:** Grounded Side-Burnt Document Text Restoration  
**Repository:** `https://github.com/KommanaboyinaPraveenKumar/Burn_restore.git`  
**Date:** November 2026  
**Document Status:** AUTHORITATIVE FINAL AUDIT & CORRECTION PLAN  
**Purpose:** Comprehensive audit of all codebase modules, metrics, visual grounding, budget enforcement, dataset integrity, API, frontend, and publication artifacts prior to final release validation.

---

## 1. Executive Summary & Audit Matrix

Following an exhaustive codebase inspection across all Python source modules (`model.py`, `train.py`, `dataset.py`, `dataset_generator.py`, `metrics.py`), inference engines (`restoration_engine.py`, `burn_detector.py`, `budget_estimator.py`), API services (`api/`), test suites (`tests/`), checkpoints, and research reports (`docs/`, `paper/`), eight critical and high-priority issues have been identified.

This plan details each issue, root cause, severity, proposed correction, and verification procedure.

---

## 2. Comprehensive Issue Inventory

### Issue 1: Metric Inconsistency Across Historical Reports & Evaluation Splits
- **Severity:** **CRITICAL**
- **Location:** `docs/benchmark_results.json`, `docs/EXPERIMENT_TRACKER.md`, `docs/PHASE_4_EVALUATION_BENCHMARK_REPORT.md`, `docs/REPRODUCIBILITY.md`, `README.md`, `docs/FINAL_PROJECT_REPORT.md`.
- **Root Cause & Why It Is an Issue:**
  1. In Phase 4, `run_comprehensive_benchmark.py` evaluated across all 6 pages (34 severed lines) in some runs, but across 10 lines from `sample_page_006.png` in others.
  2. Baseline B (`Text_Only_T5`) was reported as $\text{CER} = 2.3082$ and $\text{WER} = 6.4300$ in Phase 4 `docs/benchmark_results.json`, but in Phase 6 EXP-A was reported as $\text{CER} = 2.1592$ and $\text{WER} = 5.0633$.
  3. In Phase 6, EXP-G was reported as $\text{CER} = 1.8420$ using post-hoc word-snapping, whereas in Phase 4 Baseline G was reported as $2.1592$ without word-snapping.
  4. Multiple conflicting definitions of "Normalized CER" were used: some scripts computed macro-averaged bounded accuracy $1 - \min(1, \text{CER}) \in [0.89, 0.95]$, while others reported $1 - \text{CER}$ clamped at $0.0$.
  5. The standard corpus-level definition $\text{CER} = \frac{\sum \text{edit\_distance}}{\sum \text{target\_chars}}$ was conflated with macro-average $\frac{1}{M}\sum \frac{\text{edit\_distance}_i}{\text{target\_chars}_i}$.
- **Proposed Correction:**
  1. Author a single canonical evaluation script (`scripts/run_canonical_evaluation.py`).
  2. Evaluate all baselines strictly on the official held-out test split (`sample_page_006.png`, 10 severed lines).
  3. Report both canonical corpus-level micro CER/WER and sample-level macro CER/WER explicitly with mathematical formulas.
  4. Record exact checkpoint paths, file sizes, and SHA-256 hashes for each experiment.
  5. Generate authoritative, non-conflicting artifacts: `docs/FINAL_METRICS.json`, `docs/FINAL_METRICS_REPORT.md`, and `docs/FINAL_EXPERIMENT_COMPARISON.csv`.
- **Verification:** Run `python scripts/run_canonical_evaluation.py` and verify all baseline comparisons share identical sample sets and metric definitions.

---

### Issue 2: Visual Modality Grounding Invariance (Language Prior Dominance)
- **Severity:** **CRITICAL**
- **Location:** `model.py` (`MultiScaleForensicEncoder`, `GatedCrossModalFusion`, `encode_multimodal`), `scripts/evaluate_visual_grounding.py`.
- **Root Cause & Why It Is an Issue:**
  1. When visual crops are blacked out (zeroed), spatially shuffled, Gaussian-blurred, or inverted, the model produces virtually identical output strings ($\text{CER} = 2.1592$ vs $2.1547$).
  2. Diagnostic inspection revealed that LayerNorm in `MultiScaleForensicEncoder` over `tokens + spatial_pos` squashes the feature variance of ResNet features relative to spatial embeddings.
  3. T5-small linguistic prior assigns $>99\%$ of decoder probability mass based on the surviving line prefix. The difference in T5 encoder last hidden states between a clean and zeroed image is only $0.2144$.
  4. Cross-attention gate parameter is $\approx 0.0001$ ($\text{sigmoid} \approx 0.5000$), meaning the gate remains in its initial unadapted state because fine-tuning on 4 pages was insufficient to ground character strokes.
- **Proposed Correction:**
  1. Add detailed diagnostic telemetry to `inference/restoration_engine.py` and `scripts/evaluate_visual_grounding.py` measuring visual feature norms, spatial embedding norms, cross-attention entropy, and gate weights.
  2. Implement a positive control test evaluating whether visual stroke differences can alter token probabilities in a controlled synthetic probe.
  3. Honestly document that visual grounding is weak and that the language model prior dominates generation on this 6-page dataset. State clearly in all research documentation that fine-grained character-stroke physical grounding remains an open research frontier. Create `docs/FINAL_VISUAL_GROUNDING_AUDIT.md`.
- **Verification:** Run `scripts/evaluate_visual_grounding.py` with full diagnostic logging and inspect before/after token-level probability shifts.

---

### Issue 3: Conflation of Decoder-Level Budget Enforcement vs Post-Processing Word Snapping
- **Severity:** **HIGH**
- **Location:** `model.py` (`PhysicalBudgetLogitsProcessor`), `inference/budget_estimator.py`, `docs/FINAL_PROJECT_REPORT.md`.
- **Root Cause & Why It Is an Issue:**
  1. The project claimed "100% budget violation before correction, 0% after word-snapping". Word-snapping is a post-processing string truncation heuristic, NOT proof that the autoregressive decoder enforces physical constraints.
  2. `PhysicalBudgetLogitsProcessor` in `model.py` calculates character excess on decoded tokens during beam search and boosts EOS logits while penalizing other tokens.
  3. When tested, `PhysicalBudgetLogitsProcessor` stopped generation at 23 characters for a 20.9-character budget, whereas unconstrained generation ran to 41 characters. Thus, the decoder constraint DOES reduce runaway text, but does not guarantee 0% margin violation without word snapping.
- **Proposed Correction:**
  1. Strictly decouple and report two independent metrics:
     - **Decoder-Level Margin Compliance:** Percentage of generated raw strings whose length is $\le \text{budget} + \text{buffer}$ during autoregressive generation.
     - **Post-Processing Margin Compliance:** Percentage of final strings conforming to budget after `BudgetEstimator` whitespace snapping.
  2. Add unit tests in `tests/test_budget_estimator.py` and `tests/test_model_architecture.py` covering multi-character SentencePiece tokens, punctuation, whitespace, small budgets ($\le 5$ chars), and budgets smaller than the target. Create `docs/FINAL_PHYSICAL_BUDGET_AUDIT.md`.
- **Verification:** Run standalone tests verifying decoder-level compliance and post-processing compliance independently.

---

### Issue 4: Non-Deterministic Python `hash()` in Dataset Augmentation Seeding
- **Severity:** **HIGH**
- **Location:** `Burn_restore/dataset.py` line 197.
- **Root Cause & Why It Is an Issue:**
  `sample_seed = (self.config.seed + idx * 7919 + abs(hash(stem)) % 10007) & 0x7FFFFFFF` uses Python's built-in `hash()`, which is randomly salted per process (`PYTHONHASHSEED`), violating cross-process reproducibility.
- **Proposed Correction:**
  Replace `abs(hash(stem))` with a deterministic SHA-256 hash:
  `int(hashlib.sha256(stem.encode('utf-8')).hexdigest()[:8], 16)`.
  Synchronize to both root and `Burn_restore/`.
- **Verification:** Run dataset split and item loading across separate Python processes; assert seeds and generated samples are bit-for-bit identical.

---

### Issue 5: Phase 3 Anomaly: Identical Validation CER Across Experiments EXP-A to EXP-E
- **Severity:** **MEDIUM**
- **Location:** `docs/phase_3_diagnostic_report.json`, `docs/PHASE_3_TRAINING_OPTIMIZATION_REPORT.md`.
- **Root Cause & Why It Is an Issue:**
  In Phase 3, EXP-A through EXP-E reported identical validation CER of 4.1250. Inspection of `docs/phase_3_diagnostic_report.json` revealed that the validation loop evaluated only a single line instance (`target = "agent."`, len 6) on the validation page, where all under-trained models produced default 30-character punctuation sequences (`edit_distance = 30`, `CER = 30 / 6 = 5.0`).
- **Proposed Correction:**
  Document this exact root cause in `docs/FINAL_CORRECTION_PLAN.md` and `docs/FINAL_METRICS_REPORT.md`. Verify that `train.py` evaluates across all severed lines on the validation split.
- **Verification:** Audit `train.py` validation loop across all validation lines.

---

### Issue 6: Backend Route Aliasing & Error Response Completeness
- **Severity:** **MEDIUM**
- **Location:** `api/routes/restoration.py`, `api/main.py`.
- **Root Cause & Why It Is an Issue:**
  The REST API defines `/restore/image`, but some CLI scripts and previous documentation referred to `/restore/page`. Also, testing for invalid file formats, zero-byte uploads, and missing bounding boxes should be verified.
- **Proposed Correction:**
  Add route alias `@router.post("/restore/page", include_in_schema=False)` pointing to `restore_document_image`. Add comprehensive test cases in `tests/test_api.py` for `/health`, `/model/info`, `/restore/image`, `/restore/region`, `/restore/batch`, empty files, and invalid formats.
- **Verification:** Run `unittest discover -s tests -p "test_api.py"` with 100% pass rate.

---

### Issue 7: Live Frontend Browser Workflow Validation
- **Severity:** **HIGH**
- **Location:** `api/static/index.html`.
- **Root Cause & Why It Is an Issue:**
  The application includes a single-page HTML5/JS workbench, but the complete end-to-end user workflow in an actual browser has not been formally audited and documented.
- **Proposed Correction:**
  Launch the local FastAPI server in the background, deploy the Chrome DevTools browser subagent to interact with `http://127.0.0.1:8000/`, upload a sample document, verify auto-detection overlay, test manual bounding box drag, trigger restoration, inspect line outputs, test copy and download actions, and test invalid upload error alerts. Record all browser DOM checks in `docs/FINAL_APPLICATION_VALIDATION.md`.
- **Verification:** Browser subagent execution logs and verified DOM state.

---

### Issue 8: Research Paper & Documentation Consistency Audit
- **Severity:** **MEDIUM**
- **Location:** `paper/main.tex`, `paper/README.md`, `docs/MODEL_CARD.md`, `docs/DATASET_CARD.md`, `README.md`.
- **Root Cause & Why It Is an Issue:**
  Older text-only baseline numbers (2.3082 vs 2.1592) and unclear descriptions of visual grounding limitations must be resolved across the paper and model cards. Explicit disclosures are required regarding the 6-page synthetic fixture size, the external OPRB survey, the dominant role of linguistic priors, and the distinction between decoder logits constraints and word snapping.
- **Proposed Correction:**
  Update `paper/main.tex` and all documentation cards to reflect canonical metrics from `docs/FINAL_METRICS.json`. Compile or document exact LaTeX status.
- **Verification:** Check all metrics and claims across `paper/` and `docs/`.

---

## 3. Execution Schedule & Verification Checkpoints

1. **Fix Code Defects:**
   - Replace `hash(stem)` in `Burn_restore/dataset.py` with deterministic SHA-256.
   - Add `/restore/page` alias in `api/routes/restoration.py`.
   - Update `PhysicalBudgetLogitsProcessor` character counting and SentencePiece handling.
2. **Standardize Evaluation Protocol:**
   - Implement `scripts/run_canonical_evaluation.py`.
   - Execute all baselines on official test split (`sample_page_006.png`, 10 lines).
   - Export `docs/FINAL_METRICS.json`, `docs/FINAL_METRICS_REPORT.md`, `docs/FINAL_EXPERIMENT_COMPARISON.csv`.
3. **Investigate Visual Grounding:**
   - Execute `scripts/evaluate_visual_grounding.py` with feature norm telemetry.
   - Author `docs/FINAL_VISUAL_GROUNDING_AUDIT.md`.
4. **Audit Physical Budget Enforcement:**
   - Measure decoder-level vs post-processing compliance independently.
   - Author `docs/FINAL_PHYSICAL_BUDGET_AUDIT.md`.
5. **Audit Dataset & Reproducibility:**
   - Author `docs/FINAL_DATASET_AUDIT.md` and `docs/FINAL_REPRODUCIBILITY_AUDIT.md`.
6. **Validate Application & Live Browser Frontend:**
   - Test all API endpoints in `tests/test_api.py`.
   - Launch server and run browser subagent workflow.
   - Author `docs/FINAL_APPLICATION_VALIDATION.md`.
7. **Synchronize Documentation & Research Paper:**
   - Update `paper/main.tex`, `paper/README.md`, `README.md`, `docs/MODEL_CARD.md`, `docs/DATASET_CARD.md`.
   - Create `docs/PROJECT_COMPLETION_CERTIFICATE.md`.

