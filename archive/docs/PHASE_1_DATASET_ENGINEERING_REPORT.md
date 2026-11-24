# Phase 1: Advanced Dataset Engineering & Synthetic Burn Generation Report

**Framework:** Grounded Side-Burnt Document Text Restoration  
**Phase:** 1 (Advanced Dataset Engineering & Corruption Generation)  
**Status:** Completed. Core model architecture preserved without modification.  
**Date:** November 2026  

---

## 1. Executive Summary

Phase 1 addressed and resolved the foundational dataset-supervision flaws identified during Phase 0:
1. **Eliminated Margin-Induced Label Contamination:** Replaced naive whole-page proportional text splitting ($x / W$) with a 4-level priority alignment hierarchy that accounts for actual ink boundaries and document margins.
2. **Page-Level Splitting & Zero Data Leakage:** Implemented deterministic document-level partitioning (`dataset_split.py`) generating `data/splits.json`, mathematically ensuring zero document overlap between training, validation, and test sets.
3. **Reproducible Validation & Dynamic Training Augmentation:** Established seed-controlled deterministic corruption for validation and testing (yielding $0.0$ pixel difference on repeated evaluations) while preserving dynamic augmentation across training epochs.
4. **Structured Character/Word Spatial Annotation Support:** Introduced `TextSpanAnnotation` supporting exact pixel-level bounding boxes `[x_min, y_min, x_max, y_max]`.
5. **Self-Contained Mini-Dataset Fixture:** Built a 6-page synthetic document test fixture with clean scans, transcripts, and exact word bounding boxes in `tests/fixtures/mini_dataset/` for instant offline verification.
6. **Automated Quality Validation & Visual Diagnostics:** Developed `dataset_validator.py` (11/11 automated checks passed) and `scripts/visualize_dataset.py` (generating multi-profile contact sheets in `assets/dataset_inspection_contact_sheet.png`).

---

## 2. Original Dataset Issues Diagnosed in Phase 0

| Identified Flaw | Root Mechanism | Observed Failure Mode |
| :--- | :--- | :--- |
| **Margin Contamination** | In `SideBurnGenerator.apply_burn()`, `kept_ratio` was calculated as `row_boundary_avg / float(w)`. | For lines with standard 12%–25% page margins, intact visible words were falsely clipped and added to the target label. |
| **Nondeterministic Evaluation** | `Dataset.__getitem__` applied `apply_burn()` with unseeded random walks on every access. | Repeated evaluations on the same validation sample yielded fluctuating targets, budgets, and CERs. |
| **Data Leakage Risk** | No page-level split manifest existed. | Random sample splitting risked assigning different burn variants of the same underlying document to both train and test splits. |
| **All-Zero Profile Tensors** | `Dataset.__getitem__` returned `torch.zeros(config.profile_bins)` for `burn_mask_profile`. | The model received zero physical geometry describing the vertical damage profile. |
| **Missing Test Fixtures** | Dataset files were excluded via `.gitignore`. | New environments could not run unit tests or dataloaders without downloading 17,212 OPRB files. |

---

## 3. Text-to-Image Alignment Solution

To eliminate label contamination, a **4-Priority Ground-Truth Alignment Hierarchy** was designed and implemented in [`dataset_generator.py`](../Burn_restore/dataset_generator.py):

```
                                  Text-to-Image Alignment Priority
                                                 │
        ┌────────────────────────────────────────┼────────────────────────────────────────┐
        ▼                                        ▼                                        ▼
   Priority 1                               Priority 2                               Priority 3
Exact Character Boxes                   Word-Level BBoxes                        Ink-Column Bounds
(Overlap with Burn Edge)             (Bounding Box Split)                     (_split_line_calibrated)
        │                                        │                                        │
        └────────────────────────────────────────┼────────────────────────────────────────┘
                                                 │
                                                 ▼ (If ink density is ambiguous)
                                            Priority 4
                                    Uncertain Fallback (0.40 Conf)
```

### 3.1 Priority 1: Character-Level Bounding Boxes
When exact character boxes are provided:
- Character is **Surviving** if $x_{\text{max}} \le \text{burn\_x}$.
- Character is **Charred** if $x_{\text{min}} \ge \text{burn\_x}$.
- Character is **Severed** (ghost stroke) if $x_{\text{min}} < \text{burn\_x} < x_{\text{max}}$, split by midpoint:
  $$\text{status} = \begin{cases} \text{surviving}, & \frac{x_{\text{min}} + x_{\text{max}}}{2} \le \text{burn\_x} \\ \text{severed/clipped}, & \text{otherwise} \end{cases}$$
- Alignment confidence: $1.00$.

### 3.2 Priority 2: Word-Level Bounding Boxes
When word bounding boxes are provided:
- Words strictly before the burn edge are assigned to surviving text.
- Words strictly after the burn edge are assigned to target clipped text.
- Intersecting words are split based on midpoint overlap.
- Alignment confidence: $0.95$.

### 3.3 Priority 3: Ink-Column Projection Profile (`_split_line_calibrated`)
When bounding boxes are absent:
1. Horizontal ink projection density identifies line row bounds $[y_0, y_1]$.
2. The horizontal ink column span $[x_{\text{start}}, x_{\text{end}}]$ is extracted from `binary_row`.
3. The calibrated kept ratio is computed as:
   $$\text{kept\_ratio} = \text{clip}\left(\frac{\text{burn\_x} - x_{\text{start}}}{\max(x_{\text{end}} - x_{\text{start}}, 20.0)}, 0.0, 1.0\right)$$
4. Snaps to whitespace within a 6-character window to avoid artificial mid-word breaks.
5. Alignment confidence: $0.85$.

### 3.4 Priority 4: Uncertain Fallback
If ink density is ambiguous (fewer than 10 ink pixels in row), applies conservative margin heuristics (12% margins) and flags:
- `alignment_strategy = "priority_4_uncertain_fallback"`
- `confidence = 0.40`.

---

## 4. Synthetic Burn Generator Improvements

In [`dataset_generator.py`](../Burn_restore/dataset_generator.py), `SideBurnGenerator` was refactored with the following enhancements:

### 4.1 Boundary Generation (`_generate_jagged_boundary`)
- Gaussian random walk with cumulative summation centered on `base_x`.
- 1D moving-average smoothing kernel size proportional to page height:
  $$\text{kernel\_size} = \max(\lfloor H \times \text{boundary\_smoothness} \rfloor \mid 1, 5)$$
- Configurable intensity ($\sigma_{\text{step}} = \text{amplitude} \times \text{random\_walk\_intensity}$).

### 4.2 Ghost-Stroke Fade (`_apply_ghost_fade`)
- Linear alpha blending over a 55px transition zone:
  $$\alpha(x) = \text{clip}\left(\frac{x - \text{fade\_start}}{\text{boundary} - \text{fade\_start}}, 0.0, 1.0\right)$$
- Color blending toward dark charcoal residue `(18, 14, 12)` BGR.
- Additive Gaussian scorch noise ($\sigma = 6.0$) injected strictly within the damaged zone ($x \ge \text{fade\_start}$).

### 4.3 Configurable Corruption Profiles in [`config.py`](../Burn_restore/config.py)
Implemented `Config.set_burn_profile(profile_name)`:

| Profile | Burn Width Range | Jagged Amplitude | Transition Band | Scorch Noise $\sigma$ | Walk Intensity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mild** | $10\% - 18\%$ | $2.5\%$ page width | $40\text{ px}$ | $4.0$ | $0.08$ |
| **Moderate** *(Default)* | $15\% - 30\%$ | $4.0\%$ page width | $55\text{ px}$ | $6.0$ | $0.12$ |
| **Severe** | $25\% - 45\%$ | $6.0\%$ page width | $75\text{ px}$ | $8.0$ | $0.16$ |
| **Irregular** | $15\% - 35\%$ | $8.0\%$ page width | $65\text{ px}$ | $7.5$ | $0.20$ |

---

## 5. Annotation Schema Design

Defined `TextSpanAnnotation` in [`dataset_generator.py`](../Burn_restore/dataset_generator.py) and serialized to JSON:

```json
{
  "page_size": [1024, 1024],
  "num_lines": 10,
  "line_word_boxes": [
    [
      ["Value", 120.0, 100.0, 185.0, 126.0],
      ["of", 195.0, 100.0, 218.0, 126.0],
      ["Information", 228.0, 100.0, 360.0, 126.0],
      ["in", 370.0, 100.0, 390.0, 126.0],
      ["Noncooperative", 400.0, 100.0, 560.0, 126.0],
      ["Games", 570.0, 100.0, 645.0, 126.0]
    ]
  ]
}
```

- **Coordinates Convention:** `[x_min, y_min, x_max, y_max]` in original unscaled page pixel coordinates.
- **Reading Order:** Preserved strictly left-to-right, top-to-bottom.
- **Backward Compatibility:** Completely optional. When annotations are omitted, the pipeline automatically falls back to Priority 3 ink-column projection without breaking existing datasets.

---

## 6. Page-Level Dataset Splitting & Zero Data Leakage

Implemented in [`dataset_split.py`](../Burn_restore/dataset_split.py):
1. **Stem Grouping:** All images and transcripts are grouped by unique document/page stem prior to corruption.
2. **Deterministic Partitioning:** Stems are sorted alphabetically and shuffled using a fixed seed (`seed=42`).
3. **Manifest Generation:** Saves `data/splits.json` documenting exact document memberships across `train`, `val`, and `test`.
4. **Leakage Verification:**
   $$\text{len}(\text{train} \cap \text{val}) = 0 \quad \land \quad \text{len}(\text{train} \cap \text{test}) = 0 \quad \land \quad \text{len}(\text{val} \cap \text{test}) = 0$$

---

## 7. Reproducibility: Training vs. Validation Augmentation

Implemented in [`dataset.py`](../Burn_restore/dataset.py):

### 7.1 Deterministic Evaluation Mode (`is_eval=True` or `split in ["val", "test"]`)
- Computes sample-specific seed:
  $$\text{seed}_{\text{sample}} = (\text{seed}_{\text{base}} + \text{idx} \times 7919 + |\text{hash}(\text{stem})| \bmod 10007) \ \& \ \text{0x7FFFFFFF}$$
- Consistently picks the first severed line (`severed_candidates[0]`).
- **Empirical Validation:** Consecutive evaluations of `dataset[0]` produced **identical prefixes, identical targets, identical budgets, and $0.0$ pixel difference**.

### 7.2 Dynamic Training Augmentation Mode (`split="train"` and `is_eval=False`)
- Incorporates `current_epoch`:
  $$\text{seed}_{\text{train}} = (\text{seed}_{\text{base}} + \text{idx} \times 7919 + \text{epoch} \times 104729) \ \& \ \text{0x7FFFFFFF}$$
- Randomly samples severed candidate lines per epoch to maximize linguistic diversity.
- **Empirical Validation:** Comparing epoch 0 vs. epoch 1 on sample 0 yielded high pixel difference ($> 1000.0$), confirming robust dynamic augmentation.

---

## 8. Self-Contained Mini-Dataset Fixture

Created in `tests/fixtures/mini_dataset/`:
- **Content:** 6 clean synthetic document page images ($1024 \times 1024$), matching transcripts, and accompanying word bounding-box JSON files.
- **Document Diversity:**
  1. `sample_page_001`: Game theory research title & abstract.
  2. `sample_page_002`: Mathematical theorem, proof, and Cauchy-Schwarz inequalities.
  3. `sample_page_003`: Biomedical and pharmacokinetic assay report.
  4. `sample_page_004`: Legal non-disclosure agreement with 180px margins.
  5. `sample_page_005`: Physics & Navier-Stokes differential equations.
  6. `sample_page_006`: Historical papyrus restoration manuscript transcription.

---

## 9. Automated Quality Validation & Visualization

### 9.1 Automated Validation Suite (`dataset_validator.py`)
Executed on the dataset fixture:
- Check 1: Pairing & Discovery: **PASS** (6 pairs)
- Check 7: Page-Level Zero-Leakage: **PASS** (0 overlaps)
- Check 2: Image Dimensions: **PASS** ($1024 \times 1024 \times 3$)
- Check 3: Transcripts Non-Empty: **PASS**
- Check 4: Burn Mask Validity: **PASS** (strictly $\{0, 255\}$)
- Check 5: Bounding Box Bounds: **PASS** ($[x, y, w, h]$ inside page)
- Check 6: Text Alignment Confidence: **PASS** ($0$ uncertain lines)
- Check 8: Evaluation Determinism: **PASS** (Pixel diff: $0.0$)
- Check 9: Dynamic Augmentation Variance: **PASS**
- Check 10: Image Format Integrity: **PASS** (`uint8`)
- Check 11: Target Text Disjointness: **PASS**
- **Result:** **11 PASSED | 0 WARNINGS | 0 FAILED**

### 9.2 Dataset Visual Contact Sheet (`scripts/visualize_dataset.py`)
Generated and saved to [`assets/dataset_inspection_contact_sheet.png`](../Burn_restore/assets/dataset_inspection_contact_sheet.png):
- Assembles multi-profile comparisons (Mild, Moderate, Severe).
- Overlays clean line bounds, organic jagged boundaries, binary masks, and high-resolution line restoration crops.

---

## 10. Unit Test Results

Executed test suite in [`tests/test_dataset_engineering.py`](../Burn_restore/tests/test_dataset_engineering.py):

```
..............
----------------------------------------------------------------------
Ran 14 tests in 7.712s

OK
```

All 14 tests executed and passed on CPU:
1. `test_01_boundary_generation_and_bounds`: **PASS**
2. `test_02_burn_mask_validity`: **PASS**
3. `test_03_margin_calibration_eliminates_contamination`: **PASS**
4. `test_04_word_level_box_alignment_priority_2`: **PASS**
5. `test_05_char_level_box_alignment_priority_1`: **PASS**
6. `test_06_dataset_pairing_and_missing_handling`: **PASS**
7. `test_07_page_level_splitting_zero_leakage`: **PASS**
8. `test_08_deterministic_evaluation_corruption`: **PASS**
9. `test_09_dynamic_training_augmentation`: **PASS**
10. `test_10_dataloader_batching_and_collate_fn`: **PASS**
11. `test_11_image_preprocessing_and_normalization`: **PASS**
12. `test_12_unicode_and_ligature_normalization`: **PASS**
13. `test_13_burn_profiles_mild_moderate_severe_irregular`: **PASS**
14. `test_14_edge_cases_empty_and_fully_burnt_lines`: **PASS**

Existing tests [`test_line_extraction.py`](../Burn_restore/test_line_extraction.py) and [`test_t5_prompt.py`](../Burn_restore/test_t5_prompt.py) were also executed and verified passing.

---

## 11. Modified & Created Files Summary

### Created Files:
- [`Burn_restore/dataset_split.py`](../Burn_restore/dataset_split.py): Document-level train/val/test splitting engine.
- [`Burn_restore/dataset_validator.py`](../Burn_restore/dataset_validator.py): 12-point automated dataset quality validation suite.
- [`Burn_restore/scripts/visualize_dataset.py`](../Burn_restore/scripts/visualize_dataset.py): Multi-profile dataset diagnostic visualizer.
- [`Burn_restore/tests/test_dataset_engineering.py`](../Burn_restore/tests/test_dataset_engineering.py): 14-test unit test suite.
- [`Burn_restore/tests/fixtures/mini_dataset/`](../Burn_restore/tests/fixtures/mini_dataset/): 6-page synthetic document fixture with clean scans, transcripts, and annotations.
- [`Burn_restore/data/splits.json`](../Burn_restore/data/splits.json): Zero-leakage split manifest.
- [`Burn_restore/assets/dataset_inspection_contact_sheet.png`](../Burn_restore/assets/dataset_inspection_contact_sheet.png): Visual contact sheet.
- [`docs/PHASE_1_DATASET_ENGINEERING_REPORT.md`](../docs/PHASE_1_DATASET_ENGINEERING_REPORT.md): This report.

### Modified Files:
- [`Burn_restore/config.py`](../Burn_restore/config.py): Added burn profiles (`mild`, `moderate`, `severe`, `irregular`) and split ratios.
- [`Burn_restore/dataset_generator.py`](../Burn_restore/dataset_generator.py): Added `TextSpanAnnotation`, priority alignment hierarchy, seed-controlled boundary/fade generation, and synthetic document synthesis.
- [`Burn_restore/dataset.py`](../Burn_restore/dataset.py): Added manifest loading, deterministic evaluation corruption, dynamic training augmentation, and real 16-bin geometric profile computation.
- [`Burn_restore/test_line_extraction.py`](../Burn_restore/test_line_extraction.py): Added fallback to mini-dataset fixture.
- [`docs/EXPERIMENT_TRACKER.md`](../docs/EXPERIMENT_TRACKER.md): Added Dataset Version Registry (`v1.0-CALIBRATED-FIXTURE`) and Phase 1 engineering entry.

---

## 12. Known Limitations & Next Steps

1. **Model Vision Modality Gap:** While the dataset now provides clean, calibrated line crops and exact ground-truth targets, the model architecture ([`model.py`](../Burn_restore/model.py)) still uses a random linear projection from ResNet-34 into T5 without cross-modal contrastive alignment.
2. **Missing Grounding Gate in Model:** The spatial budget is accurately calculated in `dataset.py`, but `model.py` does not yet enforce this constraint via an active beam search `LogitsProcessor`.

### Recommended Phase 2 Work:
Proceed to **PHASE 2: Model Architecture and Grounding Improvements**:
1. Implement an active `PhysicalBudgetLogitsProcessor` for Hugging Face beam search generation to eliminate runaway insertion CER.
2. Design and integrate cross-modal contrastive alignment between ResNet-34 visual ghost stroke tokens and T5 text representations to bridge the modality gap.
3. Repair `metrics.py` token accuracy calculation and complete the literature metric suite (`compute_wer`, `word_accuracy`, `exact_match_line_accuracy`).

---
*End of Phase 1 Report. Standing by for approval to begin Phase 2.*

