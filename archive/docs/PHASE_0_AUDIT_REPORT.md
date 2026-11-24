# Phase 0: Complete Technical Audit & Baseline Analysis
**Framework:** Grounded Side-Burnt Document Text Restoration  
**Repository:** `https://github.com/KommanaboyinaPraveenKumar/Burn_restore.git`  
**Date:** November 2026  
**Auditor:** Senior Deep Learning Research Engineer & Systems Architect  

---

## 1. Executive Summary

A comprehensive architectural, code-level, and empirical audit of the `Burn_restore` codebase was executed to assess the feasibility, correctness, and baseline performance of the proposed **Grounded Side-Burnt Document Text Restoration** framework.

### Key Audit Findings
1. **Architectural Divergence:** The repository documentation ([`README.md`](../Burn_restore/README.md)) describes an earlier modular system (`GlobalContextEncoder` DistilBERT + `LocalForensicEncoder` ResNet-34 + custom `CrossModalFusion` + `SpatialAlignmentGate` + `TransformerDecoderHead`). In contrast, the actual codebase has migrated to an integrated Seq2Seq model ([`model.py`](../Burn_restore/model.py): `ForensicT5Restorer`) combining `t5-small` with a spatial `resnet34` prefix projection.
2. **Missing Spatial Grounding Gate:** The documented mathematical grounding constraint:
   $$\text{bias}(\text{step}) = \text{ReLU}(\text{step} - \text{budget}) \times \text{alignment\_penalty\_weight}$$
   is **not implemented** in the neural network forward or generate pass. In `model.py`, `budget` is an unreferenced dummy tensor, and `line_budget` in `generate()` is completely unused. Length constraints were merely applied post-hoc via Python string slicing (`pred[:max_chars]`) in benchmark scripts.
3. **Severe Modality Neglect (The Visual Gap):** Inspection of [`specimens_data.json`](../Burn_restore/specimens_data.json) confirms that the multimodal model (`pred_ours`) and a blind text-only language model (`pred_llm`) generate **identical predictions** across benchmark samples (`"mathematics."`, `"ple of"`, `"we"`). The randomly initialized visual linear projection layer (`nn.Linear(512, 512)`) without dedicated cross-modal alignment pretraining causes T5 to treat visual ghost stroke tokens as uninformative noise.
4. **Data Contamination in Proportional Splitting:** In [`dataset_generator.py`](../Burn_restore/dataset_generator.py), `apply_burn()` calculates text cuts using whole-page width proportions rather than ink bounding boxes. On pages with standard 12%–25% margins, intact visible words to the left of the burn boundary are erroneously sliced off and placed into the target ground truth, contaminating training supervision.
5. **Missing Pretrained Weights & Primary Dataset:** `checkpoints/final_checkpoint.pt` and `data/oprb_train/` were excluded by `.gitignore` and are absent from the cloned repository. Out-of-the-box execution of evaluation scripts fails with `FileNotFoundError`.
6. **Token Accuracy Evaluation Bug:** In [`metrics.py`](../Burn_restore/metrics.py), `compute_token_accuracy` truncates sequences to `min_len`. If a target has 4 tokens and the prediction emits only 2 matching tokens, accuracy is erroneously reported as $1.0$ ($100\%$), ignoring omission and insertion penalties.

---

## 2. Existing Project Architecture Map

```
Burn_restore/
├── config.py                                 # Central Config dataclass (burn parameters, dims, LR)
├── dataset_generator.py                      # Component 1: OpenCV burn simulation engine
├── dataset.py                                # Component 2: LineGroundedDocumentDataset (PyTorch)
├── model.py                                  # Component 3: ForensicT5Restorer (T5-small + ResNet-34)
├── train.py                                  # Component 4: Single-folder training loop
├── metrics.py                                # Evaluation metrics (CER, BLEU, ROUGE-L, UCSM)
├── download_oprb.py                          # Hugging Face Hub OPRB dataset downloader
├── preprocess_oprb.py                        # OPRB TSV bounding-box line clustering
├── test_evaluate.py                          # 50-sample quantitative evaluation script
├── test_budget_eval.py                       # Physical length-budget comparison script
├── test_line_extraction.py                   # Ink-band segmentation smoke test
├── test_t5_prompt.py                         # T5 sentinel span infilling validation
├── evaluate_literature_metrics.py            # Literature-aligned evaluation (Top-K, UCSM)
├── generate_comprehensive_method_benchmark.py# 6-panel qualitative visual comparison generator
├── generate_publication_specimens.py         # 4-panel publication specimen generator
├── specimens_data.json                       # Qualitative sample records & competing predictions
├── evaluation_results.txt                    # Stored evaluation outputs
├── evaluation_literature_metrics.txt         # Stored literature benchmark metrics
├── PROGRESS_REPORT.md                        # Running execution log (Entries 1–13)
├── SESSION_STATE.md                          # Context handoff & session recovery state
├── RESEARCH_REPORT_RESTORATION_METHODS.md    # SOTA survey (DocRevive, Ithaca, DocDiff)
├── README.md                                 # High-level architecture documentation
├── report.tex / report.html / report.pdf     # Academic reports & LaTeX paper
└── assets/                                   # Real dataset crops and qualitative comparison panels
```

---

## 3. File-by-File Implementation Analysis

### 3.1 [`config.py`](../Burn_restore/config.py)
- **Role:** Defines the central `Config` dataclass storing hyperparameters for corruption, cropping, neural architectures, and optimization.
- **Verification:**
  - `burn_min_ratio = 0.15`, `burn_max_ratio = 0.30` (physical side destruction: 15% to 30% width).
  - `transition_zone_px = 55` (ghost-stroke fade band width).
  - `line_crop_size = (128, 256)` (height, width for focused line crop).
  - `text_model_name = "t5-small"`, `embed_dim = 512`, `avg_char_width_px = 14.0`.
- **Finding:** Contains residual configuration fields from the deprecated DistilBERT architecture (`decoder_layers=6`, `decoder_heads=8`, `decoder_ff_dim=2048`, `profile_bins=16`) that are unreferenced in `ForensicT5Restorer`.

### 3.2 [`dataset_generator.py`](../Burn_restore/dataset_generator.py)
- **Role:** Implements `SideBurnGenerator` for synthetic document damage.
- **Verification:**
  1. `_generate_jagged_boundary`: Generates a Gaussian random walk smoothed with a moving average kernel ($1\%$ of height, minimum 5px). Centered around zero so `base_x` acts as the physical anchor. Verified mathematically sound.
  2. `_apply_ghost_fade`: Implements linear alpha blending:
     $$\alpha = \text{clip}\left(\frac{x - \text{fade\_start}}{\text{boundary} - \text{fade\_start}}, 0, 1\right)$$
     Blends toward dark charcoal `(18, 14, 12)` with additive Gaussian scorch noise ($\sigma = 6.0$). Confined strictly to the damaged region.
  3. `_binarize`: Otsu thresholding with inversion (`THRESH_BINARY_INV + THRESH_OTSU`).
  4. `_segment_lines`: Horizontal ink projection profile (`row_density > min_ink_row_density`) with gap merging (`line_gap_merge_px=6`).
  5. `_resolve_line_bands`: Uniform-split fallback when detected ink bands != transcribed lines.
- **Critical Flaw:** In `apply_burn()` (lines 243-244), the text cut is computed as:
  ```python
  kept_ratio = row_boundary_avg / float(w)
  kept, clipped = self._proportional_split(text, kept_ratio)
  ```
  This computes the split relative to whole page width `w=1024`, completely ignoring left and right margins. In our test with a 200px margin, visible text was incorrectly classified as clipped, contaminating the target labels.

### 3.3 [`dataset.py`](../Burn_restore/dataset.py)
- **Role:** Implements `LineGroundedDocumentDataset` (aliased to `SideBurnDocumentDataset`).
- **Verification:**
  - Correctly pairs `.png`/`.jpg` images with matching `.txt` transcripts by filename stem.
  - Generates focused $128 \times 256$ line crops centered vertically on the target line and horizontally at the burn edge ($\pm 120\text{px}$). Normalizes with ImageNet statistics.
  - Implements T5 native sentinel span infilling:
    - Input: `Context: {page_context} | Incomplete line: {prefix} <extra_id_0>`
    - Target: `<extra_id_0> {target_text} <extra_id_1>`
- **Critical Flaw:** `apply_burn` is executed dynamically inside `__getitem__` without per-index random seeding. Consecutive accesses to `dataset[i]` yield different burn boundaries, different target texts, and different character budgets. Moreover, because no train/val split logic is provided, multiple burn variants of the same source page leak across evaluation runs.

### 3.4 [`model.py`](../Burn_restore/model.py)
- **Role:** Implements `ForensicT5Restorer` (aliased to `GroundedDocumentRestorer`).
- **Verification:**
  - Encodes visual crops via `resnet34` (final pooling/fc removed), yielding spatial features of shape `(B, 512, 4, 8)`.
  - Flattens to 32 visual tokens: `(B, 32, 512)`.
  - Projects through `nn.Linear(512, d_model)` + `nn.LayerNorm(d_model)`.
  - Concatenates visual tokens with T5 text token embeddings: `torch.cat([vis_tokens, text_embeds], dim=1)` and feeds them into `self.t5.encoder`.
- **Critical Flaws:**
  1. `SpatialAlignmentGate` is absent. `budget = torch.zeros(...)` is a dead variable in `forward()`.
  2. `line_budget` in `generate()` is unreferenced.
  3. Prepending visual tokens directly into text embedding space without contrastive stroke pretraining causes severe modality neglect.

### 3.5 [`train.py`](../Burn_restore/train.py)
- **Role:** Main training loop for `ForensicT5Restorer`.
- **Verification:**
  - Uses AdamW with differential parameter groups (visual adapter `lr=1e-4`, T5 backbone `lr=config.learning_rate`).
  - Sets padding tokens in `labels` to `-100` so cross-entropy loss correctly ignores padding.
- **Critical Flaws:**
  1. **No Validation Split:** All data passed to `--images` and `--texts` is used for training. No held-out validation loss or CER is tracked.
  2. **No Best-Model Checkpointing:** Only saves `final_checkpoint.pt` at the end of the final epoch.
  3. **No Learning Rate Scheduler:** No warmup or decay schedule.
  4. **Token Accuracy Inflation:** Accuracy is computed over all tokens including the guaranteed sentinel tokens `<extra_id_0>` and `<extra_id_1>`, artificially inflating the reported accuracy.

### 3.6 [`metrics.py`](../Burn_restore/metrics.py)
- **Role:** Standalone evaluation metrics.
- **Verification:**
  - `compute_cer`: Evaluates edit distance / target length.
  - `compute_normalized_levenshtein`: $1.0 - \text{edit\_dist} / \max(\text{len}(P), \text{len}(T))$.
  - `compute_bleu`: Pure-Python sentence BLEU with Chen & Cherry smoothing.
  - `compute_rouge_l`: Longest common subsequence F1.
  - `normalize_text`: Unicode NFKD ligature decomposition and typography cleanup.
  - `compute_topk_accuracy`: Top-$K$ beam exact match.
  - `compute_ucsm`: DocRevive CVPR 2024 composite metric.
- **Critical Flaws:**
  1. `compute_token_accuracy` truncates sequences to `min_len`, completely ignoring length mismatch errors.
  2. `compute_wer`, `word_accuracy`, and `exact_match_line_accuracy` are missing.

---

## 4. Dataset Quality & Damage Synthesis Assessment

| Synthesis Step | Expected Research Behaviour | Current Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Jagged Boundary** | Organic non-linear scorch line | Smoothed Gaussian random walk ($\pm 4\%$ page width) | **Pass** |
| **Ghost Stroke Fade** | Low-contrast ink remnants in transition band | Linear alpha blend toward charcoal with scorch noise | **Pass** |
| **Missing Width Profile** | True missing pixel width per row | $(w - \text{boundary})$ vector | **Pass** |
| **Line Splitting** | Exact alignment with physical burn boundary | Naive whole-page proportional cut in `apply_burn` | **Fail (Margin Contamination)** |
| **Whitespace Snapping** | Preserves word boundaries | Snaps to whitespace within 6-char window | **Partial (Ignores partial glyphs)** |
| **Evaluation Determinism** | Fixed, reproducible damage for benchmark | Random generation on every call without seed control | **Fail (High Variance / Leakage)** |

---

## 5. Model Architecture & Grounding Assessment

### Modality Neglect Analysis
In [`specimens_data.json`](../Burn_restore/specimens_data.json), comparing outputs across models:
- **Specimen 1 (arXiv 1401.0001, Title Header):**
  - Prefix: `"Value of information in noncooperative"`
  - Ground Truth Target: `"games"`
  - `pred_ours`: `"mathematics."` (CER: 1.80)
  - `pred_llm` (Text-Only T5): `"mathematics."` (CER: 1.80)
- **Specimen 2 (Scientific Body Text):**
  - Prefix: `"...Braess' paradox and"`
  - Ground Truth Target: `"Cournot exam-"`
  - `pred_ours`: `"ple of"` (CER: 0.92)
  - `pred_llm`: `"ple of"` (CER: 0.92)
- **Specimen 3 (Mathematical Proof):**
  - Prefix: `"...by the above proportionality,"`
  - Ground Truth Target: `"(18)"`
  - `pred_ours`: `"we"` (CER: 1.00)
  - `pred_llm`: `"we"` (CER: 1.00)

**Conclusion:** The multimodal model's predictions are **100% identical** to a text-only language model that has never seen an image. The visual tokens contribute virtually zero gradient influence to the generated text distribution.

---

## 6. Grounding Constraint Audit

### Mathematical Assessment
The formula:
$$\text{bias}(\text{step}) = \text{ReLU}(\text{step} - \text{budget}) \times \text{alignment\_penalty\_weight}$$
is intended to penalize decoding once the token count exceeds the physical character budget:
$$\text{budget} = \frac{\text{missing\_px}}{\text{avg\_char\_width\_px}}$$

### Flaws in Current Implementation:
1. **Not Active in Model:** Neither `forward()` nor `generate()` applies this bias.
2. **Space Discrepancy:** The budget is in character space, but T5 decodes in subword token space (1 subword token $\approx 3.2$ characters). Applying a character budget directly to token decoding steps causes severe under-generation if unscaled.
3. **Logits Processor Missing:** In Hugging Face transformers, autoregressive decoding constraints must be implemented as a `LogitsProcessor`. No logits processor was implemented.
4. **Post-Hoc Slicing:** Previous scripts implemented length bounding purely as `pred[:max_chars]`, which brutally clips words mid-string and does not shape the decoding search beam.

---

## 7. Metrics Audit & Verification

Empirical test results on edge cases:

| Input Condition | `compute_cer` Result | `compute_token_accuracy` Result | Evaluation Assessment |
| :--- | :---: | :---: | :--- |
| Identical strings (`"hello"`, `"hello"`) | `0.0` | `1.0` | Correct |
| Runaway insertion (`"supercalifragilistic..."`, `"cat"`) | **`10.33` (1033% error)** | N/A | Flaw: Unbounded CER blows up macro-averages. |
| Missing tokens (`p=[10, 20]`, `t=[10, 20, 30, 40]`) | N/A | **`1.0` (100% accuracy)** | **Severe Bug:** Truncation to `min_len` ignores omissions. |
| Empty prediction (`""`, `"text"`) | `1.0` | N/A | Correct |
| Empty target (`"text"`, `""`) | `1.0` | N/A | Correct |
| Both empty (`""`, `""`) | `None` | `None` | Handled gracefully |

---

## 8. Baseline Experimental & System Status (Step 8)

### System Environment
- **OS:** Windows 11 (64-bit AMD64)
- **Python Version:** 3.11.9
- **PyTorch Version:** 2.10.0+cpu
- **CUDA Available:** `False` (CPU execution only on current PyTorch build)
- **Transformers Version:** 4.57.6
- **OpenCV Version:** 4.11.0
- **Torchvision Status:** Not currently installed in base environment (blocked by Windows `MAX_PATH` during standard pip installation; requires local venv install).
- **Pretrained Weights Verification:**
  - `t5-small` tokenizer: **Verified loaded** from Hugging Face hub (32,100 tokens).
  - `t5-small` model: **Verified loaded** and functional (generates valid infill text).
  - `resnet34`: Requires `torchvision` installation.
  - `final_checkpoint.pt`: **Missing** from repository (gitignored).
- **Dataset Availability:** `data/oprb_train/` is **missing** from repository (gitignored).

### Historical Baseline Performance (From Recorded Repository Logs)
From [`evaluation_results.txt`](../Burn_restore/evaluation_results.txt) and [`evaluation_literature_metrics.txt`](../Burn_restore/evaluation_literature_metrics.txt):
- **Bounded Macro Strip CER:** `0.8559` (raw: `0.8797`, normalized: `0.8742`)
- **Corpus Strip CER:** `0.9626`
- **Composite Full-Page CER:** `0.1901` - `0.1979`
- **Sentence BLEU-4:** `0.1832` - `0.2010`
- **ROUGE-L:** `0.2103` - `0.2250`
- **DocRevive UCSM:** `0.2216`
- **Top-1 Exact Match Accuracy:** `4.0%`
- **Top-3 Exact Match Accuracy:** `10.0%`
- **Top-5 Exact Match Accuracy:** `10.0%`

---

## 9. Bugs and Technical Risks Summary

1. **Ground Truth Margin Error:** `dataset_generator.py` uses page-width proportional cuts, injecting intact visible text into target strings.
2. **Missing Grounding Gate:** Model `forward` and `generate` completely ignore `budget`.
3. **Modality Neglect:** Random linear adapter results in $0\%$ visual grounding (`pred_ours == pred_llm`).
4. **Token Accuracy Truncation Bug:** `compute_token_accuracy` reports $100\%$ accuracy on severely truncated predictions.
5. **Missing Standard NLP Metrics:** `WER`, `character_accuracy`, `exact_match_line_accuracy` are absent from `metrics.py`.
6. **No Validation Pipeline:** `train.py` trains on 100% of input data without validation evaluation or best-checkpoint selection.
7. **Missing Fixtures & Assets:** Checkpoints and datasets are omitted from git tracking.

---

## 10. Prioritized Improvement Roadmap

| Priority | Proposed Improvement | Target Metric Impact | Expected Impact on Grounding | Complexity |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Calibrated Ink-Bound Line Splitting** | Prevents label noise; CER $-0.05$ | Ensures target matches physical burn cut | Low |
| **2** | **Spatial Alignment Logits Processor** | Directly reduces runaway insertion CER | Halts beam search when physical margin is reached | Medium |
| **3** | **Contrastive Cross-Modal Alignment** | BLEU $+0.10$, CER $-0.15$ | Forces model to differentiate ghost strokes from blanks | High |
| **4** | **Subword-Character Hybrid Heads** | Resolves partial glyph fragments | Directly maps 2px ink remnants to candidate characters | High |
| **5** | **Reproducible Val Split & Seeded Corruption** | Reliable, scientifically defensible evaluation | Zero data leakage between train and test | Low |
| **6** | **Complete Literature Metric Suite (WER, Top-K)** | Enables direct benchmarking against DocRevive/Ithaca | Standardized academic comparison | Low |

---

## 11. Recommended Phase 1 Implementation Plan

During **PHASE 1: Dataset Engineering and Corruption Generation**:
1. **Unify Splitting Logic:** Refactor `SideBurnGenerator.apply_burn()` to use `_split_line_calibrated()`, computing cuts using actual ink row column boundaries `(x_start, x_end)` rather than page width `w`.
2. **Deterministic Validation Benchmarking:** Implement seeded, reproducible damage generation for validation/test splits so evaluations are consistent across epochs.
3. **Document-Level Train/Val Splitting:** Partition dataset strictly by document filename stem before corrupting, eliminating cross-split data leakage.
4. **Synthetic Test Fixture:** Generate a verified, self-contained mini-dataset (clean images, transcripts, and calibrated burn pairs) within the repository for unit testing and CI smoke runs without requiring external downloads.
5. **Character-Level Bounding Box Annotation Support:** Add optional support for character/word-level bounding boxes (e.g., from OCR or TSV) to replace heuristic proportional splitting with exact pixel-to-glyph alignment.

---
*End of Phase 0 Audit Report. Standing by for approval to begin Phase 1.*

