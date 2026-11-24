# Model Card: ForensicT5Restorer v2.0

## Model Details
- **Model Name:** ForensicT5Restorer v2.0
- **Model Architecture:** Multimodal Dual-Stream Sequence-to-Sequence Transformer
  - **Text Backbone:** T5-small ($d_{\text{model}} = 512$, 6 encoder layers, 6 decoder layers, 8 attention heads, 60.5M parameters)
  - **Visual Backbone:** Native ResNet-34 with Multi-Scale Layer 3 + Layer 4 Feature Fusion (22.5M parameters)
  - **Spatial Embeddings:** 2D spatial positional embeddings ($4 \times 8$ grid = 32 visual tokens)
  - **Cross-Modal Fusion:** Gated Multi-Head Cross-Attention with learnable sigmoid residual gate
  - **Alignment Objective:** InfoNCE Contrastive Stroke Aligner ($\tau = 0.07$)
  - **Constraint Processing:** Autoregressive `PhysicalBudgetLogitsProcessor`
- **Total Parameters:** 85,476,929 (~85.5 Million)
- **Trainable Parameters:** 85,476,929 (Stage 2 Joint Optimization)
- **Primary Checkpoint:** `checkpoints/best_checkpoint.pt` (978.68 MB, SHA-256: `2403d3190e8e3c2863d23734c881f1b60e8bc58c762652ad96e9a1a72c06fc12`)
- **License:** Apache 2.0 / Academic Research Use

---

## Intended Use
- **Primary Use Case:** Research-grade reconstruction and infilling of partially destroyed text lines along fire-damaged (side-burnt) document margins.
- **Target Users:** Document intelligence researchers, digital preservationists, cultural heritage archivists.
- **Application Context:** Interactive exploration where human archivists inspect restored text hypotheses alongside physical character budgets and diagnostic visual overlays.

---

## Out-of-Scope & Misuse Cases
- **High-Stakes Legal Forensics:** The model must **never** be used as authoritative evidence in legal proceedings or judicial document verification without human expert confirmation.
- **Unverified Inpainting:** Model outputs must never be painted onto original historical scans and presented as verified authentic archival content.
- **Whole-Document Generation:** The model is specialized for margin span infilling ($\le 32$ tokens) and is not intended for ungrounded creative text generation.

---

## Training Data & Provenance
- **Dataset:** Calibrated Synthetic Document Fixture v1.0.
- **Lineage:** Clean multi-column academic layouts corrupted with calibrated Gaussian random-walk burn contours, charcoal blending, and scorch noise.
- **Split Distribution:** Page-level zero-leakage split (`data/splits.json`): 4 training pages ($66.7\%$), 1 validation page ($16.7\%$), 1 testing page ($16.7\%$).
- **External Corpus Notice:** The external 4.22 GB Open Paleographic Document Restoration Benchmark (OPRB) was surveyed under CC-BY-4.0 but has not yet been downloaded or fine-tuned locally.

---

## Benchmark Performance
Evaluated across the standardized evaluation fixture:
- **Mean Raw CER:** 2.1592 (unbounded Levenshtein distance due to insertion runaway).
- **Mean Normalized CER:** 2.1397 (bounded).
- **Mean Word Error Rate (WER):** 5.0633.
- **Strict Position Character Accuracy:** 2.00% (exact 1-to-1 index alignment).
- **Average Inference Latency (CPU):** 1,785 ms per line ($0.55\text{ lines/sec}$).
- **Comparison to Baseline:** Reduces raw CER by 0.1490 absolute and WER by 1.3667 absolute compared to Text-Only T5 (CER 2.3082, WER 6.4300).

---

## Known Scientific Limitations
1. **Weak Visual Stroke Grounding:** In controlled visual stress testing, zeroed visual inputs ($\text{CER} = 2.1547$) and spatially shuffled visual inputs ($\text{CER} = 2.1592$) produce almost identical error to normal crops ($\text{CER} = 2.1592$). The model relies primarily on linguistic context rather than fine character stroke shapes.
2. **Insertion Runaway & Budget Violations:** The unconstrained autoregressive decoder frequently continues generating words beyond the physical boundary. Post-generation word snapping in `BudgetEstimator` is required to enforce margin compliance.
3. **Small Sample Size:** Current benchmark results derive from 30--34 line instances on 6 fixture pages. Statistical conclusions are preliminary ($p = 0.1730$).

---

## Ethical Considerations
- **Historical Hallucination Risk:** Generative language models can hallucinate plausible historical names, dates, or terms that distort historical reality. The application explicitly separates surviving text from generated text and provides budget violation warnings.

