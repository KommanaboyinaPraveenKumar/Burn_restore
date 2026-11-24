# Dataset Acquisition and Survey Report: Grounded Side-Burnt Document Text Restoration

**Project:** Grounded Side-Burnt Document Text Restoration  
**Date:** November 2026  
**Status:** Evaluated and Documented  

---

## 1. Executive Summary

This report provides a formal evaluation of public document image datasets to support high-fidelity research on side-burnt text restoration. Side-burnt document restoration requires:
1. High-resolution document images with clear typography and natural paragraph structures.
2. Character or word-level spatial bounding boxes to establish the ground-truth boundary between surviving ink and physically missing text.
3. Permissive licensing for academic reproducibility and open-source publication.
4. Manageable storage and download footprints compatible with local workstation environments.

---

## 2. Dataset Candidates Survey

| Attribute | Primary Candidate: OPRB | Alternative 1: DocBank | Alternative 2: FUNSD | Alternative 3: SROIE | Baseline: Synthetic Fixture v1.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dataset Name** | **OPRB** (Occluded Pages Restoration Benchmark) | DocBank | FUNSD | SROIE 2019 | Calibrated Fixture Dataset |
| **Source / Repo** | Hugging Face: [`kpurkayastha/OPRB`](https://huggingface.co/datasets/kpurkayastha/OPRB) | GitHub / ArXiv:2006.01038 | ICDAR / Hugging Face | ICDAR 2019 | Local Workspace (`tests/fixtures/mini_dataset/`) |
| **License** | **Creative Commons Attribution 4.0 (CC-BY-4.0)** | CC-BY-NC-SA 4.0 | Academic / Research | Research Only | Permissive (Native Project Assets) |
| **Download Size** | **Train:** 3.03 GB (`img`) + 97.6 MB (`gt`)<br>**Test:** 1.06 GB (`img`) + 34.2 MB (`gt`) | ~45 GB uncompressed | ~16 MB | ~50 MB | 0 MB (Already present on disk) |
| **Document Count** | ~50,000 document pages | 500,000 pages | 199 form pages | 1,000 receipt scans | 6 document pages (34 severed lines) |
| **Annotation Format** | Word TSV: `word \t x1 \t y1 \t x2 \t y2 \t r \t g \t b \t font \t tag` | Token-level TSV & XML | Semantic entity JSON | OCR TSV coordinates | Clean text lines + word bounding boxes |
| **Image Resolution** | $1024 \times 1024$ pixels | Varied ($1000 \times 1000$ to $2000 \times 2000$) | $754 \times 1000$ pixels | Low-res smartphone receipts | $1024 \times 1024$ pixels (Standardized) |
| **Layout Diversity** | Single-column, multi-column, abstracts, legal | Single and multi-column articles | Dense bureaucratic forms | Narrow tabular thermal receipt paper | Multi-layout academic, legal, technical |
| **Suitability for Side-Burn** | **Optimal:** Exact word bounding boxes match side-burn boundary geometry | High, but prohibitive storage requirements (~45 GB) | Poor: Form tables and boxes do not match natural line continuation | Very Poor: Receipts lack margins and paragraph continuation | High for debugging and testing; limited scale for general training |

---

## 3. In-Depth Evaluation of OPRB

### 3.1 Lineage and Provenance
OPRB (Occluded Pages Restoration Benchmark) was created by K. Purkayastha as an extension of the DocBank dataset (Li et al., 2020, ArXiv:2006.01038). It consists of document pages extracted from open-access scientific publications across computer science, physics, mathematics, and quantitative biology.

### 3.2 Ground Truth Geometry
Each ground-truth file (`.txt` / `.tsv`) records individual word occurrences with 10 tab-separated fields:
$$\text{word} \quad x_1 \quad y_1 \quad x_2 \quad y_2 \quad r \quad g \quad b \quad \text{font} \quad \text{tag}$$
- Coordinates $(x_1, y_1)$ and $(x_2, y_2)$ define the bounding box of each word normalized to page dimensions.
- Font family and RGB ink color metadata enable forensic distinction between header text, body text, and footnotes.
- In Phase 1, `SideBurnGenerator` was engineered to accept these word bounding boxes via Priority 2 alignment, directly mapping words into surviving prefix vs severed target.

### 3.3 Storage and Network Footprint
- **Total Compressed Download Size:**
  - `train/text/img.tar.gz`: 3,028.04 MB
  - `train/text/gt.tar.gz`: 97.63 MB
  - `test/text/img.tar.gz`: 1,057.77 MB
  - `test/text/gt.tar.gz`: 34.23 MB
- **Total Disk Requirement (Extracted):** ~8.5 GB.
- **Local Disk Verification:** Workspace drive `C:` has **211.73 GB** of free space, easily accommodating the full dataset.

---

## 4. Dataset Acquisition & Preprocessing Strategy

1. **Test Set Track (Immediate Download / Evaluation):**
   - The test set (`test/text/img.tar.gz` [1.06 GB] and `test/text/gt.tar.gz` [34.2 MB]) can be acquired to benchmark unseen real document performance.
2. **Preprocessing Pipeline (`Burn_restore/preprocess_oprb.py`):**
   - Parses word bounding boxes into natural line structures based on vertical center clustering and horizontal sorting.
   - Cleans unicode typographical ligatures (`\ufb01` $\rightarrow$ `fi`).
   - Produces clean paired transcripts (`data/oprb/cleaned_texts/`).
3. **Execution Protocol:**
   - In accordance with Phase 4 guidelines, automatic downloading of large multi-gigabyte archives requires explicit operator confirmation or staging. The evaluation framework has been implemented so that:
     - Synthetic and multi-track benchmarks run immediately on local verified data.
     - The download script (`download_oprb.py`) is validated and ready for full-scale execution upon approval.

---

## 5. Scientific Recommendation

- **Primary Benchmark Corpus:** Acquire `kpurkayastha/OPRB` test split (1.06 GB image archive + 34 MB ground truth) for large-scale unseen test evaluation.
- **Immediate Evaluation:** Utilize the 6-page multi-track fixture dataset (34 physical line instances) to validate all 7 baseline models, 10 ablation configurations, and 5 random seeds.

