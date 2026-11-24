# Dataset Card: Synthetic Document Fixture v1.0 & OPRB Survey

## Dataset Description
- **Dataset Name:** Synthetic Document Fixture v1.0 (Active Local Fixture)
- **External Surveyed Dataset:** Open Paleographic Document Restoration Benchmark (OPRB)
- **Primary Domain:** Historical and academic document page images with simulated right-margin fire damage.
- **Corpus Formats:** High-resolution PNG images ($800 \times 1000$ pixels), clean UTF-8 text transcripts, and word-level bounding-box annotations.

---

## Active Local Dataset (Synthetic Document Fixture v1.0)
The repository currently evaluates a verified 6-page synthetic fixture designed for reproducible benchmarking:
- **Total Pages:** 6 multi-column academic pages.
- **Total Text Lines:** 180+ lines, of which 30--34 lines intersect the right-margin damage boundary.
- **Image Resolution:** $800 \times 1000$ pixels.
- **Damage Provenance:** Generated using `SideBurnGenerator` (Phase 1):
  - Gaussian random walk boundary curve ($b_x(y)$).
  - Charcoal BGR color blending ($(18, 14, 12)$).
  - Scorch Gaussian noise injection ($\sigma = 6.0$).
  - Ghost-stroke fade band ($55\text{ px}$ width).

### Document-Level Zero-Leakage Split (`data/splits.json`):
- **Train Split (4 pages, 66.7%):** `sample_page_002`, `sample_page_003`, `sample_page_004`, `sample_page_005`.
- **Validation Split (1 page, 16.7%):** `sample_page_001`.
- **Test Split (1 page, 16.7%):** `sample_page_006`.
- **Disjoint Guarantee:** Document pages are mutually exclusive across splits. Lines from test pages never appear in training.

---

## External Dataset Survey: OPRB (`kpurkayastha/OPRB`)
To prepare for large-scale pretraining, an acquisition survey (`docs/dataset_acquisition_report.md`) was conducted:
- **Repository:** Hugging Face `kpurkayastha/OPRB`
- **Lineage:** Derived from DocBank (academic layouts) and arXiv documents.
- **License:** **Creative Commons Attribution 4.0 International (CC-BY-4.0)**.
- **Volumes:**
  - `train_images.tar.gz` (3.03 GB) + `train_gt.tar.gz` (97.6 MB)
  - `test_images.tar.gz` (1.06 GB) + `test_gt.tar.gz` (34.2 MB)
  - Total Compressed: ~4.22 GB (~8.5 GB uncompressed).
- **Current Status:** Surveyed and cataloged; **not downloaded locally**, preserving local bandwidth and disk space until explicitly authorized.

---

## Ground-Truth Alignment Strategy
Ground truth missing text is established via a 4-level priority cascade:
1. **Level 1 (Exact Character Boxes):** Word bounding box intersection with the burn front $b_x$.
2. **Level 2 (Word Box Coordinate Intersection):** Geometric boundary splitting where complete words to the left of $b_x$ are assigned to surviving prefix and words to the right to missing target.
3. **Level 3 (Ink Projection Profile):** Horizontal density thresholding on pixel columns.
4. **Level 4 (Uncertain Fallback):** Flags ambiguous boundary cuts.

---

## Known Dataset Limitations
1. **Small Sample Size:** The 6-page fixture provides $N = 30\text{--}34$ physical line instances. While sufficient to verify algorithmic correctness, it cannot replace large-scale training.
2. **Synthetic vs. Natural Fire Damage:** Synthetic random walks do not fully simulate high-temperature paper embrittlement, smoke soot gradients, 3D paper cockling, or water damage from firefighting efforts.

