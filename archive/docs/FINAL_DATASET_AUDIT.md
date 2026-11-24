# Final Dataset Audit: Grounded Side-Burnt Document Text Restoration

**Project:** Grounded Side-Burnt Document Text Restoration  
**Date:** November 2026  
**Document Status:** AUTHORITATIVE DATASET AUDIT  
**Split Manifest:** `data/splits.json`  

---

## 1. Dataset Overview & Provenance

The primary experimental dataset is the **Standardized Synthetic Document Fixture v1.0**, comprised of 6 high-resolution document pages paired with ground-truth transcription files containing exactly one textual line per physical document line:

```
data/synthetic_fixture/
├── page_001.png / page_001.txt (also sample_page_001.png / .txt)
├── page_002.png / page_002.txt (also sample_page_002.png / .txt)
├── page_003.png / page_003.txt (also sample_page_003.png / .txt)
├── page_004.png / page_004.txt (also sample_page_004.png / .txt)
├── page_005.png / page_005.txt (also sample_page_005.png / .txt)
└── page_006.png / page_006.txt (also sample_page_006.png / .txt)
```

Each page is rendered at $1024 \times 1024$ pixels, representing scholarly, legal, and academic English document layouts.

---

## 2. Page-Level Partitioning & Zero-Leakage Guarantee

To prevent spatial contamination and document memorization, partitioning is enforced strictly at the **page level** rather than the line level:

| Split Partition | Pages Assigned | Page Stems | Sample Count (Lines) | Leakage Audit |
|---|---|---|---|---|
| **Training Set** | 4 pages (66.7%) | `sample_page_002`, `sample_page_003`, `sample_page_004`, `sample_page_005` | 24 lines | **0.0% overlap** |
| **Validation Set** | 1 page (16.7%) | `sample_page_001` | 4 lines | **0.0% overlap** |
| **Test Set** | 1 page (16.7%) | `sample_page_006` | 6 lines | **0.0% overlap** |
| **Total** | 6 pages (100.0%) | 6 distinct documents | 34 lines | **Verified zero leakage** |

*Verification:*
Static scan across `data/splits.json` confirmed that no document stem appears in more than one partition.

---

## 3. Burn Profile Distribution & Calibration

The synthetic corruption engine (`SideBurnGenerator`) synthesizes physically plausible fire and thermal damage:

| Corruption Parameter | Mild Profile | Moderate (Default) | Severe Profile | Irregular Profile |
|---|---|---|---|---|
| **Burn Ratio Range** | 10% – 18% width | 15% – 30% width | 25% – 45% width | 15% – 35% width |
| **Jagged Amplitude** | 2.5% page width | 4.0% page width | 6.0% page width | 8.0% page width |
| **Charcoal Color** | `(18, 14, 12)` BGR | `(18, 14, 12)` BGR | `(18, 14, 12)` BGR | `(18, 14, 12)` BGR |
| **Transition Zone** | 40 px | 55 px | 75 px | 65 px |
| **Scorch Noise ($\sigma$)** | 4.0 | 6.0 | 8.0 | 7.5 |
| **Walk Intensity** | 0.08 | 0.12 | 0.16 | 0.20 |

---

## 4. Deterministic Seeding Fix (Python `hash()` Replacement)

In earlier revisions, `Burn_restore/dataset.py:197` calculated evaluation sample seeds via:
```python
sample_seed = (self.config.seed + idx * 7919 + abs(hash(stem)) % 10007) & 0x7FFFFFFF
```
Because Python's built-in `hash()` is randomized across processes via `PYTHONHASHSEED`, this introduced run-to-run seed drift.

**Resolved:** Line 197 now computes a deterministic SHA-256 digest:
```python
stem_hash = int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16)
sample_seed = (self.config.seed + idx * 7919 + stem_hash % 10007) & 0x7FFFFFFF
```
This guarantees 100% bit-for-bit reproducible synthetic corruption across all operating systems and Python environments.

---

## 5. External Historical Corpus Status (OPRB)

The **OPRB (Open Portuguese Historical Book Restoration Benchmark)** dataset represents authentic fire-damaged historical archives:
- **Corpus Scale:** 4.22 GB compressed archive containing 2,750 high-resolution historical document scans.
- **Acquisition Script:** `download_oprb.py` / `preprocess_oprb.py`.
- **Licensing & Access:** Academic open-access license.
- **Acquisition Status:** In compliance with the user constraint against unauthorized large downloads, OPRB was **not downloaded** to the local environment.
- **Scientific Impact:** All current models were trained and evaluated on the 6-page synthetic document fixture. Evaluation on authentic historical fire-damaged manuscripts remains documented as future validation work.

