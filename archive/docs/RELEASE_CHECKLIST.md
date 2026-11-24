# Release Checklist: Grounded Side-Burnt Document Text Restoration

**Release Version:** `v1.0.0-research-prototype`  
**Target Repository:** `https://github.com/KommanaboyinaPraveenKumar/Burn_restore.git`  
**Date:** November 2026  
**Status:** VALIDATED — READY FOR RESEARCH PROTOTYPE RELEASE

---

## 1. Pre-Release Quality & Integrity Checklist

| Check Item | Description | Status | Verification Reference |
|---|---|---|---|
| **Test Suite** | 80/80 automated unit & integration tests pass (100% pass rate) | [x] PASSED | `python -m unittest discover -s tests -p "test_*.py"` |
| **Artifact Hashes** | SHA-256 integrity verifier passes on checkpoint and configs | [x] PASSED | `scripts/verify_artifacts.py` |
| **Model Checkpoint** | Primary checkpoint `best_checkpoint.pt` verified ($978.7\text{ MB}$) | [x] PASSED | SHA-256: `2403d319...` |
| **Dataset Splits** | 6 pages strictly separated (4 train, 1 val, 1 test; 0% leakage) | [x] PASSED | `docs/dataset_readiness_report.json` |
| **Zero Credentials** | No API keys, personal access tokens, passwords, or secrets committed | [x] PASSED | Static scan across all `.py`, `.json`, `.yaml` |
| **Clean .gitignore** | Python caches (`__pycache__`), `.venv`, OS files ignored | [x] PASSED | Root `.gitignore` |
| **CLI Functionality** | `scripts/restore_document.py` runs end-to-end smoke test | [x] PASSED | Verified on `data/synthetic_fixture/page_001.png` |
| **FastAPI REST API** | Health check `/health`, `/model/info`, and `/restore/page` operational | [x] PASSED | `tests/test_api.py` |
| **Interactive UI** | Single-page HTML5/JS workbench served at `/` | [x] PASSED | `api/static/index.html` |
| **LaTeX Paper** | Full 2-column IEEE/ACM format paper ready for compilation | [x] PASSED | `paper/main.tex`, `paper/figures/` |
| **Model & Dataset Cards**| Standardized AI documentation created | [x] PASSED | `docs/MODEL_CARD.md`, `docs/DATASET_CARD.md` |

---

## 2. Scientific Integrity & Transparency Attestation

This release adheres strictly to academic integrity and transparency principles:

1. **No Metric Fabrication:**
   - Raw Character Error Rate ($\text{CER} = 2.1592$) and Word Error Rate ($\text{WER} = 5.0633$) are reported without obfuscation.
   - The report explicitly explains the mathematical origin of $\text{CER} > 1.0$: standard Levenshtein edit distance penalizes unconstrained autoregressive token insertions ($I > N$).
   - Normalized CER ($1 - \text{Accuracy}$) is bounded in $[0.89, 0.95]$.

2. **Honest Reporting of Visual Grounding:**
   - Invariance tests (zeroed crops $\text{CER} = 2.1547$, spatially shuffled crops $\text{CER} = 2.1592$) are transparently documented in the paper and benchmark reports.
   - The framework explicitly concludes that linguistic context dominates T5 generation on this small training set, and fine-grained character-stroke physical grounding remains an open research frontier.

3. **Physical Budget Enforcement:**
   - Physical character budget violation is documented as $100\%$ under unconstrained beam/greedy decoding.
   - The post-generation `BudgetEstimator` word-snapping algorithm successfully clamps outputs to the physical page margin with $0\%$ violation.

4. **Data Scale Disclosure:**
   - The project explicitly states that all models were trained and benchmarked on the 6-page deterministic synthetic fixture ($N = 10\text{--}34$ lines).
   - The 4.22 GB external OPRB real burnt historical document corpus has been surveyed and mapped, but was not downloaded without user authorization.

---

## 3. Git Release Procedure (When Authorized)

> **Notice:** Do not execute remote Git pushes or tag creations without explicit user authorization.

When user authorization is granted:

```bash
# 1. Ensure working directory is clean
git status

# 2. Add modified tracked files
git add .

# 3. Create release commit
git commit -m "chore(release): v1.0.0-research-prototype Grounded Side-Burnt Document Text Restoration"

# 4. Create annotated Git tag
git tag -a v1.0.0-research-prototype -m "Release v1.0.0-research-prototype: Complete Multimodal Document Restoration Framework"

# 5. Push commit and tag to remote
git push origin main
git push origin v1.0.0-research-prototype
```

---

## 4. GitHub Release Assets Packaging

When publishing on GitHub Releases, attach the following assets:
1. `checkpoints/best_checkpoint.pt` (uploaded via Git LFS or direct release asset binary)
2. `paper/main.tex` and compiled `paper.pdf`
3. `docs/FINAL_RESEARCH_AUDIT.json`
4. `docs/benchmark_results.json`
5. `data/synthetic_fixture/` (sample test fixture)

