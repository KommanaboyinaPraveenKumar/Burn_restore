# Research Paper: Grounded Side-Burnt Document Text Restoration

This directory contains the complete LaTeX source, verified BibTeX bibliography, and publication-ready figures for the academic paper:

**"Grounded Side-Burnt Document Text Restoration: A Multimodal Framework for Context-Aware Recovery of Partially Damaged Text"**

---

## Directory Contents

- `main.tex`: Full research paper source in standard two-column conference layout.
- `references.bib`: Verified peer-reviewed bibliography (IEEE, CVPR, NeurIPS, JMLR, COLING, ICML) with real DOIs and URLs.
- `figures/`: High-resolution figures generated directly from verified project benchmark data:
  - `figure1_architecture.png`: Dual-stream architecture diagram (T5 + ResNet-34 + Gated Fusion).
  - `figure2_baselines_comparison.png`: Baseline comparison bar chart across Baselines A through G.
  - `figure3_ablation_study.png`: 10-component ablation matrix horizontal bar chart.
  - `figure4_burn_severity.png`: Error vs. burn corruption severity profiles.
  - `figure5_visual_stress_tests.png`: Visual modality grounding stress test chart.
  - `restoration_panel_*.png`: Multi-panel diagnostic contact sheets.

---

## Compilation Instructions

### 1. Using Overleaf (Recommended)
1. Zip the `paper/` folder:
   ```bash
   zip -r burn_restore_paper.zip paper/
   ```
2. Go to [Overleaf](https://www.overleaf.com), click **New Project** $\rightarrow$ **Upload Project**, and select `burn_restore_paper.zip`.
3. Set the compiler to **pdfLaTeX** and click **Recompile**.

### 2. Using Local TeX Live or MacTeX
Run the standard LaTeX compilation sequence from the `paper/` directory:
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### 3. Using latexmk (Automated)
```bash
cd paper/
latexmk -pdf main.tex
```

The resulting `main.pdf` document is publication-ready.

