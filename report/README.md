# Project report (LaTeX)

`main.tex` is the UROP project report in the department template's structure.

- **Compile:** pdfLaTeX, twice so the table of contents and cross-references fill in.
  On Overleaf, upload this folder and compile `main.tex`. Locally, run `latexmk -pdf main.tex`
  (MiKTeX or TeX Live).
- **Names:** set the student names, register numbers, guide and date in the macro block at the
  top of `main.tex`.
- **Logo:** put the university logo at `figures/srm_logo.png`. A labelled box is shown until then.
- **Figures:** `python scripts/make_report_figures.py` (CPU only) regenerates everything in
  `figures/` from the evaluation results.
- **Baseline section:** `baseline_results.tex` holds the OCR + LLM baseline results.
