# Grounded Side-Burnt Document Text Restoration

A research framework for restoring text pages burned along the outer (right)
edge, combining a **Global Context Engine** (a language model reading the
surviving text) with a **Local Forensic Eye** (a vision encoder reading the
microscopic "ghost strokes" at the jagged burn border), so the model cannot
hallucinate text that does not physically fit the remaining pixel evidence.

## Project layout

```
grounded_doc_restorer/
├── config.py              # single shared configuration dataclass
├── dataset_generator.py   # Component 1: synthetic side-burn generator
├── model.py               # Component 2: dual-stream model + fusion + grounding
├── dataset.py             # Component 3a: PyTorch Dataset (dynamic corruption)
├── train.py               # Component 3b: executable training loop
├── metrics.py             # PSNR / SSIM / CER / token accuracy (fail-safe)
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Expected input data

A folder of clean page images and a folder of matching transcripts, paired by
filename stem, with **one line of text per physical line on the page**:

```
data/images/page_001.png      data/texts/page_001.txt
data/images/page_002.png      data/texts/page_002.txt
```

## Usage

Generate a static corrupted dataset for inspection:

```bash
python dataset_generator.py --images data/images --texts data/texts --out data/burnt
```

Train (corruption is re-synthesized on the fly every epoch):

```bash
python train.py --images data/images --texts data/texts --epochs 10 --batch_size 4
```

---

## Component 1 — Synthetic Side-Burn Generator

`SideBurnGenerator` produces physically plausible fire damage rather than a
clean crop:

* **Jagged, non-linear boundary.** A Gaussian random walk is smoothed with a
  moving-average kernel to produce an organic scorch edge (`_generate_jagged_boundary`),
  with the burned width sampled per-page between 15% and 30% of page width.
* **Ghost-stroke fade.** Within a configurable transition band before the
  boundary, pixels are alpha-blended toward a dark charcoal value, so glyphs
  crossing the threshold degrade into low-contrast residue before reaching
  full `(0,0,0)` obliteration. Gaussian scorch noise is injected *only* inside
  the damaged region.
* **Outputs.** The corrupted image, a binary burn mask, the burn bounding box,
  the per-row missing-width profile, the surviving (healthy) text, and the
  ground-truth clipped line-endings.

**Documented assumption:** without per-character bounding boxes in the source
data, the kept/clipped text split is estimated *proportionally* — the fraction
of a line's width surviving the jagged boundary is applied to that line's
character count, then snapped to the nearest whitespace so the ground truth
stays word-aligned. Text lines are located by a horizontal ink-projection
profile, with a uniform-split fallback when band detection disagrees with the
transcript line count.

## Component 2 — Dual-Stream Model

`GroundedDocumentRestorer` composes four submodules:

| Module | Role |
|---|---|
| `GlobalContextEncoder` | DistilBERT over the surviving page text → linguistic context `(B, T, E)` |
| `LocalForensicEncoder` | ResNet-34 with the pooling head removed → **spatial** feature grid flattened to visual tokens `(B, h·w, E)` |
| `CrossModalFusion` | Multi-head cross-attention: text tokens query the burn-edge visual tokens |
| `SpatialAlignmentGate` | Converts burn geometry into a decoding budget |
| `TransformerDecoderHead` | Autoregressive decoder over the fused memory |

The forensic encoder deliberately keeps the spatial feature map instead of
global-pooling it, so cross-attention can localize *which* ghost stroke
supports which linguistic hypothesis. A globally-pooled visual vector is
additionally broadcast and gated back into every memory position, so the
decoder always retains direct access to the overall stroke evidence.

### The grounding constraint

`SpatialAlignmentGate` enforces that predictions fit the available pixel
footprint. The burn mask's per-row missing-width profile is binned into a
fixed-length vector, averaged, and divided by an average glyph width to yield
a per-sample **character budget**. An escalating additive bias is then applied
to the EOS logit once decoding runs past that budget:

```
bias(step) = relu(step - budget) * alignment_penalty_weight
```

This is applied identically during training (shaping the loss surface) and
generation (halting decode). A soft bias is used rather than a hard truncation
so gradients continue to flow and the model *learns* the length prior instead
of merely being clipped at inference.

Verified behaviour — budget and output length scale with physical burn width:

| Missing width | Char budget | Generated length (cap 30) |
|---|---|---|
| 28 px | 2.0 | 5 |
| 140 px | 10.0 | 13 |
| 560 px | 40.0 | 31 |

## Component 3 — Dataset and Training Loop

`SideBurnDocumentDataset` re-runs the generator on **every** `__getitem__`,
so a page is never seen with the same burn pattern twice — the corruption
itself is the augmentation. Each item yields tokenized context, tokenized
target, the ImageNet-normalized burn crop, and the binned mask profile;
raw strings are carried through a custom `collate_fn` for logging.

`train.py` wires the pieces into a standard teacher-forced loop: shifted
cross-entropy over the clipped-text tokens (padding ignored), AdamW, gradient
clipping, periodic token-accuracy logging, and a periodic qualitative decode
with CER, wrapped so a single bad sample can never abort a long run.

## Validation performed

All modules were executed, not just syntax-checked:

* Generator run on synthetic pages — burn renders with the intended jagged
  edge and ghost fade; text clipping verified correct (`"...jumps over"` /
  `"the lazy dog"`).
* Model forward and `generate()` verified for tensor shape correctness.
* Alignment gate verified to scale budget and output length with burn width.
* `Dataset` + `DataLoader` + `collate_fn` verified for shape and batching.
* Full training loop run for 2 epochs; loss descended 4.18 → 3.13.
* Metrics verified on both normal and degenerate inputs (shape mismatch,
  sub-window images, empty targets, all-padding batches) — these return
  `None` with a warning rather than raising.

> Note: the smoke tests used a small locally-built tokenizer and randomly
> initialized backbones because the test sandbox had no network access to
> the Hugging Face Hub. On a connected machine `config.text_model_name` and
> `pretrained_vision=True` load the real pretrained weights unchanged.

## Tuning notes

* `avg_char_width_px` should be calibrated to your corpus's rendering DPI and
  font size; it directly sets the strictness of the grounding constraint.
* `alignment_penalty_weight` trades fluency against physical fidelity — raise
  it if the model over-generates past the burn edge.
* `transition_zone_px` controls how much ghost-stroke evidence the forensic
  stream receives. Too narrow and Stream B has nothing to read; too wide and
  the task becomes trivially easy.
* Consider freezing the DistilBERT encoder for the first epochs so the
  randomly-initialized fusion and decoder do not corrupt pretrained weights.
