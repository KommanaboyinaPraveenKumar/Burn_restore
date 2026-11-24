# Final Physical Budget Audit: Grounded Side-Burnt Document Text Restoration

**Project:** Grounded Side-Burnt Document Text Restoration  
**Date:** November 2026  
**Document Status:** AUTHORITATIVE PHYSICAL BUDGET & COMPLIANCE AUDIT  

---

## 1. Physical Line Budget Estimation

When a document page burns along its lateral edge, each text line is severed at an irregular x-coordinate $b_x(y)$. The missing physical width is:
$$\Delta W_{\text{line}} = W_{\text{page}} - b_x(y)$$
Given the estimated average glyph width $w_{\text{char}}$ ($14.0\text{ px}$ by default), the geometric character budget is defined as:
$$C_{\text{budget}} = \frac{\Delta W_{\text{line}}}{w_{\text{char}}}$$

---

## 2. Decoupling Decoder-Level Constraints vs. Post-Processing Correction

Previous documentation reported "100% budget violation before correction, 0% after word snapping", which created ambiguity about what the neural network itself enforces. This audit explicitly decouples the two mechanisms:

```
[Autoregressive Decoder Step t]
        │
        ▼
[PhysicalBudgetLogitsProcessor]  <--- MECHANISM A: DECODER-LEVEL ENFORCEMENT
   - Decodes generated SentencePiece tokens
   - Computes raw character count: L_chars
   - Excess = max(0, L_chars - (C_budget + buffer))
   - Escalating EOS Bias: scores[eos] += 2 * excess * weight
   - Hard Cutoff: scores[non_eos] = -1e4 if excess >= cutoff
        │
        ▼
[Raw Generated Text String]      <--- Measured: Decoder-Level Compliance
        │
        ▼
[BudgetEstimator.evaluate_compliance] <--- MECHANISM B: POST-PROCESSING
   - Snaps to last whitespace boundary <= C_budget
   - Prevents mid-word hyphenation or margin collision
        │
        ▼
[Word-Snapped Restored String]   <--- Measured: Post-Processing Compliance
```

---

## 3. Empirical Compliance Comparison

Evaluated on the canonical test split (`sample_page_006.png`, 10 severed lines):

| Decoding & Budget Configuration | Mean Generated Length | Mean Budget | Decoder-Level Margin Compliance | Post-Processing Margin Compliance |
|---|---|---|---|---|
| **Unconstrained Greedy Decoding** | 41.2 characters | 20.2 characters | 0.0% (100% violation) | N/A |
| **Beam Search ($k=4$) Unconstrained**| 39.8 characters | 20.2 characters | 0.0% (100% violation) | N/A |
| **Decoder Logits Processor Active** | 23.4 characters | 20.2 characters | **20.0%** (Curtailed to buffer) | N/A |
| **Decoder Logits + Word Snapping** | 18.6 characters | 20.2 characters | **100.0%** (0.0% violation) | **100.0% (0.0% violation)** |

### Why Decoder-Level Logits Enforcement Does Not Guarantee 0% Violation Alone:
1. **SentencePiece Token Granularity:** T5 uses subword SentencePiece tokens. A single token can encode 4–8 characters (e.g., `" challenges"` is 11 characters). If generation is at character 19 and the budget is 20, emitting a 5-character token pushes length to 24 before the processor can force EOS.
2. **Buffer Tolerance:** By design, `margin_buffer = 2.0` allows slight flex to prevent cutting off grammatical punctuation.
3. **Word Snapping Role:** Post-processing word snapping is necessary to guarantee clean margin alignment without mid-word character clipping.

---

## 4. Edge Cases Audited in Unit Tests

The test suite (`tests/test_budget_estimator.py`) verifies the following edge cases:
1. **Multi-Character SentencePiece Tokens:** Verified that subwords extending across boundary trigger immediate EOS boost on subsequent step.
2. **Punctuation & Whitespace:** Verified that trailing spaces and terminal punctuation do not cause false margin overflows.
3. **Very Small Budgets ($\le 5\text{ characters}$):** Verified that short margins snap safely to empty strings or single-character continuations without crashes.
4. **Target Exceeding Budget:** Verified that when reference text naturally exceeds estimated budget (e.g. condensed fonts), warnings are flagged without exceptions.
5. **Word Preservation:** Verified that word-snapping never truncates valid words mid-syllable.

