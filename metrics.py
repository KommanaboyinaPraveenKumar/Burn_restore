"""
metrics.py
==========
Evaluation helpers for the restoration pipeline. Image-quality metrics
(PSNR / SSIM) are computed defensively: mismatched shapes, degenerate crops,
or a missing optional dependency never crash the training/eval loop -- they
log a warning and return ``None`` so the caller can skip that sample.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------- #
# Image-quality metrics
# ---------------------------------------------------------------------- #
def compute_psnr(img_a: np.ndarray, img_b: np.ndarray, max_val: float = 255.0) -> Optional[float]:
    """Peak Signal-to-Noise Ratio between two same-shaped uint8 images."""
    try:
        if img_a.shape != img_b.shape:
            warnings.warn(f"[metrics] PSNR shape mismatch {img_a.shape} vs {img_b.shape}; skipping.")
            return None
        mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
        if mse == 0:
            return float("inf")
        return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(f"[metrics] PSNR computation failed gracefully: {exc}")
        return None


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> Optional[float]:
    """Structural Similarity Index. Falls back gracefully if scikit-image
    is unavailable or inputs are degenerate (too small, mismatched, etc.)."""
    try:
        import importlib
        skimage_metrics = importlib.import_module("skimage.metrics")
        ssim = getattr(skimage_metrics, "structural_similarity")
    except Exception:
        warnings.warn("[metrics] scikit-image not installed; SSIM skipped.")
        return None
    try:
        if img_a.shape != img_b.shape:
            warnings.warn(f"[metrics] SSIM shape mismatch {img_a.shape} vs {img_b.shape}; skipping.")
            return None
        gray_a = img_a.mean(axis=2) if img_a.ndim == 3 else img_a
        gray_b = img_b.mean(axis=2) if img_b.ndim == 3 else img_b
        min_dim = min(gray_a.shape)
        if min_dim < 7:
            warnings.warn("[metrics] Image too small for default SSIM window; skipping.")
            return None
        score, _ = ssim(gray_a, gray_b, full=True, data_range=255.0)
        return float(score)
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(f"[metrics] SSIM computation failed gracefully: {exc}")
        return None


# ---------------------------------------------------------------------- #
# Text normalization & helpers
# ---------------------------------------------------------------------- #
import unicodedata

def normalize_text(text: str) -> str:
    """Decomposes LaTeX/PDF ligatures (fi, fl, ffi) and normalizes curly quotes,
    dashes, and typographical variants so edit distance measures true semantic content
    rather than unicode encoding differences."""
    if not text:
        return ""
    # Unicode NFKD decomposition: \ufb01 ('fi') -> 'f' + 'i'
    s = unicodedata.normalize('NFKD', text)
    # Typography normalization
    s = s.replace('\u2019', "'").replace('\u2018', "'")
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    s = s.replace('\u2014', "-").replace('\u2013', "-").replace('\u2212', "-")
    # Collapse multiple whitespaces
    s = " ".join(s.split())
    return s.strip()


def _levenshtein(a: str, b: str) -> int:
    """Classic O(len(a)*len(b)) edit-distance dynamic program (no external
    dependency required)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def _levenshtein_tokens(a: List[str], b: List[str]) -> int:
    """Token-level / word-level Levenshtein edit distance."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


# ---------------------------------------------------------------------- #
# Text metrics (CER, WER, Token Accuracy, Character & Word Accuracy)
# ---------------------------------------------------------------------- #
def compute_cer(pred_text: str, target_text: str) -> Optional[float]:
    """Character Error Rate = edit_distance / max(1, len(target))."""
    try:
        if len(target_text) == 0:
            return 0.0 if len(pred_text) == 0 else 1.0
        dist = _levenshtein(pred_text, target_text)
        return dist / max(len(target_text), 1)
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(f"[metrics] CER computation failed gracefully: {exc}")
        return None


def compute_wer(pred_text: str, target_text: str) -> Optional[float]:
    """Word Error Rate = edit_distance(pred_words, target_words) / max(1, len(target_words)).
    Strictly accounts for insertions, deletions, and substitutions at word level."""
    try:
        norm_pred = normalize_text(pred_text)
        norm_target = normalize_text(target_text)
        p_words = norm_pred.split()
        t_words = norm_target.split()
        if len(t_words) == 0:
            return 0.0 if len(p_words) == 0 else 1.0
        dist = _levenshtein_tokens(p_words, t_words)
        return float(dist / len(t_words))
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(f"[metrics] WER computation failed gracefully: {exc}")
        return None


def compute_character_accuracy(pred_text: str, target_text: str) -> float:
    """Edit-distance-derived character accuracy: 1.0 - CER, clamped to [0.0, 1.0].
    Strictly distinct from position-based character accuracy."""
    norm_pred = normalize_text(pred_text)
    norm_target = normalize_text(target_text)
    if not norm_target:
        return 1.0 if not norm_pred else 0.0
    cer = compute_cer(norm_pred, norm_target)
    return float(max(0.0, 1.0 - (cer if cer is not None else 1.0)))


def compute_position_character_accuracy(pred_text: str, target_text: str) -> float:
    """Position-based character accuracy: computes exact character matches at identical
    indices divided by max(len(pred), len(target)). Penalizes misalignment shifts.
    Never interchangeable with Levenshtein-derived character accuracy."""
    norm_p = normalize_text(pred_text)
    norm_t = normalize_text(target_text)
    max_len = max(len(norm_p), len(norm_t))
    if max_len == 0:
        return 1.0
    min_len = min(len(norm_p), len(norm_t))
    matches = sum(1 for i in range(min_len) if norm_p[i] == norm_t[i])
    return float(matches / max_len)


def compute_detailed_levenshtein_rates(pred_text: str, target_text: str) -> Dict[str, float]:
    """Computes exact counts and rates of insertions, deletions, and substitutions
    relative to target length:
      insertion_rate = insertions / max(1, len(target))
      deletion_rate = deletions / max(1, len(target))
      substitution_rate = substitutions / max(1, len(target))
    """
    p_norm = normalize_text(pred_text)
    t_norm = normalize_text(target_text)
    n, m = len(p_norm), len(t_norm)

    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if p_norm[i - 1] == t_norm[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)

    i, j = n, m
    insertions = 0
    deletions = 0
    substitutions = 0
    matches = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] and p_norm[i - 1] == t_norm[j - 1]:
            matches += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            insertions += 1
            i -= 1
        else:
            deletions += 1
            j -= 1

    denom = max(1, m)
    return {
        "matches": matches,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "total_edits": insertions + deletions + substitutions,
        "insertion_rate": round(float(insertions / denom), 4),
        "deletion_rate": round(float(deletions / denom), 4),
        "substitution_rate": round(float(substitutions / denom), 4),
    }


def compute_physical_grounding_metrics(
    pred_text: str,
    target_text: str,
    budget: float,
    buffer: float = 2.0,
) -> Dict[str, Any]:
    """Computes physical budget compliance, excess characters, and over/under generation."""
    pred_len = len(normalize_text(pred_text))
    targ_len = len(normalize_text(target_text))

    excess = max(0.0, float(pred_len - (budget + buffer)))
    is_violation = excess > 0.0

    len_diff = pred_len - targ_len
    over_gen = len_diff > 2
    under_gen = len_diff < -2

    # Visually supported character estimate (character matches supported by visual crop length)
    rates = compute_detailed_levenshtein_rates(pred_text, target_text)
    vis_supported_rate = round(float(rates["matches"] / max(1, targ_len)), 4)

    return {
        "budget_violation": is_violation,
        "budget_excess_chars": round(excess, 2),
        "over_generated": over_gen,
        "under_generated": under_gen,
        "length_difference": len_diff,
        "visually_supported_rate": vis_supported_rate,
        "eos_compliant": "<extra_id" not in pred_text and "<pad>" not in pred_text,
    }


def compute_region_bounded_psnr_ssim(
    clean_img: np.ndarray,
    restored_img: np.ndarray,
    burn_x_start: int,
) -> Dict[str, Optional[float]]:
    """Calculates PSNR and SSIM bounded strictly to the restoration region (x >= burn_x_start).
    Ensures undamaged document content does not artificially inflate image restoration scores."""
    if clean_img.shape != restored_img.shape:
        return {"bounded_psnr": None, "bounded_ssim": None}

    w = clean_img.shape[1]
    x0 = max(0, min(burn_x_start, w - 10))
    region_clean = clean_img[:, x0:]
    region_restored = restored_img[:, x0:]

    psnr = compute_psnr(region_clean, region_restored)
    ssim = compute_ssim(region_clean, region_restored)
    return {"bounded_psnr": psnr, "bounded_ssim": ssim}


def compute_word_accuracy(pred_text: str, target_text: str) -> float:
    """Word-level accuracy: 1.0 - WER, clamped to [0.0, 1.0]."""
    wer = compute_wer(pred_text, target_text)
    return float(max(0.0, 1.0 - (wer if wer is not None else 1.0)))


def compute_exact_match_line_accuracy(pred_text: str, target_text: str) -> float:
    """Exact-match line accuracy: 1.0 if normalized strings match identically, else 0.0."""
    norm_pred = normalize_text(pred_text).strip().lower()
    norm_target = normalize_text(target_text).strip().lower()
    return 1.0 if norm_pred == norm_target else 0.0


def compute_exact_match_document_accuracy(pred_lines: List[str], target_lines: List[str]) -> float:
    """Exact-match document accuracy: 1.0 if all lines match identically, else 0.0."""
    if len(pred_lines) != len(target_lines) or len(target_lines) == 0:
        return 0.0
    for p, t in zip(pred_lines, target_lines):
        if normalize_text(p).strip().lower() != normalize_text(t).strip().lower():
            return 0.0
    return 1.0


def compute_token_accuracy(pred_ids: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> Optional[float]:
    """Token-level accuracy evaluating against the entire non-padding target sequence.
    
    Fixes the Phase 0 min_len truncation flaw:
    Previously, sequences were sliced to min_len, so predicting only the first token
    of a 15-token target returned 100% accuracy.
    
    Now:
      1. Strips pad tokens from both sequences.
      2. Compares token-by-token.
      3. Denominator is max(len(target_tokens), len(pred_tokens)).
      4. Missing target tokens and extra inserted tokens are strictly penalized.
    Returns float in [0.0, 1.0], or None if target is empty.
    """
    try:
        # Handle 1D or 2D inputs
        if pred_ids.dim() > 1:
            pred_ids = pred_ids.squeeze()
        if target_ids.dim() > 1:
            target_ids = target_ids.squeeze()

        # Extract non-pad tokens
        target_tokens = target_ids[target_ids != pad_id]
        pred_tokens = pred_ids[pred_ids != pad_id]

        target_len = target_tokens.size(0)
        pred_len = pred_tokens.size(0)

        if target_len == 0:
            return 1.0 if pred_len == 0 else 0.0

        min_len = min(pred_len, target_len)
        max_len = max(pred_len, target_len)

        if min_len == 0:
            return 0.0

        matches = (pred_tokens[:min_len] == target_tokens[:min_len]).sum().float().item()
        acc = matches / float(max_len)
        return float(min(max(acc, 0.0), 1.0))
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(f"[metrics] Token accuracy computation failed gracefully: {exc}")
        return None

# Additional text similarity metrics for research reporting

def compute_normalized_levenshtein(pred_text: str, target_text: str) -> Optional[float]:
    """Return similarity = 1 - (levenshtein distance / max length). Returns None if both strings empty."""
    try:
        if not pred_text and not target_text:
            return None
        lev = _levenshtein(pred_text, target_text)
        max_len = max(len(pred_text), len(target_text))
        return 1.0 - lev / max_len if max_len > 0 else None
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(f"[metrics] Normalized Levenshtein computation failed: {exc}")
        return None

# Standalone pure-Python BLEU calculation with smoothing
def compute_bleu(pred_text: str, target_text: str, max_n: int = 4) -> Optional[float]:
    """Compute sentence-level BLEU with smoothing in pure Python (no NLTK dependency)."""
    try:
        ref_tokens = target_text.strip().split()
        hyp_tokens = pred_text.strip().split()
        if not ref_tokens or not hyp_tokens:
            return 0.0

        # Brevity penalty
        ref_len = len(ref_tokens)
        hyp_len = len(hyp_tokens)
        if hyp_len == 0:
            return 0.0
        bp = 1.0 if hyp_len > ref_len else np.exp(1.0 - ref_len / hyp_len)

        # Modified n-gram precision with +1 smoothing (Lin & Och / Chen & Cherry)
        precisions = []
        for n in range(1, max_n + 1):
            if hyp_len < n:
                precisions.append(1.0 / (hyp_len + 1))
                continue
            hyp_ngrams = {}
            for i in range(hyp_len - n + 1):
                ngram = tuple(hyp_tokens[i:i + n])
                hyp_ngrams[ngram] = hyp_ngrams.get(ngram, 0) + 1

            ref_ngrams = {}
            for i in range(len(ref_tokens) - n + 1):
                ngram = tuple(ref_tokens[i:i + n])
                ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1

            clipped = sum(min(count, ref_ngrams.get(ngram, 0)) for ngram, count in hyp_ngrams.items())
            total = max(1, sum(hyp_ngrams.values()))
            # Smoothing: add 1 to clipped, add 1 to total for n > 1 if clipped == 0
            prec = (clipped + 0.1) / (total + 0.1)
            precisions.append(prec)

        log_precision = sum((1.0 / max_n) * np.log(p) for p in precisions)
        return float(bp * np.exp(log_precision))
    except Exception as exc:  # defensive guard
        warnings.warn(f"[metrics] BLEU computation failed: {exc}")
        return 0.0

# Simple ROUGE-L implementation (longest common subsequence)
def _lcs(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[m]

def compute_rouge_l(pred_text: str, target_text: str) -> Optional[float]:
    """ROUGE-L F1 score based on longest common subsequence."""
    try:
        lcs_len = _lcs(pred_text, target_text)
        if lcs_len == 0:
            return 0.0
        prec = lcs_len / len(pred_text) if pred_text else 0.0
        rec = lcs_len / len(target_text) if target_text else 0.0
        if prec + rec == 0:
            return 0.0
        return (2 * prec * rec) / (prec + rec)
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(f"[metrics] ROUGE-L computation failed: {exc}")
        return None


def extract_infill_text(raw_decoded: str) -> str:
    """Extracts restored text generated between <extra_id_0> and <extra_id_1> or EOS.
    Falls back gracefully if raw text has no special span tokens."""
    text = raw_decoded
    if "<extra_id_0>" in text:
        text = text.split("<extra_id_0>", 1)[1]
    if "<extra_id_1>" in text:
        text = text.split("<extra_id_1>", 1)[0]
    for i in range(2, 10):
        if f"<extra_id_{i}>" in text:
            text = text.split(f"<extra_id_{i}>", 1)[0]
    if "| prefix" in text:
        text = text.replace("| prefix", "")
    return text.strip()


# ---------------------------------------------------------------------- #
# Literature-Aligned Multi-Candidate Metrics & UCSM
# (Ref: Google DeepMind Ithaca & CVPR 2024 DocRevive UCSM)
# ---------------------------------------------------------------------- #


def compute_normalized_cer(pred_text: str, target_text: str) -> Optional[float]:
    """CER evaluated on ligature-normalized and typographically clean text strings."""
    norm_pred = normalize_text(pred_text)
    norm_target = normalize_text(target_text)
    return compute_cer(norm_pred, norm_target)


def compute_topk_accuracy(candidates: List[str], target_text: str, k: int = 1) -> float:
    """Computes Top-K exact match accuracy across beam candidates (Ithaca / DeepMind paradigm).
    Returns 1.0 if target is within the top-k candidates, else 0.0."""
    norm_target = normalize_text(target_text).lower()
    if not norm_target:
        return 0.0
    for cand in candidates[:k]:
        norm_cand = normalize_text(cand).lower()
        if norm_cand == norm_target or (len(norm_target) > 2 and norm_target in norm_cand):
            return 1.0
    return 0.0


def compute_ucsm(
    pred_text: str,
    target_text: str,
    bleu: float = 0.0,
    rouge: float = 0.0
) -> float:
    """Unified Context Similarity Metric (DocRevive, CVPR 2024).
    Balances character edit similarity, length alignment, and linguistic overlap:
      UCSM = 0.35 * Sim_edit + 0.35 * ROUGE_L + 0.15 * BLEU + 0.15 * Sim_len
    Range: [0.0, 1.0] where 1.0 is a perfect reconstruction.
    """
    norm_pred = normalize_text(pred_text)
    norm_target = normalize_text(target_text)
    
    # 1. Edit distance similarity
    cer = compute_cer(norm_pred, norm_target)
    sim_edit = max(0.0, 1.0 - (cer if cer is not None else 1.0))
    
    # 2. Length similarity
    l_p = len(norm_pred)
    l_t = len(norm_target)
    max_l = max(l_p, l_t, 1)
    sim_len = max(0.0, 1.0 - abs(l_p - l_t) / float(max_l))
    
    # 3. Composite UCSM
    ucsm = 0.35 * sim_edit + 0.35 * rouge + 0.15 * bleu + 0.15 * sim_len
    return float(np.clip(ucsm, 0.0, 1.0))


# ---------------------------------------------------------------------- #
# Canonical Corpus-Level Evaluation Protocol
# ---------------------------------------------------------------------- #
def compute_corpus_evaluation(
    predictions: List[str],
    targets: List[str],
    budgets: Optional[List[float]] = None,
    margin_buffer: float = 2.0,
) -> Dict[str, Any]:
    """Authoritative, canonical corpus evaluation protocol for the benchmark.
    
    Computes both:
      1. Micro-averaged corpus metrics:
         Corpus CER = sum(edit_dist_char) / sum(target_chars)
         Corpus WER = sum(edit_dist_word) / sum(target_words)
      2. Macro-averaged sample metrics:
         Macro CER = mean(sample_cer)
         Macro WER = mean(sample_wer)
      3. Bounded accuracy, exact match, BLEU-4, ROUGE-L, and independent
         decoder-level vs post-processed budget violation rates.
    """
    assert len(predictions) == len(targets), (
        f"Length mismatch: {len(predictions)} predictions vs {len(targets)} targets."
    )
    n = len(targets)
    if n == 0:
        return {
            "sample_count": 0,
            "corpus_cer": 0.0,
            "corpus_wer": 0.0,
            "macro_cer": 0.0,
            "macro_wer": 0.0,
            "char_accuracy": 0.0,
            "exact_match_rate": 0.0,
            "bleu_4": 0.0,
            "rouge_l": 0.0,
            "decoder_budget_violation_rate": 0.0,
        }

    total_char_dist = 0
    total_target_chars = 0
    total_word_dist = 0
    total_target_words = 0

    sample_cers: List[float] = []
    sample_wers: List[float] = []
    char_accuracies: List[float] = []
    word_accuracies: List[float] = []
    exact_matches: List[float] = []
    bleu_scores: List[float] = []
    rouge_scores: List[float] = []
    decoder_violations: int = 0
    word_snapped_violations: int = 0

    for i in range(n):
        pred = predictions[i]
        targ = targets[i]
        norm_p = normalize_text(pred)
        norm_t = normalize_text(targ)

        # Character-level
        c_dist = _levenshtein(norm_p, norm_t)
        c_denom = len(norm_t)
        total_char_dist += c_dist
        total_target_chars += c_denom

        s_cer = float(c_dist / max(1, c_denom)) if c_denom > 0 else (0.0 if len(norm_p) == 0 else 1.0)
        sample_cers.append(s_cer)
        char_accuracies.append(max(0.0, 1.0 - min(1.0, s_cer)))

        # Word-level
        p_words = norm_p.split()
        t_words = norm_t.split()
        w_dist = _levenshtein_tokens(p_words, t_words)
        w_denom = len(t_words)
        total_word_dist += w_dist
        total_target_words += w_denom

        s_wer = float(w_dist / max(1, w_denom)) if w_denom > 0 else (0.0 if len(p_words) == 0 else 1.0)
        sample_wers.append(s_wer)
        word_accuracies.append(max(0.0, 1.0 - min(1.0, s_wer)))

        # Exact match
        em = 1.0 if norm_p.lower() == norm_t.lower() else 0.0
        exact_matches.append(em)

        # BLEU & ROUGE
        bleu = compute_bleu(norm_p, norm_t) or 0.0
        rouge = compute_rouge_l(norm_p, norm_t) or 0.0
        bleu_scores.append(bleu)
        rouge_scores.append(rouge)

        # Physical budget evaluation
        if budgets is not None and i < len(budgets) and budgets[i] is not None:
            b = float(budgets[i])
            if b > 0:
                # Decoder-level raw length check
                if len(norm_p) > (b + margin_buffer):
                    decoder_violations += 1

                # Word-snapped length check
                # Simulate snapping to nearest whitespace within budget
                if len(norm_p) <= (b + margin_buffer):
                    snapped = norm_p
                else:
                    truncated = norm_p[: int(b)]
                    last_space = truncated.rfind(" ")
                    snapped = truncated[:last_space].strip() if last_space > 0 else truncated.strip()

                if len(snapped) > (b + margin_buffer):
                    word_snapped_violations += 1

    corpus_cer = float(total_char_dist / max(1, total_target_chars))
    corpus_wer = float(total_word_dist / max(1, total_target_words))

    evaluated_budgets = n if (budgets is not None and len(budgets) > 0) else 0

    return {
        "sample_count": n,
        "total_target_characters": total_target_chars,
        "total_target_words": total_target_words,
        "total_char_edit_distance": total_char_dist,
        "total_word_edit_distance": total_word_dist,
        "corpus_cer": round(corpus_cer, 4),
        "corpus_wer": round(corpus_wer, 4),
        "macro_cer": round(float(np.mean(sample_cers)), 4),
        "macro_wer": round(float(np.mean(sample_wers)), 4),
        "char_accuracy_bounded": round(float(np.mean(char_accuracies)), 4),
        "word_accuracy_bounded": round(float(np.mean(word_accuracies)), 4),
        "exact_match_rate": round(float(np.mean(exact_matches)), 4),
        "mean_bleu_4": round(float(np.mean(bleu_scores)), 4),
        "mean_rouge_l": round(float(np.mean(rouge_scores)), 4),
        "decoder_budget_violation_rate": round(float(decoder_violations / max(1, evaluated_budgets)), 4) if evaluated_budgets else 0.0,
        "word_snapped_budget_violation_rate": round(float(word_snapped_violations / max(1, evaluated_budgets)), 4) if evaluated_budgets else 0.0,
    }



