"""
metrics.py
==========
Evaluation helpers for the restoration pipeline. Image-quality metrics
(PSNR / SSIM) are computed defensively: mismatched shapes, degenerate crops,
or a missing optional dependency never crash the training/eval loop -- they
log a warning and return ``None`` so the caller can skip that sample.
"""

import warnings
from typing import List, Optional

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
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
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
# Text metrics
# ---------------------------------------------------------------------- #
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


def compute_cer(pred_text: str, target_text: str) -> Optional[float]:
    """Character Error Rate = edit_distance / max(1, len(target))."""
    try:
        if len(target_text) == 0:
            return None if len(pred_text) == 0 else 1.0
        dist = _levenshtein(pred_text, target_text)
        return dist / max(len(target_text), 1)
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(f"[metrics] CER computation failed gracefully: {exc}")
        return None


def compute_token_accuracy(pred_ids: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> Optional[float]:
    """Token‑level accuracy ignoring padding positions.
    Truncates both sequences to the shorter length, masks out padding tokens in the
    target, and returns the fraction of correct predictions as a float in [0, 1].
    Returns ``None`` if there are no non‑pad tokens.
    """
    try:
        # Align lengths
        min_len = min(pred_ids.shape[0], target_ids.shape[0])
        pred = pred_ids[:min_len]
        target = target_ids[:min_len]
        # Mask out padding tokens in the target
        mask = target != pad_id
        if mask.sum().item() == 0:
            return None
        correct = (pred == target) & mask
        # Compute proportion and ensure it does not exceed 1.0 due to numerical issues
        acc = correct.sum().float() / mask.sum().float()
        return float(min(acc.item(), 1.0))
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
