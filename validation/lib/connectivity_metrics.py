"""
Specificity-aware connectivity-matrix metrics.

§21 of doc 004 noted that Pearson r on raw streamline counts is
permissive of scale bias and binary topology mismatch — dmipy-JAX's
r=0.82 on DiSCo coexists with FP≈70/70 (every GT-zero pair receives
streamlines). This module provides:

- ``connectivity_lin_ccc``: Lin's CCC on upper-triangle entries.
  Penalises systematic bias that Pearson r does not.
- ``connectivity_dice_f1``: thresholded binary Dice + F1 + precision +
  recall on upper-triangle entries.
- ``summarize_method``: per-SNR roll-up across {Pearson, CCC, Dice,
  F1, precision, recall}.
"""

from __future__ import annotations

import numpy as np


def _upper_tri_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    iu = np.triu_indices(a.shape[0], k=1)
    return np.asarray(a)[iu].astype(np.float64), np.asarray(b)[iu].astype(np.float64)


def connectivity_lin_ccc(pred: np.ndarray, gt: np.ndarray) -> float:
    """Lin's CCC on upper-triangle entries of two connectivity matrices.

    CCC = 2·cov(p, g) / (var(p) + var(g) + (mean(p) − mean(g))²).
    Returns nan if either side has zero variance or fewer than 3 valid
    pairs.
    """
    p, g = _upper_tri_pairs(pred, gt)
    valid = np.isfinite(p) & np.isfinite(g)
    if valid.sum() < 3:
        return float("nan")
    p, g = p[valid], g[valid]
    var_p, var_g = p.var(ddof=0), g.var(ddof=0)
    if var_p < 1e-30 and var_g < 1e-30:
        return float("nan")
    cov = ((p - p.mean()) * (g - g.mean())).mean()
    denom = var_p + var_g + (p.mean() - g.mean()) ** 2
    if denom < 1e-30:
        return float("nan")
    return float(2.0 * cov / denom)


def connectivity_lin_ccc_normalised(
    pred: np.ndarray, gt: np.ndarray,
) -> float:
    """Lin's CCC after sum-normalising both matrices' upper triangle.

    Raw streamline counts (×10²) cannot be CCC-compared to a GT in
    different units (e.g., DiSCo's GT in mm² strand area, mean ≈ 0.0004).
    The (mean_p − mean_g)² term in the CCC denominator dominates,
    collapsing CCC to 0 regardless of actual agreement.

    Sum-normalising both sides to the probability simplex puts them on
    the same "fraction of total connectivity" scale and lets CCC report
    a meaningful concordance.
    """
    p, g = _upper_tri_pairs(pred, gt)
    valid = np.isfinite(p) & np.isfinite(g)
    p, g = p[valid], g[valid]
    if len(p) < 3 or p.sum() <= 0 or g.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    g = g / g.sum()
    var_p, var_g = p.var(ddof=0), g.var(ddof=0)
    if var_p < 1e-30 and var_g < 1e-30:
        return float("nan")
    cov = ((p - p.mean()) * (g - g.mean())).mean()
    denom = var_p + var_g + (p.mean() - g.mean()) ** 2
    if denom < 1e-30:
        return float("nan")
    return float(2.0 * cov / denom)


def connectivity_dice_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Binary Dice/F1 + precision + recall + TP/FP/FN/TN on upper-triangle.

    A pair is called "connected" if its count is strictly greater than
    ``threshold``. GT is binarised at threshold 0 (any positive entry
    is a true connection).
    """
    p, g = _upper_tri_pairs(pred, gt)
    pred_pos = p > threshold
    gt_pos = g > 0.0
    tp = int((pred_pos & gt_pos).sum())
    fp = int((pred_pos & ~gt_pos).sum())
    fn = int((~pred_pos & gt_pos).sum())
    tn = int((~pred_pos & ~gt_pos).sum())
    denom_dice = 2 * tp + fp + fn
    dice = (2 * tp / denom_dice) if denom_dice > 0 else float("nan")
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    f1 = dice  # equivalent for binary classification
    return {
        "dice": float(dice),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": float(threshold),
    }


def summarize_method(
    cmats: dict[int, np.ndarray],
    gt: np.ndarray,
    threshold: float = 0.0,
) -> dict[int, dict[str, float]]:
    """Roll up Pearson r + CCC + Dice/F1 across a {snr: cmat} dict."""
    from validation.lib.force_disco_connectivity import (
        connectivity_pearson,
    )
    out: dict[int, dict[str, float]] = {}
    for snr, cmat in cmats.items():
        r = connectivity_pearson(cmat, gt)
        ccc_raw = connectivity_lin_ccc(cmat, gt)
        ccc_norm = connectivity_lin_ccc_normalised(cmat, gt)
        d = connectivity_dice_f1(cmat, gt, threshold=threshold)
        out[snr] = {
            "pearson_r": float(r),
            "lin_ccc": float(ccc_raw),
            "lin_ccc_sumnorm": float(ccc_norm),
            **d,
        }
    return out
