"""
Tests for specificity-aware connectivity metrics (CCC + Dice/F1).

§21.8 noted Pearson r is permissive of scale bias and binary topology
mismatch. These tests pin:

  - ``connectivity_lin_ccc``: Lin's CCC on the upper-triangle entries
    of a connectivity matrix vs GT. Identical-matrices → 1.0; pure
    scale of one matrix → CCC drops while Pearson stays 1.
  - ``connectivity_dice_f1``: binary Dice + F1 at a count threshold.
    Identical-binary → 1.0; full-FP matrix vs sparse GT → low Dice.
"""

import numpy as np
import pytest


_helper = pytest.importorskip("validation.lib.connectivity_metrics")


# --------------------------------------------------------------------------- #
# 1. CCC on connectivity-matrix upper triangle
# --------------------------------------------------------------------------- #

class TestConnectivityLinCCC:
    def test_identical_matrices_give_unity(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, size=(16, 16))
        a = (a + a.T) / 2
        np.fill_diagonal(a, 0)
        assert abs(_helper.connectivity_lin_ccc(a, a) - 1.0) < 1e-9

    def test_scale_bias_drops_ccc_below_pearson(self):
        """CCC must drop when one matrix is a scaled copy of the other,
        even though Pearson r stays at 1.0. This is the failure mode
        §21.8 flagged: Pearson r=0.82 may hide a 2× scale bias."""
        from validation.lib.force_disco_connectivity import (
            connectivity_pearson,
        )
        rng = np.random.default_rng(1)
        gt = rng.uniform(0, 100, size=(16, 16))
        gt = (gt + gt.T) / 2
        np.fill_diagonal(gt, 0)
        pred = 5.0 * gt  # perfectly correlated but scaled 5×
        r = connectivity_pearson(pred, gt)
        ccc = _helper.connectivity_lin_ccc(pred, gt)
        assert abs(r - 1.0) < 1e-6, f"sanity: pearson should be 1, got {r}"
        # CCC should reflect the scale bias — drop well below 1
        assert ccc < 0.7, f"CCC should drop with 5× scale, got {ccc}"

    def test_zero_variance_gives_nan(self):
        a = np.zeros((16, 16))
        b = np.ones((16, 16))
        np.fill_diagonal(b, 0)
        result = _helper.connectivity_lin_ccc(a, b)
        assert np.isnan(result), f"expected nan, got {result}"

    def test_sumnorm_ccc_handles_unit_mismatch(self):
        """Comparing raw streamline counts to mm² GT areas: means differ
        by orders of magnitude → raw CCC collapses to ~0 while normalised
        CCC reports the actual concordance after putting both sides on
        the probability simplex.

        This is the DiSCo regime: GT mean ≈ 4e-4 mm², pred mean ≈ 130
        streamlines."""
        rng = np.random.default_rng(10)
        gt = rng.uniform(0, 3e-3, size=(16, 16))
        gt = (gt + gt.T) / 2
        np.fill_diagonal(gt, 0)
        # pred = perfectly proportional but scaled by 1e6
        pred = gt * 1.0e6
        ccc_raw = _helper.connectivity_lin_ccc(pred, gt)
        ccc_norm = _helper.connectivity_lin_ccc_normalised(pred, gt)
        # Raw CCC collapses (mean-diff term dominates)
        assert ccc_raw < 0.1, f"raw CCC should collapse, got {ccc_raw}"
        # Normalised CCC should be ~1 (pred is a pure scaling of GT)
        assert ccc_norm > 0.99, f"normalised CCC should be ~1, got {ccc_norm}"


# --------------------------------------------------------------------------- #
# 2. Dice + F1 on thresholded connectivity matrix
# --------------------------------------------------------------------------- #

class TestConnectivityDiceF1:
    def test_identical_binary_gives_unity(self):
        rng = np.random.default_rng(2)
        gt = rng.uniform(0, 10, size=(16, 16))
        gt = (gt + gt.T) / 2
        np.fill_diagonal(gt, 0)
        # threshold above which we call "connected"
        d = _helper.connectivity_dice_f1(gt, gt, threshold=0.0)
        assert abs(d["dice"] - 1.0) < 1e-9
        assert abs(d["f1"] - 1.0) < 1e-9
        # Dice == F1 for binary classification
        assert abs(d["dice"] - d["f1"]) < 1e-12

    def test_full_fp_vs_sparse_gt_drops_dice(self):
        """50 GT positives, 70 GT negatives. If pred says every pair is
        connected (the dmipy-JAX FP=70/70 regime), Dice should drop —
        precision goes to ~24/(24+70) ≈ 0.26 while recall stays at 1.0.
        Dice/F1 = 2·P·R/(P+R) ≈ 0.41."""
        # Synthetic GT: 16×16 symmetric, upper triangle has 50 ones / 70 zeros
        n = 16
        iu = np.triu_indices(n, k=1)
        rng = np.random.default_rng(3)
        gt_ut = np.zeros(len(iu[0]))
        idx = rng.choice(len(iu[0]), size=50, replace=False)
        gt_ut[idx] = 1.0
        gt = np.zeros((n, n))
        gt[iu] = gt_ut
        gt = gt + gt.T

        # pred: every off-diagonal pair has nonzero count
        pred = np.ones((n, n))
        np.fill_diagonal(pred, 0)

        d = _helper.connectivity_dice_f1(pred, gt, threshold=0.0)
        # P = TP/(TP+FP) = 50/120, R = 50/50 = 1
        # Dice = 2·(50/120)·1 / (50/120 + 1) = 100/170 ≈ 0.588
        expected = 2 * (50 / 120) / (50 / 120 + 1)
        assert abs(d["dice"] - expected) < 1e-9, f"got {d['dice']} expected {expected}"
        assert abs(d["recall"] - 1.0) < 1e-9
        assert abs(d["precision"] - 50 / 120) < 1e-9

    def test_threshold_sweep_monotone_precision(self):
        """At higher thresholds, a noisy prediction's precision should
        be ≥ the precision at lower thresholds (FP-pruning effect),
        unless TPs are pruned faster than FPs. Sanity check across
        a clear pred."""
        n = 16
        rng = np.random.default_rng(4)
        gt = (rng.uniform(0, 10, size=(n, n)) > 7).astype(float)
        gt = ((gt + gt.T) / 2 > 0).astype(float)
        np.fill_diagonal(gt, 0)
        # Pred: GT scaled up + uniform low-count FP noise everywhere
        pred = gt * 100 + rng.uniform(0, 5, size=(n, n))
        pred = (pred + pred.T) / 2
        np.fill_diagonal(pred, 0)
        d_lo = _helper.connectivity_dice_f1(pred, gt, threshold=0.0)
        d_hi = _helper.connectivity_dice_f1(pred, gt, threshold=10.0)
        assert d_hi["precision"] >= d_lo["precision"] - 1e-9


# --------------------------------------------------------------------------- #
# 3. Top-level summary across SNRs
# --------------------------------------------------------------------------- #

class TestSummarizeMethod:
    def test_returns_per_snr_dict(self):
        """``summarize_method`` accepts a dict of {snr: cmat} + GT and
        returns per-SNR Pearson + CCC + Dice/F1."""
        n = 16
        rng = np.random.default_rng(5)
        gt = rng.uniform(0, 10, size=(n, n))
        gt = (gt + gt.T) / 2
        np.fill_diagonal(gt, 0)
        cmats = {10: gt + rng.normal(0, 1, gt.shape),
                 30: gt + rng.normal(0, 0.5, gt.shape),
                 50: gt.copy()}
        for k in cmats:
            cmats[k] = (cmats[k] + cmats[k].T) / 2
            np.fill_diagonal(cmats[k], 0)
        out = _helper.summarize_method(cmats, gt, threshold=0.0)
        assert set(out.keys()) == {10, 30, 50}
        for snr in (10, 30, 50):
            for key in ("pearson_r", "lin_ccc", "lin_ccc_sumnorm",
                        "dice", "f1", "precision", "recall"):
                assert key in out[snr], f"missing {key} at SNR={snr}"
        # SNR=50 case is identical to GT → all metrics should be ~1
        assert abs(out[50]["pearson_r"] - 1.0) < 1e-6
        assert abs(out[50]["lin_ccc"] - 1.0) < 1e-6
        assert abs(out[50]["lin_ccc_sumnorm"] - 1.0) < 1e-6
        assert abs(out[50]["dice"] - 1.0) < 1e-6
