"""
Tests for the FORCE library fibre-count rebalancer.

Audit recommendation #4 (doc 004 §17.7): the default library is
~10/20/70 % over (1f, 2f, 3f) due to `Dirichlet(2,1,1)` over fibre
fractions. For DiSCo (single-strand-dominated), this drives
false-positive multi-fibre tracking. Rebalancing to a single-fibre-
favouring distribution (e.g. 80/10/10) should reduce the FP rate.

These tests pin the rebalancer's contracts.
"""

import numpy as np
import pytest


rebalance = pytest.importorskip("validation.lib.force_library_rebalance")


def _make_fake_sims(n_per_fibres: dict) -> dict:
    """Synthesise a tiny FORCE-like sims dict for testing.

    n_per_fibres maps {1: n1, 2: n2, 3: n3} → total N entries with
    that fibre-count distribution.
    """
    fcs = []
    for nf, n in n_per_fibres.items():
        fcs.extend([float(nf)] * n)
    N = len(fcs)
    rng = np.random.default_rng(0)
    return {
        "signals": rng.normal(size=(N, 50)).astype(np.float32),
        "num_fibers": np.array(fcs, dtype=np.float32),
        "labels": np.zeros((N, 362), dtype=np.uint8),
        "odfs": rng.normal(size=(N, 362)).astype(np.float32),
        "dispersion": rng.uniform(0.01, 0.3, size=N).astype(np.float32),
        "wm_fraction": rng.uniform(0, 1, size=N).astype(np.float32),
        "gm_fraction": rng.uniform(0, 1, size=N).astype(np.float32),
        "csf_fraction": rng.uniform(0, 1, size=N).astype(np.float32),
        "nd": rng.uniform(0, 1, size=N).astype(np.float32),
        "fa": rng.uniform(0, 1, size=N).astype(np.float32),
        "md": rng.uniform(0, 3e-3, size=N).astype(np.float32),
        "rd": rng.uniform(0, 3e-3, size=N).astype(np.float32),
        "ufa_wm": rng.uniform(0, 1, size=N).astype(np.float32),
        "ufa_voxel": rng.uniform(0, 1, size=N).astype(np.float32),
        "fraction_array": rng.uniform(0, 1, size=(N, 3)).astype(np.float32),
    }


class TestRebalanceLibrary:
    def test_target_distribution_is_met_within_5pct(self):
        sims = _make_fake_sims({1: 1000, 2: 2000, 3: 7000})  # 10/20/70
        target = {1: 0.80, 2: 0.10, 3: 0.10}
        out = rebalance.rebalance_force_library(sims, target_fractions=target, seed=42)
        nf = out["num_fibers"]
        N = len(nf)
        for k, p in target.items():
            actual = (nf == k).sum() / N
            assert abs(actual - p) < 0.05, (
                f"n_fibers={k}: target {p:.2f}, actual {actual:.3f}"
            )

    def test_preserves_all_keys_with_consistent_shapes(self):
        sims = _make_fake_sims({1: 100, 2: 200, 3: 700})
        target = {1: 0.7, 2: 0.2, 3: 0.1}
        out = rebalance.rebalance_force_library(sims, target_fractions=target)
        N = len(out["num_fibers"])
        for k, v in sims.items():
            assert k in out
            assert out[k].shape[0] == N, (
                f"{k}: shape[0] {out[k].shape[0]} != {N}"
            )
            assert out[k].shape[1:] == v.shape[1:]

    def test_capped_when_target_exceeds_available(self):
        """If we ask for more 1-fibre entries than exist, the rebalancer
        caps at all-available rather than failing or oversampling."""
        sims = _make_fake_sims({1: 50, 2: 200, 3: 700})  # only 50 1-fibre
        target = {1: 0.8, 2: 0.1, 3: 0.1}
        out = rebalance.rebalance_force_library(sims, target_fractions=target)
        # All 50 1-fibre entries should be present
        assert (out["num_fibers"] == 1).sum() == 50
        # 2-fibre and 3-fibre should still be ~1/8 of total each
        N = len(out["num_fibers"])
        n2 = (out["num_fibers"] == 2).sum()
        n3 = (out["num_fibers"] == 3).sum()
        # Total = 50 + 50/0.8 * 0.1 + 50/0.8 * 0.1 = 50 + 6.25 + 6.25 ~ 62
        assert 55 <= N <= 75
        assert 5 <= n2 <= 10
        assert 5 <= n3 <= 10

    def test_no_duplicates_within_fibre_count(self):
        """Subsampling without replacement: each library entry appears at
        most once in the output."""
        sims = _make_fake_sims({1: 1000, 2: 2000, 3: 7000})
        target = {1: 0.5, 2: 0.3, 3: 0.2}
        out = rebalance.rebalance_force_library(sims, target_fractions=target, seed=7)
        # Use the unique first-feature column of signals as proxy
        sig0 = out["signals"][:, 0]
        # Allow tiny collision chance via float bits; assert no significant dupes
        assert len(np.unique(sig0)) >= 0.99 * len(sig0)
