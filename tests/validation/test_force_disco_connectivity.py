"""
Integration tests pinning the FORCE → DiSCo connectivity-matrix pipeline.

This reproduces the FORCE paper §3.2 protocol literally:
  1. Fit FORCE on DiSCo single-shell b=1900 with the tuned library.
  2. Extract peaks via force_peaks(fit) → PeaksAndMetrics.
  3. Run LocalTracking (EuDX-style) on the peaks within the ROI mask.
  4. Compute connectivity_matrix(streamlines, ROI labels).
  5. Compare to ground-truth Connectivity_Matrix_Cross-Sectional_Area.txt.

Each test pins one stage of the pipeline.
"""

from pathlib import Path

import numpy as np
import pytest


conn = pytest.importorskip("validation.lib.force_disco_connectivity")


DISCO_ROOT = Path.home() / ".dipy" / "disco" / "disco_1"


@pytest.fixture(scope="module")
def disco_ready():
    if not DISCO_ROOT.exists():
        pytest.skip(f"DiSCo not at {DISCO_ROOT}")


# --------------------------------------------------------------------------- #
# 1. Ground-truth connectivity matrix loader
# --------------------------------------------------------------------------- #

class TestGroundTruthConnectivity:
    def test_load_gt_returns_16x16(self, disco_ready):
        gt = conn.load_gt_connectivity(subject=1)
        assert gt.shape == (16, 16)
        # Cross-sectional areas are nonnegative
        assert (gt >= 0).all()
        # Has 50 nonzero entries per DiSCo subject 1 spec
        assert (gt > 0).sum() == 50


# --------------------------------------------------------------------------- #
# 1b. Lin's CCC (penalises systematic bias)
# --------------------------------------------------------------------------- #

class TestLinCCC:
    def test_identical_arrays_ccc_is_one(self):
        a = np.linspace(0, 1, 100)
        assert conn.lin_ccc(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_uniform_bias_drops_ccc_below_pearson(self):
        """A uniform 0.17 shift (the §16.3 NDI bias) leaves Pearson r=1 but
        crushes CCC. This is the headline reason for adding CCC to the
        reporting suite. For uniform a∈[0,1] + 0.17 shift the empirical
        CCC is ~0.85."""
        a = np.linspace(0, 1, 200)
        b = a + 0.17  # uniform shift
        r_pearson = float(np.corrcoef(a, b)[0, 1])
        ccc = conn.lin_ccc(a, b)
        assert r_pearson == pytest.approx(1.0, abs=1e-6)
        # Pearson stays perfect, CCC drops well below 1
        assert ccc < 0.90, f"CCC {ccc} should reflect bias; Pearson r={r_pearson}"
        assert ccc > 0.6, f"CCC {ccc} too low for a r=1 + 0.17-shift relationship"

    def test_scale_difference_drops_ccc(self):
        """Pearson is also scale-invariant; CCC penalises scale mismatch."""
        a = np.linspace(0, 1, 200)
        b = a * 2.0  # scaled
        ccc = conn.lin_ccc(a, b)
        # Pearson would be 1.0; CCC should be ~0.5 (half-line vs full)
        assert ccc < 0.95, f"CCC {ccc} should reflect scale bias"

    def test_mask_restricts_computation(self):
        a = np.zeros(100); b = np.zeros(100)
        a[:50] = np.linspace(0, 1, 50)
        b[:50] = np.linspace(0, 1, 50)
        # Out-of-mask region: noise that would crush CCC if included
        rng = np.random.default_rng(0)
        a[50:] = rng.normal(scale=100, size=50)
        b[50:] = -rng.normal(scale=100, size=50)
        mask = np.zeros(100, dtype=bool); mask[:50] = True
        assert conn.lin_ccc(a, b, mask=mask) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# 2. Connectivity matrix Pearson r
# --------------------------------------------------------------------------- #

class TestConnectivityPearson:
    def test_self_correlation_is_one(self):
        rng = np.random.default_rng(0)
        m = rng.exponential(size=(16, 16))
        m = m + m.T  # symmetric
        assert conn.connectivity_pearson(m, m) == pytest.approx(1.0, abs=1e-6)

    def test_zero_matrix_returns_nan(self):
        m = np.zeros((16, 16))
        assert np.isnan(conn.connectivity_pearson(m, m))

    def test_uses_upper_triangle_only(self):
        """Both matrices should be reduced to their symmetric (upper-tri,
        excluding diagonal) entries before correlation, to avoid double-
        counting symmetric pairs and the trivial-zero diagonal."""
        rng = np.random.default_rng(7)
        a = rng.exponential(size=(16, 16)); a = a + a.T
        b = a.copy()
        # Perturb only the lower triangle (which we'll discard) — pearson
        # should still be 1.0
        b[np.tril_indices(16, k=-1)] += rng.normal(scale=10, size=16 * 15 // 2)
        r = conn.connectivity_pearson(a, b)
        assert r == pytest.approx(1.0, abs=1e-6), (
            f"Function appears to use full matrix not upper-triangle; r={r}"
        )


# --------------------------------------------------------------------------- #
# 3. Pipeline smoke: fit a single ROI, get peaks, tract, count streamlines
# --------------------------------------------------------------------------- #

class TestTractographySmoke:
    def test_streamline_generation_returns_nonempty_default_lib(self, disco_ready):
        """End-to-end smoke: load DiSCo, refit FORCE on a 3-ROI subset,
        run tractography on the default-library fit. Must produce at
        least one streamline.
        """
        result = conn.run_force_connectivity(
            subject=1, snr=30, tuned=False,
            _smoke_roi_subset=(1, 2, 3), seed_density=1,
        )
        assert result["streamlines_count"] > 0
        assert result["connectivity"].shape == (16, 16)

    def test_tuned_library_breaks_tractography(self, disco_ready):
        """**Pinned upstream bug**: a FORCE library generated with
        ``wm_threshold=1.0`` (paper §3.2 protocol for DiSCo) has:

          - all-zero ``sims['odfs']`` rows
          - ``force_peaks(fit).peak_dirs`` all zeros
          - ``force_peaks(fit).peak_indices`` all -1

        even though ``peak_values`` (derived from ``fit.fracs``) are
        nonzero. Both ``LocalTracking`` and ``eudx_tracking`` therefore
        produce **zero streamlines** on a tuned-library fit.

        An earlier read suggested ``eudx_tracking`` works on the tuned
        library via ``peak_indices`` + ``peak_values``; that was wrong —
        ``peak_indices`` is also degenerate (all -1).

        When dipy fixes the upstream ODF path, ``odfs`` will populate,
        ``peak_dirs``/``peak_indices`` will populate, and this test will
        turn red. At that point ``run_force_connectivity`` can use
        ``tuned=True`` for the paper-aligned connectivity benchmark.
        """
        import numpy as np
        from dipy.reconst.force import load_force_simulations
        from pathlib import Path

        tuned_lib = Path.home() / ".cache" / "dipy_force" / "force_disco_500k_tuned.npz"
        if not tuned_lib.exists():
            pytest.skip("tuned DiSCo library not cached")
        sims = load_force_simulations(str(tuned_lib))
        nz_odfs = (np.abs(sims["odfs"]).sum(axis=1) > 0).sum()
        assert nz_odfs == 0, (
            f"Tuned library now has {nz_odfs} nonzero ODFs (was 0). "
            "Upstream may have fixed the ODF computation path under "
            "wm_threshold=1.0."
        )

        # Pin the downstream consequence: tractography fails.
        result = conn.run_force_connectivity(
            subject=1, snr=30, tuned=True,
            _smoke_roi_subset=(1, 2, 3), seed_density=1,
        )
        assert result["streamlines_count"] == 0, (
            f"Tuned-library tractography unexpectedly produced "
            f"{result['streamlines_count']} streamlines. Upstream ODF "
            "suppression may be partially fixed."
        )
