"""
Integration tests for the Stanford HARDI inter-method comparison.

The aim is to compare FORCE-derived metrics against established baselines
on the same data — replicating the FORCE paper's Figure C1 (FORCE vs DTI)
and Figure 5 (FORCE vs AMICO NODDI). DTI baseline is paper-aligned and
trivially available; NODDI via dmipy is a stretch goal.
"""

from pathlib import Path

import numpy as np
import pytest


inter = pytest.importorskip("validation.lib.force_inter_method")


# --------------------------------------------------------------------------- #
# 1. DTI baseline fit
# --------------------------------------------------------------------------- #

class TestDtiBaselineFit:
    def test_returns_fa_md_rd_ad_with_correct_shape(self):
        """Fit DTI on a small in-mask ROI and verify output shapes + ranges."""
        from validation.lib.force_stanford import load_stanford_hardi

        data, _, mask, gtab = load_stanford_hardi()

        # 5×5×1 in-mask ROI for speed
        idx = np.argwhere(mask)
        cx, cy = int(np.median(idx[:, 0])), int(np.median(idx[:, 1]))
        z = int(np.median(idx[:, 2]))
        small_mask = np.zeros_like(mask)
        small_mask[cx-2:cx+3, cy-2:cy+3, z:z+1] = (
            mask[cx-2:cx+3, cy-2:cy+3, z:z+1]
        )

        maps = inter.fit_dti_baseline(data, gtab, small_mask)

        for k in ("fa", "md", "rd", "ad"):
            assert maps[k].shape == data.shape[:3]
        # FA must lie in [0, 1] within mask
        fa_in = maps["fa"][small_mask]
        assert (fa_in >= -1e-3).all() and (fa_in <= 1.0 + 1e-3).all()
        # MD must be positive within mask
        md_in = maps["md"][small_mask]
        assert (md_in > 0).all()


# --------------------------------------------------------------------------- #
# 2. Pearson correlation helper
# --------------------------------------------------------------------------- #

class TestMaskedPearson:
    def test_perfectly_correlated_returns_one(self):
        a = np.linspace(0, 1, 100)
        b = a.copy()
        mask = np.ones(100, dtype=bool).reshape(10, 10, 1)
        r = inter.masked_pearson(a.reshape(10, 10, 1), b.reshape(10, 10, 1), mask)
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_anticorrelated_returns_minus_one(self):
        a = np.linspace(0, 1, 100).reshape(10, 10, 1)
        b = -a + 1.0
        mask = np.ones(a.shape, dtype=bool)
        r = inter.masked_pearson(a, b, mask)
        assert r == pytest.approx(-1.0, abs=1e-6)

    def test_uncorrelated_close_to_zero(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(10, 10, 10))
        b = rng.normal(size=(10, 10, 10))
        mask = np.ones(a.shape, dtype=bool)
        r = inter.masked_pearson(a, b, mask)
        assert abs(r) < 0.2  # 1000 samples → r ~ 0 with std ~ 0.03

    def test_mask_ignored_voxels_dont_count(self):
        """Voxels outside the mask must not influence the correlation."""
        # In-mask: 5 voxels with perfect linear relation.
        # Out-of-mask: 5 voxels of pure noise that would crush the correlation
        # if included.
        rng = np.random.default_rng(0)
        a = np.zeros((2, 5, 1))
        b = np.zeros((2, 5, 1))
        a[0, :, 0] = np.linspace(0, 1, 5)
        b[0, :, 0] = np.linspace(0, 1, 5)
        a[1, :, 0] = rng.normal(size=5) * 100
        b[1, :, 0] = rng.normal(size=5) * 100
        mask = np.zeros((2, 5, 1), dtype=bool)
        mask[0, :, 0] = True  # only the first row in-mask
        r = inter.masked_pearson(a, b, mask)
        assert r == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# 3. Map cache resolution
# --------------------------------------------------------------------------- #

class TestForceMapsLoad:
    def test_load_force_maps_returns_expected_keys(self):
        cache = (Path.home() / ".cache" / "force_stanford"
                 / "force_stanford_maps.npz")
        if not cache.exists():
            pytest.skip(
                f"FORCE maps not cached at {cache}; "
                "run validate_force_stanford_hardi.py first."
            )
        maps = inter.load_force_maps()
        for k in ("fa", "md", "nd", "dispersion", "wm_fraction",
                  "csf_fraction", "mask"):
            assert k in maps
        assert maps["mask"].dtype == bool
