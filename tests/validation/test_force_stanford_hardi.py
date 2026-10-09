"""
Integration tests pinning the dipy FORCE → Stanford HARDI replication.

The aim of this run is to confirm the FORCE paper's headline result on its
own design data (Figure 6: NODDI parameter maps from a single-shell
acquisition). These tests verify wiring + sanity ranges; the visual
comparison to the paper's published figure happens manually after the run.
"""

from pathlib import Path

import numpy as np
import pytest


force_stanford = pytest.importorskip("validation.lib.force_stanford")


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #

class TestStanfordHardiLoad:
    def test_load_returns_data_mask_gtab(self):
        """Pin the API contract: returns (data, affine, mask, gtab)."""
        data, affine, mask, gtab = force_stanford.load_stanford_hardi()
        assert data.ndim == 4
        assert affine.shape == (4, 4)
        assert mask.shape == data.shape[:3]
        # Stanford HARDI: 160 volumes (10 b=0 + 150 b=2000)
        assert data.shape[3] == 160

    def test_mask_is_boolean_with_brain_voxels(self):
        _, _, mask, _ = force_stanford.load_stanford_hardi()
        assert mask.dtype == bool
        # Stanford HARDI mask should pick out something between 100K and 1M
        # voxels in this volume size
        assert 100_000 < int(mask.sum()) < 1_000_000


# --------------------------------------------------------------------------- #
# 2. Small ROI fit — wiring sanity on real data
# --------------------------------------------------------------------------- #

class TestSmallRoiFit:
    def test_force_fit_on_5x5x1_roi_returns_expected_attrs(self):
        """Fit FORCE on a tiny in-mask ROI; verify the FORCEFit attrs all
        populate. This is the wiring check on real Stanford HARDI."""
        import warnings as _w
        _w.filterwarnings("ignore")

        from dipy.reconst.force import FORCEModel, load_force_simulations

        cache = Path.home() / ".cache" / "dipy_force" / "force_matched_500k.npz"
        if not cache.exists():
            pytest.skip(
                f"FORCE library cache not present at {cache}; "
                "run validate_force_matched.py first."
            )

        data, _, mask, gtab = force_stanford.load_stanford_hardi()
        sims = load_force_simulations(str(cache))
        model = FORCEModel(gtab, simulations=sims, n_neighbors=50)

        # Pick a 5×5×1 in-mask slab
        idx = np.argwhere(mask)
        # Choose an ROI roughly in the middle of the brain
        z = int(np.median(idx[:, 2]))
        slab_mask = np.zeros_like(mask)
        cx, cy = int(np.median(idx[:, 0])), int(np.median(idx[:, 1]))
        slab_mask[cx - 2:cx + 3, cy - 2:cy + 3, z:z + 1] = mask[cx - 2:cx + 3, cy - 2:cy + 3, z:z + 1]
        n_vox = int(slab_mask.sum())
        assert n_vox > 0, "Slab does not intersect mask"

        fit = model.fit(data, mask=slab_mask)
        # All paper-reported attrs must populate
        for attr in ("fa", "md", "wm_fraction", "gm_fraction",
                     "csf_fraction", "num_fibers", "dispersion", "nd"):
            arr = getattr(fit, attr)
            assert arr.shape == data.shape[:3], (
                f"FORCEFit.{attr} shape {arr.shape} != volume shape {data.shape[:3]}"
            )


# --------------------------------------------------------------------------- #
# 3. Output map sanity ranges
# --------------------------------------------------------------------------- #

class TestForceMapSanity:
    def test_fa_within_unit_interval(self):
        """FORCE FA must lie in [0, 1] within the brain mask."""
        from validation.lib.force_stanford import (
            fit_force_or_load_cached_maps,
        )
        cache = Path.home() / ".cache" / "dipy_force" / "force_matched_500k.npz"
        if not cache.exists():
            pytest.skip("FORCE library cache not present.")
        maps = fit_force_or_load_cached_maps(library_cache=str(cache),
                                              n_jobs=1, _smoke_n_voxels=200)
        fa = maps["fa"]
        mask = maps["mask"]
        fa_in = fa[mask]
        # FA is bounded in [0, 1] by definition. Allow a little float slack.
        assert (fa_in >= -1e-3).all() and (fa_in <= 1.0 + 1e-3).all()

    def test_tissue_fractions_sum_close_to_one(self):
        """Within the mask, WM + GM + CSF (CSF == FW per FORCE) ≈ 1."""
        from validation.lib.force_stanford import (
            fit_force_or_load_cached_maps,
        )
        cache = Path.home() / ".cache" / "dipy_force" / "force_matched_500k.npz"
        if not cache.exists():
            pytest.skip("FORCE library cache not present.")
        maps = fit_force_or_load_cached_maps(library_cache=str(cache),
                                              n_jobs=1, _smoke_n_voxels=200)
        m = maps["mask"]
        total = (
            maps["wm_fraction"][m]
            + maps["gm_fraction"][m]
            + maps["csf_fraction"][m]
        )
        # Should sum to ~1 within mask. Allow ±0.05 slack for matcher
        # interpolation across discrete library entries.
        assert np.median(np.abs(total - 1.0)) < 0.05
