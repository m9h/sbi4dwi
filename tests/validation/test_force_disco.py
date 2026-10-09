"""
Integration tests pinning the DiSCo phantom benchmark wiring.

DiSCo (Diffusion-Simulated Connectivity Dataset, Rafael-Patiño et al. 2021,
DOI 10.17632/fgf86jdfg6) is the FORCE paper's §3.2 phantom benchmark and is
fetched by ``dipy.data.fetch_disco1_dataset()`` into ``~/.dipy/disco/``.

Wiring under test:
  - load_disco_subject: returns (data, mask, gtab, gt) with consistent
    shapes; sub-selects b=0 + b=1900 single-shell extraction matching the
    paper's protocol.
  - filter_to_single_shell: helper to extract a single shell + b=0 from a
    multi-shell gradient table.
  - DiSCo ground-truth maps load cleanly and have expected shapes.
"""

from pathlib import Path

import numpy as np
import pytest


disco = pytest.importorskip("validation.lib.force_disco")


DISCO_ROOT = Path.home() / ".dipy" / "disco" / "disco_1"


@pytest.fixture(scope="module")
def disco_available():
    if not DISCO_ROOT.exists():
        pytest.skip(
            f"DiSCo dataset not present at {DISCO_ROOT}; run "
            "`uv run python -c 'from dipy.data import fetch_disco1_dataset; "
            "fetch_disco1_dataset()'` to download."
        )


# --------------------------------------------------------------------------- #
# 1. Single-shell extraction
# --------------------------------------------------------------------------- #

class TestFilterToSingleShell:
    def test_picks_b0_plus_target_shell(self):
        import numpy as np

        bvals = np.array([0, 0, 1000, 1900, 1900, 3100, 13200, 1900, 0])
        bvecs = np.eye(3)[np.array([0, 1, 0, 2, 0, 1, 2, 1, 0])].astype(float)
        idx = disco.filter_to_single_shell(bvals, target_b=1900, tol=50)
        # Should pick 3 b=0 entries and 3 b=1900 entries
        kept = bvals[idx]
        assert (kept[kept == 0]).size == 3
        assert ((kept >= 1850) & (kept <= 1950)).sum() == 3
        # Total: 6 entries
        assert idx.size == 6


# --------------------------------------------------------------------------- #
# 2. DiSCo subject loader
# --------------------------------------------------------------------------- #

class TestLoadDiscoSubject:
    def test_returns_consistent_shapes(self, disco_available):
        out = disco.load_disco_subject(subject=1, snr=None, single_shell_b=1900)
        # Expect (data, mask, gtab, ground_truth_dict)
        assert "data" in out and "mask" in out and "gtab" in out and "gt" in out
        data = out["data"]
        mask = out["mask"]
        # DiSCo highRes is 40^3; single-shell extraction has 4 b=0 + 90 b=1900 = 94
        assert data.ndim == 4
        assert data.shape[:3] == mask.shape
        assert data.shape[3] == 94, f"Expected 94 single-shell volumes, got {data.shape[3]}"
        # Mask is bool
        assert mask.dtype == bool

    def test_ground_truth_maps_loaded(self, disco_available):
        out = disco.load_disco_subject(subject=1, snr=None, single_shell_b=1900)
        gt = out["gt"]
        # Expected ground-truth maps from DiSCo distribution
        assert "ndi" in gt   # Strand_Intra_Volume_Fraction
        assert "diameter" in gt
        # Maps have same spatial shape as data
        for k in ("ndi", "diameter"):
            assert gt[k].shape == out["data"].shape[:3]
        # NDI is in [0, 1]
        m = out["mask"]
        ndi = gt["ndi"][m]
        assert (ndi >= 0).all() and (ndi <= 1.0 + 1e-3).all()

    def test_snr_variant_swaps_data(self, disco_available):
        out_clean = disco.load_disco_subject(subject=1, snr=None, single_shell_b=1900)
        out_snr30 = disco.load_disco_subject(subject=1, snr=30, single_shell_b=1900)
        # Same shape, different values
        assert out_clean["data"].shape == out_snr30["data"].shape
        assert not np.allclose(out_clean["data"], out_snr30["data"])
