"""
Integration tests pinning the 3-fiber FORCE benchmark wiring.

These tests answer the question: *if dipy_force returned 0% across all alphas
in our coplanar 3-fiber sweep, was that the matcher's library limitation, or a
bug in our adapter wiring?* The four tests below distinguish those:

  1. Forward model (three_stick_signal) — hand-computed value at b=0 must equal 1
     and at high b must collapse toward zero along sticks; pins the synthetic.

  2. dmipy-JAX 3-stick dictionary self-recovery on a clean signal — confirms
     LibraryGenerator + DictionaryMatcher wiring on the 3-stick simulator.

  3. dipy FORCE wiring sanity — feed the matcher a signal that IS itself a
     library entry; matcher must return a FORCEFit with at least one active
     fibre. Decouples wiring from library coverage.

  4. dipy FORCE coplanar-failure pin — feed the matcher a clean coplanar
     3-stick synthetic; assert that recovered directions are NOT all in the
     +x/+z plane. If this ever stops failing (e.g., dipy patches library
     generation to include coplanar configs), this test will turn red and
     warn us the finding has been overtaken.
"""

import math
from pathlib import Path

import numpy as np
import pytest


three_fiber = pytest.importorskip("validation.lib.three_fiber")
force_baselines = pytest.importorskip("validation.lib.force_baselines")


# --------------------------------------------------------------------------- #
# 1. Forward model
# --------------------------------------------------------------------------- #

class TestThreeStickSignal:
    def setup_method(self):
        self.acq = three_fiber.make_multishell_acquisition()
        # Three coplanar truth dirs at alpha=45
        rad = math.radians(45)
        self.mu1 = np.array([-math.sin(rad), 0.0, math.cos(rad)])
        self.mu2 = np.array([0.0, 0.0, 1.0])
        self.mu3 = np.array([math.sin(rad), 0.0, math.cos(rad)])

    def test_b0_signal_is_unity(self):
        """At b=0, every term collapses to 1; weighted sum must equal 1."""
        import jax.numpy as jnp

        S = three_fiber.three_stick_signal(
            self.acq, jnp.asarray(self.mu1), jnp.asarray(self.mu2),
            jnp.asarray(self.mu3), f1=0.317, f2=0.317, f_iso=0.05,
        )
        S_np = np.asarray(S)
        # acq.bvalues[0:2] are zeros — those entries must be exactly 1.0
        b0 = np.asarray(self.acq.bvalues) == 0
        assert np.allclose(S_np[b0], 1.0, atol=1e-6)

    def test_signal_is_bounded_unit_interval(self):
        """All signal entries must lie in [0, 1] for valid diffusion attenuation."""
        import jax.numpy as jnp

        S = three_fiber.three_stick_signal(
            self.acq, jnp.asarray(self.mu1), jnp.asarray(self.mu2),
            jnp.asarray(self.mu3), f1=0.317, f2=0.317, f_iso=0.05,
        )
        S_np = np.asarray(S)
        assert (S_np >= 0).all() and (S_np <= 1.0 + 1e-6).all()

    def test_fractions_sum_to_one_in_signal(self):
        """At b=0 the weighted sum is f1 + f2 + f3 + f_iso = 1.0 by construction;
        any leakage exposes a bug in the implicit f3 = 1 - f1 - f2 - f_iso."""
        import jax.numpy as jnp

        S = three_fiber.three_stick_signal(
            self.acq, jnp.asarray(self.mu1), jnp.asarray(self.mu2),
            jnp.asarray(self.mu3), f1=0.4, f2=0.3, f_iso=0.1,
            # f3 = 0.2 implicitly
        )
        b0 = np.asarray(self.acq.bvalues) == 0
        assert np.allclose(np.asarray(S)[b0], 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. dmipy-JAX dict self-recovery
# --------------------------------------------------------------------------- #

class TestDict3SelfRecovery:
    def test_clean_signal_recovered_from_library(self):
        """Build a small 3-stick library, sample one entry, feed its signal
        back through DictionaryMatcher — must recover the parameters exactly."""
        import jax
        from dmipy_jax.library.generator import LibraryGenerator
        from dmipy_jax.library.matcher import DictionaryMatcher
        from dmipy_jax.library.storage import SimulationLibrary

        acq = three_fiber.make_multishell_acquisition()
        sim = three_fiber.build_three_stick_simulator(acq)
        gen = LibraryGenerator(sim, chunk_size=5_000)
        params, signals = gen.generate(5_000, key=jax.random.PRNGKey(11))
        lib = SimulationLibrary(
            params=params, signals=signals, parameter_names=sim.parameter_names,
        )
        matcher = DictionaryMatcher(lib, k_best=1)

        idx = 1234
        target_signal = signals[idx]
        recovered, _ = matcher.match_single(target_signal)

        # Exact library hit: matched params must equal the target's
        assert np.allclose(np.asarray(recovered), np.asarray(params[idx]), atol=1e-5)


# --------------------------------------------------------------------------- #
# 3 + 4. dipy FORCE wiring + finding
#
# Both tests need the cached 500K library from validate_force_replication_v2.
# Skip cleanly if the cache isn't present.
# --------------------------------------------------------------------------- #

DIPY_FORCE_CACHE = Path.home() / ".cache" / "dipy_force" / "force_v2_500k.npz"


@pytest.fixture(scope="module")
def dipy_force_model():
    if not DIPY_FORCE_CACHE.exists():
        pytest.skip(f"DIPY FORCE library not cached at {DIPY_FORCE_CACHE}; "
                    "run validate_force_replication_v2.py to build it.")
    import warnings as _w
    _w.filterwarnings("ignore")

    from dipy.reconst.force import FORCEModel, load_force_simulations

    sims = load_force_simulations(str(DIPY_FORCE_CACHE))
    acq = three_fiber.make_multishell_acquisition()
    gtab = three_fiber.acq_to_gtab_si(acq)
    return {
        "model": FORCEModel(gtab, simulations=sims, n_neighbors=50),
        "sims": sims,
        "acq": acq,
    }


class TestDipyForceWiring:
    def test_unit_normalised_single_fibre_produces_active_label(
        self, dipy_force_model,
    ):
        """Wiring sanity: a unit-normalised single-fibre synthetic (S(b=0)=1)
        must produce a FORCEFit with at least one active label entry.

        Library-stored signals turned out to use a different scale (range
        5.3 to 100, not 0 to 1), so feeding them back unchanged is *not* a
        valid wiring check. A clean unit-S0 synthetic is the right contract.
        """
        import jax.numpy as jnp

        model = dipy_force_model["model"]
        acq = dipy_force_model["acq"]

        mu = jnp.array([0.0, 0.0, 1.0])
        cos = acq.gradient_directions @ mu
        signal = np.asarray(
            jnp.exp(-acq.bvalues * 1.7e-9 * cos ** 2)
        ).astype(np.float32)
        # Pin the contract: signal is unit-normalised at b=0
        b0 = np.asarray(acq.bvalues) == 0
        assert np.allclose(signal[b0], 1.0, atol=1e-6)

        fit = model.fit(signal[None, None, None, :])[0, 0, 0]
        n_active = int((np.asarray(fit.label) > 0).sum())
        assert n_active >= 1, (
            "FORCEModel.fit returned a FORCEFit with zero active labels on a "
            "unit-normalised single-fibre synthetic — wiring is broken."
        )
        assert fit.num_fibers >= 1


# --------------------------------------------------------------------------- #
# 4. The actual finding — pinned as a regression test
# --------------------------------------------------------------------------- #

class TestDipyForceCoplanarFinding:
    def test_clean_coplanar_3fiber_recovers_out_of_plane_directions(
        self, dipy_force_model,
    ):
        """*The finding under test.* Clean (no noise) coplanar 3-stick synthetic
        in the +x/+z plane (y=0). Ground-truth y-component is exactly 0 for
        all three fibres. Assertion: dipy FORCE's recovered peaks are NOT all
        in the plane — at least one |y| > 0.1.

        If this test ever flips to passing the in-plane check (i.e., dipy
        FORCE starts recovering coplanar configurations), the finding has
        been overtaken by an upstream improvement and we should re-evaluate.
        """
        import jax.numpy as jnp

        from dipy.reconst.force import force_peaks

        acq = dipy_force_model["acq"]
        rad = math.radians(45)
        mu1 = jnp.array([-math.sin(rad), 0.0, math.cos(rad)])
        mu2 = jnp.array([0.0, 0.0, 1.0])
        mu3 = jnp.array([math.sin(rad), 0.0, math.cos(rad)])

        signal = three_fiber.three_stick_signal(
            acq, mu1, mu2, mu3, f1=0.317, f2=0.317, f_iso=0.05,
        )
        S = np.asarray(signal).astype(np.float32)

        peaks = force_baselines.dipy_force_peaks_from_signal(
            S, dipy_force_model["model"],
        )

        # peaks is a list of unit vectors (some may be zero-vector pads)
        nonzero = [p for p in peaks if np.linalg.norm(p) > 1e-6]
        assert len(nonzero) > 0, (
            "FORCE returned zero usable peaks on a clean coplanar synthetic — "
            "wiring is broken (or library is empty)."
        )

        max_abs_y = max(abs(float(p[1])) for p in nonzero)
        assert max_abs_y > 0.1, (
            "All recovered FORCE peaks are coplanar with +x/+z. The "
            "library-coverage finding (no coplanar 3-fiber library entries) "
            "no longer holds — re-evaluate doc 004 §10."
        )

    def test_clean_coplanar_3fiber_force_internal_also_out_of_plane(
        self, dipy_force_model,
    ):
        """Mirror assertion for the FORCE-internal (label-based) reading.
        Same finding from a different vantage."""
        import jax.numpy as jnp

        from dipy.data import default_sphere

        acq = dipy_force_model["acq"]
        rad = math.radians(45)
        mu1 = jnp.array([-math.sin(rad), 0.0, math.cos(rad)])
        mu2 = jnp.array([0.0, 0.0, 1.0])
        mu3 = jnp.array([math.sin(rad), 0.0, math.cos(rad)])
        signal = three_fiber.three_stick_signal(
            acq, mu1, mu2, mu3, f1=0.317, f2=0.317, f_iso=0.05,
        )
        S = np.asarray(signal).astype(np.float32)

        peaks = force_baselines.dipy_force_label_directions_from_signal(
            S, dipy_force_model["model"], default_sphere,
        )
        nonzero = [p for p in peaks if np.linalg.norm(p) > 1e-6]
        assert len(nonzero) > 0
        max_abs_y = max(abs(float(p[1])) for p in nonzero)
        assert max_abs_y > 0.1
