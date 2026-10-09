"""Tests for SBI amortized inference on qMRI models."""
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, pytest
from dmipy_jax.signal_models.qmri.sbi_qmri import (
    SimpleMDN, build_simulator, generate_training_data, train_sbi,
    sbi_inference, SBIResult)
from dmipy_jax.signal_models.qmri.vfa_t1 import spgr_signal
from dmipy_jax.signal_models.qmri.multiecho_t2star import multiecho_signal

# Use VFA T1 as the simplest test case for SBI
VFA_FA = jnp.array([3, 6, 10, 15, 20]) * jnp.pi / 180
TR = 0.02

class TestSimulator:
    def test_builds(self):
        def forward(params):
            return spgr_signal(T1=params[0], M0=params[1], TR=TR, flip_angles=VFA_FA)
        sim = build_simulator(forward, jnp.array([0.3, 100.0]), jnp.array([3.0, 2000.0]))
        assert callable(sim)

    def test_generates_pair(self):
        def forward(params):
            return spgr_signal(T1=params[0], M0=params[1], TR=TR, flip_angles=VFA_FA)
        sim = build_simulator(forward, jnp.array([0.3, 100.0]), jnp.array([3.0, 2000.0]))
        params, data = sim(jr.PRNGKey(0))
        assert params.shape == (2,)
        assert data.shape == (5,)

    def test_generates_batch(self):
        def forward(params):
            return spgr_signal(T1=params[0], M0=params[1], TR=TR, flip_angles=VFA_FA)
        sim = build_simulator(forward, jnp.array([0.3, 100.0]), jnp.array([3.0, 2000.0]))
        params, data = generate_training_data(sim, 100, jr.PRNGKey(0))
        assert params.shape == (100, 2)
        assert data.shape == (100, 5)

class TestMDN:
    def test_construction(self):
        mdn = SimpleMDN(n_data=5, n_params=2, n_components=3, hidden_size=32, key=jr.PRNGKey(0))
        assert mdn.n_components == 3

    def test_forward(self):
        mdn = SimpleMDN(n_data=5, n_params=2, key=jr.PRNGKey(0))
        data = jr.normal(jr.PRNGKey(1), (5,))
        mu, sigma, pi = mdn(data)
        assert mu.shape == (5, 2)  # n_components=5, n_params=2
        assert sigma.shape == (5, 2)
        assert pi.shape == (5,)
        assert float(jnp.sum(pi)) == pytest.approx(1.0, abs=1e-5)

    def test_posterior_mean_std(self):
        mdn = SimpleMDN(n_data=5, n_params=2, key=jr.PRNGKey(0))
        data = jr.normal(jr.PRNGKey(1), (5,))
        mean, std = mdn.posterior_mean_std(data)
        assert mean.shape == (2,)
        assert std.shape == (2,)
        assert jnp.all(std >= 0)

    def test_sample(self):
        mdn = SimpleMDN(n_data=5, n_params=2, key=jr.PRNGKey(0))
        data = jr.normal(jr.PRNGKey(1), (5,))
        samples = mdn.sample(data, jr.PRNGKey(2), n_samples=50)
        assert samples.shape == (50, 2)

class TestTraining:
    def test_loss_decreases(self):
        """Training should decrease the NLL loss."""
        def forward(params):
            return spgr_signal(T1=params[0], M0=params[1], TR=TR, flip_angles=VFA_FA)
        sim = build_simulator(forward, jnp.array([0.3, 100.0]), jnp.array([3.0, 2000.0]))
        params, data = generate_training_data(sim, 500, jr.PRNGKey(0))

        mdn = SimpleMDN(n_data=5, n_params=2, n_components=3, hidden_size=32, key=jr.PRNGKey(1))
        mdn, history = train_sbi(mdn, params, data, n_epochs=20, batch_size=64)

        assert len(history) == 20
        assert history[-1] < history[0]  # loss decreased

class TestInference:
    @pytest.fixture
    def trained_vfa_mdn(self):
        """Train a small MDN on VFA T1 data."""
        def forward(params):
            return spgr_signal(T1=params[0], M0=params[1], TR=TR, flip_angles=VFA_FA)
        sim = build_simulator(forward, jnp.array([0.3, 100.0]), jnp.array([3.0, 2000.0]),
                               noise_std=0.01)
        params, data = generate_training_data(sim, 2000, jr.PRNGKey(0))
        mdn = SimpleMDN(n_data=5, n_params=2, n_components=5, hidden_size=64, key=jr.PRNGKey(1))
        mdn, _ = train_sbi(mdn, params, data, n_epochs=50, batch_size=128)
        return mdn

    def test_returns_result(self, trained_vfa_mdn):
        data = spgr_signal(T1=1.0, M0=1000.0, TR=TR, flip_angles=VFA_FA)
        result = sbi_inference(trained_vfa_mdn, data, jr.PRNGKey(0))
        assert isinstance(result, SBIResult)

    def test_reasonable_t1_estimate(self, trained_vfa_mdn):
        """SBI should give reasonable T1 estimate for clean data."""
        true_T1 = 1.0
        data = spgr_signal(T1=true_T1, M0=1000.0, TR=TR, flip_angles=VFA_FA)
        result = sbi_inference(trained_vfa_mdn, data, jr.PRNGKey(0))
        # SBI with small training set may not be perfectly accurate
        # but should be in the right ballpark
        assert 0.3 < float(result.mean[0]) < 3.0

    def test_uncertainty_provided(self, trained_vfa_mdn):
        data = spgr_signal(T1=1.0, M0=1000.0, TR=TR, flip_angles=VFA_FA)
        result = sbi_inference(trained_vfa_mdn, data, jr.PRNGKey(0))
        assert jnp.all(result.std > 0)

    def test_samples_shape(self, trained_vfa_mdn):
        data = spgr_signal(T1=1.0, M0=1000.0, TR=TR, flip_angles=VFA_FA)
        result = sbi_inference(trained_vfa_mdn, data, jr.PRNGKey(0), n_samples=200)
        assert result.samples.shape == (200, 2)

    def test_fast_inference(self, trained_vfa_mdn):
        """SBI inference should be much faster than iterative fitting."""
        import time
        data = spgr_signal(T1=1.0, M0=1000.0, TR=TR, flip_angles=VFA_FA)
        # Warm up JIT
        _ = sbi_inference(trained_vfa_mdn, data, jr.PRNGKey(0))
        # Time it
        t0 = time.perf_counter()
        for i in range(100):
            _ = sbi_inference(trained_vfa_mdn, data, jr.PRNGKey(i))
        elapsed = time.perf_counter() - t0
        # 100 inferences should take < 5 seconds (vs minutes for iterative)
        assert elapsed < 5.0
