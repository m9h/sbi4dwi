import numpy as np, jax, jax.numpy as jnp, pytest
from dmipy_jax.validation import prism_exchange as px

jax.config.update("jax_enable_x64", True)


def _scheme():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(12, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    b = np.repeat([1000.0, 3000.0], 6) * 1e6
    return jnp.asarray(b), jnp.asarray(g)


ARGS = dict(dirs=jnp.array([0.0, 0.0, 1.0]), f_i=0.55, d_par=1.7e-9, d_par_ex=1.5e-9, d_perp=0.5e-9)


def test_no_exchange_limit_is_mixture():
    b, g = _scheme()
    s = px.karger_pgse(b, g, tau_ex=1e6, delta=10e-3, Delta=30e-3, **ARGS)
    c2 = (g @ ARGS["dirs"]) ** 2
    ref = ARGS["f_i"] * jnp.exp(-b * ARGS["d_par"] * c2) + (1 - ARGS["f_i"]) * jnp.exp(-b * (ARGS["d_perp"] + (ARGS["d_par_ex"] - ARGS["d_perp"]) * c2))
    assert np.allclose(s, ref, atol=1e-6)


def test_fast_exchange_limit_is_single_compartment():
    b, g = _scheme()
    s = px.karger_pgse(b, g, tau_ex=1e-7, delta=10e-3, Delta=30e-3, **ARGS)
    c2 = (g @ ARGS["dirs"]) ** 2
    d_mean = ARGS["f_i"] * ARGS["d_par"] * c2 + (1 - ARGS["f_i"]) * (ARGS["d_perp"] + (ARGS["d_par_ex"] - ARGS["d_perp"]) * c2)
    assert np.allclose(s, jnp.exp(-b * d_mean), rtol=2e-3)


def test_signal_depends_on_diffusion_time():
    b, g = _scheme()
    s1 = px.karger_pgse(b, g, tau_ex=20e-3, delta=5e-3, Delta=20e-3, **ARGS)
    s2 = px.karger_pgse(b, g, tau_ex=20e-3, delta=5e-3, Delta=80e-3, **ARGS)
    assert float(jnp.max(jnp.abs(s1 - s2))) > 1e-3


def test_waveform_matches_closed_form_in_narrow_pulse_limit():
    b, g = _scheme()
    kw = dict(tau_ex=30e-3, delta=1e-3, Delta=40e-3)
    s_cf = px.karger_pgse(b, g, **ARGS, **kw)
    s_ode = px.karger_waveform(b, g, **ARGS, **kw, n_steps=2000, rise=2e-5)
    assert np.allclose(s_cf, s_ode, atol=3e-3)


def test_gradients_reach_tissue_and_timing():
    b, g = _scheme()
    def loss(theta):
        f_i, tau, Delta = theta
        return jnp.sum(px.karger_waveform(b, g, ARGS["dirs"], f_i, ARGS["d_par"], ARGS["d_par_ex"], ARGS["d_perp"], tau, 5e-3, Delta, n_steps=200))
    grad = jax.grad(loss)(jnp.array([0.5, 20e-3, 30e-3]))
    assert np.all(np.isfinite(np.asarray(grad))) and float(jnp.abs(grad[2])) > 0


def test_batched_kernel_shape():
    b, g = _scheme()
    dirs = jnp.array([[[0, 0, 1.0], [1.0, 0, 0]]] * 3)
    out = px.prism_exchange_wm(b, g, dirs, 0.5, 1.7e-9, 1.5e-9, 0.5e-9, 25e-3, 10e-3, 30e-3)
    assert out.shape == (3, 2, 12) and bool(jnp.all((out > 0) & (out <= 1.0 + 1e-9)))
