"""
Exchange- and time-aware white-matter kernel for the PRISM-JAX family
(doc 008 §9 item 2). One fibre population = intra-axonal stick ⇄
extra-axonal zeppelin with Kärger exchange (exchange time τ_ex), so the
signal depends on the diffusion time and the model is identifiable only
with multi-Δ data — which is what the acquisition-design layer optimises.

Two implementations of the same physics:

* `karger_pgse`  — closed form: for each measurement the 2×2 Kärger system
  dM/dt = −(diag(b/t_d · D_c) + K) M is integrated over the effective
  diffusion time t_d = Δ − δ/3 by a matrix exponential. Cheap, exact for
  narrow pulses, differentiable in every argument.
* `karger_waveform` — Diffrax ODE over an explicit gradient waveform
  g(t) (trapezoid PGSE by default, any waveform in general): the
  phase-encoding q(t) = γ∫g dt enters as −q(t)² D_c on the diagonal. This
  is the version whose gradients reach δ, Δ, ramps and free waveforms,
  i.e. the one the design layer uses; `karger_pgse` is its narrow-pulse
  limit and the unit test checks they agree.

Units: SI throughout (b in s/m², D in m²/s, times in s, τ_ex in s).
"""
from __future__ import annotations
from typing import Optional
import jax, jax.numpy as jnp
import diffrax

GAMMA = 2.6751525e8   # rad/s/T


def _exchange_matrix(f_i, tau_ex):
    """K (2×2) with detailed balance: k_ie = (1−f_i)/τ_ex·… such that the
    equilibrium fractions are (f_i, 1−f_i) and the exchange time is τ_ex
    (1/τ_ex = k_ie + k_ei)."""
    k_ie = (1.0 - f_i) / tau_ex          # intra → extra
    k_ei = f_i / tau_ex                  # extra → intra
    return jnp.array([[-k_ie, k_ei], [k_ie, -k_ei]])


def _rates(bvals, bvecs, dirs, d_par, d_par_ex, d_perp):
    """Per-measurement diffusion attenuation rates (M,2) for stick / zeppelin."""
    c2 = (bvecs @ dirs) ** 2                              # (M,)
    r_in = d_par * c2
    r_ex = d_perp + (d_par_ex - d_perp) * c2
    return jnp.stack([r_in, r_ex], -1)                    # (M,2) in m²/s


def karger_pgse(bvals, bvecs, dirs, f_i, d_par, d_par_ex, d_perp, tau_ex, delta, Delta):
    """Closed-form Kärger signal for a PGSE scheme. bvals (M,), bvecs (M,3),
    dirs (3,), delta/Delta scalars or (M,). Returns (M,) normalised signal."""
    bvals = jnp.asarray(bvals); t_d = jnp.broadcast_to(jnp.asarray(Delta) - jnp.asarray(delta) / 3.0, bvals.shape)
    R = _rates(bvals, bvecs, dirs, d_par, d_par_ex, d_perp)              # (M,2)
    K = _exchange_matrix(f_i, tau_ex)
    m0 = jnp.array([f_i, 1.0 - f_i])

    def one(b, td, r):
        A = -jnp.diag(b / td * r) + K                                    # rate matrix (1/s)
        return jnp.sum(jax.scipy.linalg.expm(A * td) @ m0)
    return jax.vmap(one)(bvals, t_d, R)


def trapezoid_pgse(t, G, delta, Delta, rise=1e-4):
    """Effective gradient amplitude of a PGSE pair (spin-echo sign flip applied)."""
    def lobe(t0):
        return jnp.clip((t - t0) / rise, 0, 1) * jnp.clip((t0 + delta - t) / rise, 0, 1)
    return G * (lobe(0.0) - lobe(Delta))


def karger_waveform(bvals, bvecs, dirs, f_i, d_par, d_par_ex, d_perp, tau_ex, delta, Delta,
                    waveform=trapezoid_pgse, n_steps: int = 400, rise: float = 1e-4):
    """Diffrax integration of the Kärger system under an explicit waveform.
    G per measurement from b = γ²G²δ²(Δ − δ/3). Differentiable in δ, Δ, b,
    the tissue parameters and the waveform's own parameters."""
    bvals = jnp.asarray(bvals)
    R = _rates(jnp.ones_like(bvals), bvecs, dirs, d_par, d_par_ex, d_perp)   # (M,2): D_c cos² etc.
    K = _exchange_matrix(f_i, tau_ex)
    m0 = jnp.array([f_i, 1.0 - f_i])
    T = Delta + delta + 2 * rise
    # b-value of the *actual* waveform at unit amplitude (ramps included), so
    # the requested b is met exactly: b = γ² G² ∫ (∫g)² dt
    ts = jnp.linspace(0.0, T, n_steps + 1); dt = T / n_steps
    q_unit = jnp.cumsum(waveform(ts, 1.0, delta, Delta, rise)) * dt
    b_unit = GAMMA ** 2 * jnp.sum(q_unit ** 2) * dt
    G = jnp.sqrt(bvals / b_unit)

    def one(g_amp, r):
        # state: (q, M_in, M_ex); q = γ ∫ g dt
        def vf(t, y, args):
            q = y[0]; M = y[1:]
            dq = GAMMA * waveform(t, g_amp, delta, Delta, rise)
            dM = (-(q ** 2) * r + 0.0) * M + K @ M
            return jnp.concatenate([jnp.array([dq]), dM])
        sol = diffrax.diffeqsolve(diffrax.ODETerm(vf), diffrax.Tsit5(), 0.0, T, T / n_steps,
                                  jnp.concatenate([jnp.array([0.0]), m0]),
                                  stepsize_controller=diffrax.ConstantStepSize(), max_steps=n_steps + 10)
        return jnp.sum(sol.ys[-1, 1:])
    return jax.vmap(one)(G, R)


def prism_exchange_wm(bvals, bvecs, dirs, f_i, d_par, d_par_ex, d_perp, tau_ex, delta, Delta):
    """(N,K,M) WM kernels for a batch: dirs (N,K,3), scalars or (N,) for the rest."""
    def per_voxel(d, fi, dp, dpe, dpp, tau):
        return jax.vmap(lambda u: karger_pgse(bvals, bvecs, u, fi, dp, dpe, dpp, tau, delta, Delta))(d)
    bc = lambda x: jnp.broadcast_to(jnp.asarray(x), (dirs.shape[0],))
    return jax.vmap(per_voxel)(dirs, bc(f_i), bc(d_par), bc(d_par_ex), bc(d_perp), bc(tau_ex))
