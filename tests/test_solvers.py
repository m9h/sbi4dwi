import pytest
import jax
import jax.numpy as jnp
from dmipy_jax.core.solvers import BlochSimulator, solve_diffusion_sde
import numpy as np

def test_bloch_simulator_relaxation():
    print("\nRunning Bloch Simulator Relaxation Test...")
    # T1 = 1.0s, T2 = 0.1s
    # Start with M = [1, 0, 0] (after 90 pulse)
    # G = 0
    t1 = 1.0
    t2 = 0.1
    m0 = jnp.array([0., 0., 1.])
    simulator = BlochSimulator(t1=t1, t2=t2, m0=m0)
    
    t_end = 0.5
    
    def grad_func(t):
        return jnp.array([0., 0., 0.])
    
    m_init = jnp.array([1., 0., 0.])
    pos = jnp.array([0., 0., 0.])
    
    # We need to manually jit or ensure simulate handles it. 
    # BlochSimulator.__call__ uses diffrax which usually handles JIT if inside jit, 
    # or we can just run it (it will compile internally via diffrax usually or run slow).
    # Let's run it eagerly or wrap in jit.
    
    sol = simulator(
        t_span=(0.0, t_end),
        m_init=m_init,
        gradient_waveform=grad_func,
        position=pos
    )
    
    m_final = sol.ys[-1] # shape (3,)
    
    # Analytical:
    # Mx(t) = Mx(0) * exp(-t/T2)
    # Mz(t) = M0 + (Mz(0) - M0) * exp(-t/T1) = 1 + (0 - 1) * exp(-t/T1)
    
    mx_sim = m_final[0]
    mz_sim = m_final[2]
    
    mx_anal = 1.0 * np.exp(-t_end / t2)
    mz_anal = 1.0 - np.exp(-t_end / t1)
    
    print(f"Mx: Sim={mx_sim:.5f}, Anal={mx_anal:.5f}")
    assert np.allclose(mx_sim, mx_anal, atol=1e-3), "Mx relaxation failed"
    print(f"Mz: Sim={mz_sim:.5f}, Anal={mz_anal:.5f}")
    assert np.allclose(mz_sim, mz_anal, atol=1e-3), "Mz relaxation failed"
    print("Bloch Simulator Relaxation Test Passed")


@pytest.mark.xfail(reason="known failure, pre-existing, see docs/decisions/010 §7.3: Diffrax 0.7 ControlTerm structure change", strict=False)
def test_diffusion_sde_msd():
    print("\nRunning Diffusion SDE MSD Test...")
    # Free diffusion in 3D
    # dY = 0 dt + sqrt(2D) dW
    # MSD = 6Dt
    
    # Use D = 1.0 for simplicity in units
    # D = 1e-3 mm^2/s = 1e-9 m^2/s
    D_coeff = 1.0 
    
    key = jax.random.PRNGKey(42)
    t_end = 1.0
    
    N_particles = 2000
    
    def run_one_particle(k):
        return solve_diffusion_sde(
            t_span=(0.0, t_end),
            y0=jnp.zeros(3),
            drift=lambda t, y, args: jnp.zeros(3),
            diffusion=lambda t, y, args: jnp.sqrt(2 * D_coeff) * jnp.eye(3), # (3, 3) matrix
            dt0=1e-2, 
            key=k
        )
        
    # vmap the solve function
    batch_run = jax.jit(jax.vmap(run_one_particle))
    keys = jax.random.split(key, N_particles)
    
    sols = batch_run(keys)
    
    final_pos = sols.ys[:, -1, :] # (N, 3)
    
    sq_displacement = jnp.sum(final_pos**2, axis=1) # (N,)
    msd = jnp.mean(sq_displacement)
    
    expected_msd = 6 * D_coeff * t_end
    
    print(f"MSD: Sim={msd:.3f}, Expected={expected_msd:.3f}")
    
    # Check simple error bound (standard error of mean is roughly sigma/sqrt(N))
    # Relative error should be small.
    rel_error = jnp.abs(msd - expected_msd) / expected_msd
    print(f"Relative Error: {rel_error:.2%}")
    
    # Allow 10% deviation for stochastic test
    assert rel_error < 0.1, "Diffusion MSD failed"
    print("Diffusion SDE Test Passed")

if __name__ == "__main__":
    test_bloch_simulator_relaxation()
    test_diffusion_sde_msd()
