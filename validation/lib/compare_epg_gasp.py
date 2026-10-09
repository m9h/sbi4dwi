
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gasp import simulation, responses
import warnings
from dmipy_jax.models.epg import JAXEPG

warnings.filterwarnings("ignore")

def compare_bssfp_profile():
    print("Running GASP vs JAX-EPG comparison for bSSFP...")
    
    # Parameters
    T1 = 1.0  # s
    T2 = 0.1  # s
    TR = 0.005 # 5ms
    alpha_deg = 30.0
    alpha_rad = np.deg2rad(alpha_deg)
    
    # GASP uses specialized parameters.
    # simulate_ssfp signature: (width, height, npcs, TRs, alpha, gradient, ...)
    # It generates a phantom internally.
    # We want just a single pixel profile or a range of off-resonances.
    
    # GASP "gradient" parameter usually corresponds to phase twist across the image
    # For a full 0-2pi profile (one band), gradient should be 2pi.
    
    width = 256
    height = 1
    gradient = 2 * np.pi
    
    # Create SSFPParams wrapper
    # SSFPParams(length, alpha, TRs, pcs)
    # length: number of experiments/phase cycles?
    # In simulate_ssfp_simple, npcs = params.length.
    # It loops ii in range(npcs).
    # We want 1 experiment.
    
    # We pass lists.
    # We want 1 configuration.
    params = simulation.SSFPParams(
        length=1,
        alpha=[alpha_rad],
        TRs=[TR],
        pcs=[0.0] # 0 phase cycle
    )
    
    print("  Simulating GASP (simulate_ssfp_simple)...")
    M_gasp = simulation.simulate_ssfp_simple(
        width=width,
        height=height,
        T1=T1,     # Simple takes T1/T2 directly
        T2=T2,
        params=params, 
        minTR=TR,
        gradient=gradient
    )
    
    # M_gasp shape is (height, width, npcs) -> (1, 256, 1)
    profile_gasp = M_gasp[0, :, 0]
    
    profile_gasp = np.abs(profile_gasp)
    
    # Run JAX EPG Simulation
    # Match GASP range: [-gradient, gradient]
    # GASP uses np.linspace default (endpoint=True)
    
    betas = np.linspace(-gradient, gradient, width, endpoint=True) # Off-resonance per TR
    
    print("  Simulating JAX-EPG...")
    
    # Vectorize JAX EPG over off-resonance
    @jax.jit
    def run_epg(beta):
        # simulate_bssfp(T1, T2, TR, alpha, off_resonance=...)
        # T1, T2 in seconds?
        # JAXEPG implementation usually takes time units consistent with TR.
        # If input T1=1000ms, TR=10ms.
        # Here T1=1.0s, TR=0.005s.
        return JAXEPG.simulate_bssfp(
            T1=T1, 
            T2=T2, 
            TR=TR, 
            alpha=alpha_rad, 
            off_resonance=beta,
            N_pulses=200,
            TE=TR/2.0
        )
        
    runs_epg = jax.vmap(run_epg)
    profile_jax = np.abs(runs_epg(betas))
    
    # Normalize? 
    # GASP might assume M0=1. JAXEPG assumes M0=1.
    
    # Compare
    # Signal may be shifted? GASP gradient 2pi might coincide differently.
    # We might need to roll the GASP profile to match JAX if phase definition differs (center freq).
    
    # Find peak matching
    # bSSFP peak is at 0 off-resonance (on resonance).
    idx_jax_max = np.argmax(profile_jax)
    idx_gasp_max = np.argmax(profile_gasp)
    
    print(f"  JAX Max Index: {idx_jax_max}")
    print(f"  GASP Max Index: {idx_gasp_max}")
    
    rolled_gasp = np.roll(profile_gasp, idx_jax_max - idx_gasp_max)
    
    # Metrics
    mse = np.mean((profile_jax - rolled_gasp)**2)
    max_diff = np.max(np.abs(profile_jax - rolled_gasp))
    
    print(f"  MSE: {mse:.6e}")
    print(f"  Max Diff: {max_diff:.6e}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(profile_jax, label='JAX EPG', linewidth=2)
    plt.plot(rolled_gasp, '--', label='GASP-SSFP', linewidth=2)
    plt.title(f"bSSFP Profile Comparison (MSE={mse:.2e})")
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/mhough/.gemini/antigravity/brain/66baa948-c273-484a-8b92-56418bbbdd83/bssfp_comparison.png')
    print("  Plot saved to bssfp_comparison.png")
    
    if max_diff < 1e-2: # Relaxed tolerance due to different shifting/phase defs
        print("PASS: Profiles match within tolerance.")
    else:
        print("FAIL: Profiles do not match closely.")

if __name__ == "__main__":
    compare_bssfp_profile()
