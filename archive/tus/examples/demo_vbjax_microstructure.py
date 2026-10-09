import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.biophysics.network.vbjax_wrapper import VBJaxNetwork
import matplotlib.pyplot as plt

def main():
    print("Running VBJAX Microstructure Integration Demo...")
    
    # 1. Mock Data Generation
    n_regions = 10
    print(f"Generating synthetic brain network with {n_regions} regions.")
    
    # Random structural connectivity (weights)
    # Dense random graph for demo
    np.random.seed(42)
    weights = np.random.rand(n_regions, n_regions)
    weights = (weights + weights.T) / 2.0 # Symmetric
    np.fill_diagonal(weights, 0)
    
    # Random delays (e.g., from distance/velocity)
    # In a real scenario, this comes from ConnectomeMapper.map_microstructure_to_velocity
    delays = np.random.uniform(2.0, 15.0, size=(n_regions, n_regions)) # ms
    np.fill_diagonal(delays, 0)
    
    # Convert to JAX
    weights_jax = jnp.array(weights)
    delays_jax = jnp.array(delays)
    
    # 2. Instantiate Wrapper
    print("Initializing VBJaxNetwork...")
    net = VBJaxNetwork(weights=weights_jax, delays=delays_jax, dt=0.1)
    
    # 3. Simulate
    t_max = 500.0 # ms
    print(f"Simulating for {t_max} ms...")
    
    try:
        times, states = net.simulate(t_max=t_max)
        print("Simulation successful.")
        print(f"Output shapes - Times: {times.shape}, States: {states.shape}")
        
        # Verify stats
        print(f"Mean State Activity: {jnp.mean(states):.4f}")
        print(f"Max State Activity: {jnp.max(states):.4f}")
        
        # Simple plot (if interactive, but here we just print)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
