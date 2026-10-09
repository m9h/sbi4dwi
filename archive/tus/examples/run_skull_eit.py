"""
Skull EIT Simulation with Pseudo-CT Synergy.
Demonstrates:
1. 'True' Skull Physics (Pseudo-CT -> Porosity -> Conductivity).
2. Geodesic Sensor Net (GSN) Measurements.
3. EIT Inversion initialized with Pseudo-CT Prior.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dmipy_jax.biophysics.eit import EITModel, SkullEITInversion
from dmipy_jax.biophysics.pseudo_ct import PseudoCTMapper
from dmipy_jax.pinns.sampling import DynamicCollocationSampler
from dmipy_jax.pinns.trainer import GB200PINNTrainer

def main():
    print("Setting up Skull EIT Simulation...")
    
    # 1. Defined Ground Truth
    # Simulate a spherical skull shell
    # R_inner = 0.8, R_outer = 0.9
    def true_conductivity(x):
        r = jnp.linalg.norm(x)
        # Brain, CSF, Scalp: high conductivity ~ 1.0
        # Skull: low conductivity
        is_skull = (r > 0.8) & (r < 0.9)
        return jax.lax.select(is_skull, 0.01, 1.0)
    
    # 2. Simulate Pseudo-CT "Measurement"
    # We observe T1/UTE intensity which gives us a prior
    mapper = PseudoCTMapper(method='babel')
    
    def prior_model_fn(x):
        # Inverse problem: We use Pseudo-CT to guess sigma
        # Simulate MRI: High intensity in skull (UTE)
        r = jnp.linalg.norm(x)
        is_skull = (r > 0.8) & (r < 0.9)
        # UTE Intensity: 1.0 in bone, 0.0 elsewhere (simplified)
        mri_val = jax.lax.select(is_skull, 0.8, 0.1) 
        # Add noise
        return mapper.predict_conductivity(mri_val)

    # 3. Simulate GSN Electrodes
    # Random points on unit sphere
    key = jax.random.PRNGKey(0)
    electrodes = jax.random.normal(key, (64, 3))
    electrodes = electrodes / jnp.linalg.norm(electrodes, axis=1, keepdims=True)
    
    # Simulate Measured Potentials (Forward Pass)
    # For demo, we just assign dummy potentials V = x coordinate (linear field)
    measurements = electrodes[:, 0]
    
    # 4. Setup EIT Optimization
    print("Initializing EIT Solver...")
    model = EITModel(key)
    eit_solver = SkullEITInversion(electrodes, measurements, sigma_prior_fn=prior_model_fn)
    
    # 5. Trainer Setup
    sampler = DynamicCollocationSampler(domain_bounds=((-1.,-1.,-1.), (1.,1.,1.)))
    optimizer = optax.adam(1e-3)
    # Single device mesh
    mesh = Mesh(jax.devices(), ('data',))
    
    trainer = GB200PINNTrainer(eit_solver, sampler, optimizer, mesh)
    
    # 6. Run Training
    state, static = trainer.create_train_state(key, model)
    step_fn = trainer.make_step_fn(static)
    
    n_devices = len(jax.devices())
    key_batch = jax.random.split(key, n_devices)
    key_sharding = NamedSharding(mesh, P('data'))
    key_batch = jax.device_put(key_batch, key_sharding)
    
    print("Running EIT Inversion...")
    for i in range(20):
        state, loss = step_fn(state, key_batch)
        if i % 5 == 0:
            print(f"Step {i}, Loss: {loss}")
            
    print("EIT Simulation Complete. Synergy Operational.")

if __name__ == "__main__":
    main()
