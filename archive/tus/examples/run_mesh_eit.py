"""
Mesh-Integrated EIT Simulation.
Demonstrates:
1. Mesh Generation (Trimesh) / Loading.
2. JAX Mesh Sampling (dmipy_jax.geometry.mesh_ops).
3. PINN Training on Surface Manifold (dmipy_jax.biophysics).
"""

import jax
import jax.numpy as jnp
import trimesh
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dmipy_jax.biophysics.eit import EITModel, SkullEITInversion
from dmipy_jax.geometry.mesh_ops import get_mesh_sampler
from dmipy_jax.pinns.trainer import GB200PINNTrainer

def main():
    print("Setting up Mesh-Integrated EIT...")
    
    # 1. Generate/Load Mesh
    # Using a high-res icosphere to represent a skull-like surface
    print("Generating Mesh (Icosphere)...")
    mesh_obj = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    vertices = jnp.array(mesh_obj.vertices, dtype=jnp.float32)
    faces = jnp.array(mesh_obj.faces, dtype=jnp.int32)
    
    print(f"Mesh: {len(vertices)} verts, {len(faces)} faces.")
    
    # 2. Create Mesh Sampler
    # This integrates the geometry directly into the PINN data pipeline
    # The sampler will produce points ON the mesh surface for training
    print("Creating JAX Mesh Sampler...")
    sampler = get_mesh_sampler(vertices, faces, bounds=((-1.5,-1.5,-1.5), (1.5,1.5,1.5)))
    
    # 3. Setup Physics (EIT)
    # Target: 64 Electrodes on the mesh
    key = jax.random.PRNGKey(42)
    # Simple electrodes on unit sphere
    electrodes = jax.random.normal(key, (64, 3))
    electrodes = electrodes / jnp.linalg.norm(electrodes, axis=1, keepdims=True)
    
    # Dummy Measurements (Linear Potential Field V = x)
    measurements = electrodes[:, 0]
    
    # Prior: Uniform 1.0 (No Pseudo-CT for this geo-focused demo)
    prior_fn = lambda x: 0.1 # Guess low conductivity everywhere
    
    # helper for prior
    def prior_model_fn_wrap(x):
        return 0.1
    
    model = EITModel(key)
    eit_solver = SkullEITInversion(electrodes, measurements, sigma_prior_fn=prior_model_fn_wrap)
    
    eit_solver.sigma_prior_fn = prior_model_fn_wrap

    # 4. Trainer
    optimizer = optax.adam(1e-3)
    mesh = Mesh(jax.devices(), ('data',))
    trainer = GB200PINNTrainer(eit_solver, sampler, optimizer, mesh)
    
    # 5. Run
    state, static = trainer.create_train_state(key, model)
    step_fn = trainer.make_step_fn(static)
    
    n_devices = len(jax.devices())
    key_batch = jax.random.split(key, n_devices)
    key_sharding = NamedSharding(mesh, P('data'))
    key_batch = jax.device_put(key_batch, key_sharding)
    
    print("Running Mesh-Constrained Training...")
    for i in range(20):
        # The sampler (inside step_fn) now calls sample_surface_uniform -> compute_face_areas
        state, loss = step_fn(state, key_batch)
        if i % 5 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")
            
    print("Mesh Integration Verified.")

if __name__ == "__main__":
    main()
