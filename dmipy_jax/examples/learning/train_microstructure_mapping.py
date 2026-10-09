
import os
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
from typing import Tuple

# DMIPY-JAX imports
from validation.lib.histology import HistoDataset, HistologySimulator
from dmipy_jax.learning.microstructure_mapping import MicrostructureMapper
from dmipy_jax.signal_models import cylinder_models
from dmipy_jax.core.acquisition import SimpleAcquisitionScheme

def get_virtual_ground_truth() -> Tuple[jax.Array, jax.Array, SimpleAcquisitionScheme]:
    """
    Generates synthetic data and simulates "true" signals using Physics.
    Returns:
        inputs: (N_voxels, N_features) [radius, density]
        targets: (N_voxels, N_bvals) [signals]
        acq: Acquisition scheme
    """
    print("Generating Virtual Ground Truth...")
    # 1. Load Histology Data
    loader = HistoDataset()
    _, histo_gt = loader.load_data()
    
    # Flatten
    radius = histo_gt['radius'].reshape(-1)
    density = histo_gt['density'].reshape(-1)
    
    # 2. Define Acquisition
    bvals = jnp.array([0.0, 1000.0, 1000.0, 2000.0, 3000.0]) # More shells for rich signal
    # Simple directions
    bvecs = jnp.array([
        [0.,0.,0.], 
        [1.,0.,0.], 
        [0.,1.,0.], 
        [0.,0.,1.],
        [1.,1.,0.]
    ])
    # Repeat for shapes match if needed, but SimpleAcquisitionScheme handles it if len matches
    # Ensure standard sizes
    bvals = jnp.tile(bvals, 2)
    bvecs = jnp.tile(bvecs, (2, 1)) 
    
    delta = jnp.full(bvals.shape, 0.01)
    Delta = jnp.full(bvals.shape, 0.03)
    acq = SimpleAcquisitionScheme(bvalues=bvals, gradient_directions=bvecs, delta=delta, Delta=Delta)
    
    # 3. Simulate Physics Signals
    # Using HistologySimulator with PHYSICS model
    phys_model = cylinder_models.RestrictedCylinder()
    phys_sim = HistologySimulator(model=phys_model)
    
    # To use phys_sim, we need to pass full acquisition and ground_truth dict
    # But for vmap efficiency over flattened arrays, let's call model directly in loop/vmap
    # or rely on HistologySimulator if we reshape inputs back to image.
    
    # Let's trust the simulator's efficiency if inputs are images.
    # But here we have flat arrays. 
    # It's easier to create a batch function.
    
    def simulate_voxel(r, d):
        params = {
            "diameter": r * 2 * 1e-6,
            "lambda_par": 1.7e-3,
            "volume_fraction": d,
            "mu": jnp.array([0., 0.]) # Fixed orientation for simplicity
        }
        return phys_model(acq.bvalues, acq.gradient_directions, 
                          big_delta=acq.Delta[0], small_delta=acq.delta[0], 
                          **params)
    
    signals = jax.vmap(simulate_voxel)(radius, density)
    
    # Inputs for learning: [radius, density]
    inputs = jnp.stack([radius, density], axis=-1)
    
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Targets shape: {signals.shape}")
    
    return inputs, signals, acq

def main():
    print("Running Microstructure Mapping Training Demo...")
    
    # 1. Get Data
    inputs, targets, acq = get_virtual_ground_truth()
    
    # Normalize inputs?
    # Radius is ~0.5-3.0, Density ~0.7. Already reasonable scale.
    
    # 2. Initialize Model
    # Input: 2 (radius, density)
    # Output: N_signals (len(acq.bvalues))
    n_in = inputs.shape[-1]
    n_out = targets.shape[-1]
    
    model = MicrostructureMapper(in_size=n_in, out_size=n_out, width_size=64, depth=3)
    
    # 3. Setup Training
    optim = optax.adam(1e-3)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_value_and_grad
    def loss_fn(model, x, y):
        pred = jax.vmap(model)(x)
        return jnp.mean((pred - y)**2)
    
    @eqx.filter_jit
    def step(model, opt_state, x, y):
        loss, grads = loss_fn(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # 4. Training Loop
    n_epochs = 1000
    print(f"Starting training for {n_epochs} epochs...")
    
    loss_history = []
    
    for i in range(n_epochs):
        model, opt_state, loss = step(model, opt_state, inputs, targets)
        if i % 100 == 0:
            print(f"  Epoch {i}: Loss = {loss.item():.6f}")
        loss_history.append(loss.item())
        
    print(f"Final Loss: {loss_history[-1]:.6f}")
    
    # 5. Validation using HistologySimulator with LEARNED model
    print("Validating with HistologySimulator wrapper...")
    
    # Create simulator with learned model
    learned_sim = HistologySimulator(learned_model=model)
    
    # Create dummy GT dict
    # Reusing the loader's default data structure
    loader = HistoDataset()
    _, histo_gt = loader.load_data()
    
    # Predict full image
    s_pred_img = learned_sim(acq, histo_gt)
    
    print(f"  Predicted Image Shape: {s_pred_img.shape}")
    
    # Verify vs Targets (reshaped)
    targets_img = targets.reshape(10, 10, -1)
    mse_val = jnp.mean((s_pred_img - targets_img)**2)
    print(f"  Validation Image MSE: {mse_val:.6f}")
    
    # 6. Inverse Problem (Bonus/Future)
    # The paper mentions "random forest-based microstructure mapping" often implying Signal -> Parameters.
    # Our generic MicrostructureMapper can handle that too (just swap in/out sizes).
    
    # 7. Visualization
    if os.environ.get("DISPLAY"):
        try:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1,3,1)
            plt.plot(loss_history)
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            
            # Show a voxel comparison
            plt.subplot(1,3,2)
            vox_idx = (5, 5)
            plt.plot(targets_img[vox_idx], label="Physics (GT)", marker='o')
            plt.plot(s_pred_img[vox_idx], label="Learned (MLP)", linestyle='--')
            plt.title(f"Voxel {vox_idx} Signal")
            plt.legend()
            
            plt.subplot(1,3,3)
            # Difference map (mean over bvals)
            diff = jnp.mean(jnp.abs(s_pred_img - targets_img), axis=-1)
            plt.imshow(diff)
            plt.title("Mean Absolute Error Map")
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig("demo_learning_results.png")
            print("Saved demo_learning_results.png")
        except Exception as e:
            print(f"Vis failed: {e}")

if __name__ == "__main__":
    main()
