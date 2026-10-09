"""
Skull EIT Inversion Module.
Estimates skull conductivity map from electrode measurements.
Uses ConductivityPDELoss and Pseudo-CT Priors.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from dmipy_jax.biophysics.conductivity_pinn import ConductivityPDELoss

class EITModel(eqx.Module):
    """
    Joint model for EIT Inversion:
    1. Potential V(x) network (Field).
    2. Conductivity sigma(x) network (Parameter).
    """
    v_net: eqx.nn.MLP
    sigma_net: eqx.nn.MLP
    
    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        # Potential V: x,y,z -> scalar V
        self.v_net = eqx.nn.MLP(in_size=3, out_size=1, width_size=64, depth=3, key=k1)
        # Conductivity Sigma: x,y,z -> scalar Sigma
        # We model log(sigma) or sigma directly. Constrained > 0.
        self.sigma_net = eqx.nn.MLP(in_size=3, out_size=1, width_size=32, depth=2, key=k2)
        
    def __call__(self, x):
        """Returns Potential V(x)"""
        return self.v_net(x)[0]
    
    def get_sigma(self, x):
        """Returns Conductivity sigma(x). Enforces positivity."""
        out = self.sigma_net(x)[0]
        # Softplus to ensure positive conductivity
        return jax.nn.softplus(out) + 1e-6

class SkullEITInversion:
    """
    Manager for the EIT Inversion problem.
    """
    def __init__(self, electrodes, measurements, sigma_prior_fn=None):
        """
        Args:
            electrodes: (N, 3) positions.
            measurements: (N,) measured potentials.
            sigma_prior_fn: Function or Model giving prior estimate of sigma(x) (e.g. from Pseudo-CT).
        """
        self.electrodes = electrodes
        self.measurements = measurements
        self.sigma_prior_fn = sigma_prior_fn
        
    def evaluate(self, model, batch):
        """
        Composite EIT Loss:
        1. PDE Residual: Div(sigma Grad V) = -I
        2. Data Mismatch: V(electrodes) = measurements
        3. Regularization: sigma ~ sigma_prior
        """
        
        # 1. PDE Loss
        # We define a helper for ConductivityPDELoss that wraps the EITModel
        # ConductivityPDELoss needs 'u' and 'sigma_fn'
        # With EITModel, 'model(x)' is 'u(x)', and 'model.get_sigma(x)' is 'sigma(x)'
        
        # We need to construct conductivity_loss on the fly or keep it stored?
        # Ideally stored. But evaluate() needs access to it.
        # Let's assume we instantiate ConductivityPDELoss inside the training loop 
        # or pass it. For now, we implement the residual explicitly here to reuse components.
        
        # Actually, let's use the ConductivityPDELoss class logic.
        # We'll create a dummy 'u' wrapper that calls model(x)
        # And 'sigma_fn' wrapper that calls model.get_sigma(x)
        
        # PDE Points
        pde_points = batch['pde_domain']
        
        # Manual PDE implementation for EIT synergy (to link V and Sigma from same object)
        def current_density_fn(x):
            v_fn = lambda p: model(p)
            grad_V = jax.grad(v_fn)(x)
            sigma = model.get_sigma(x)
            return sigma * grad_V
            
        def div_J_fn(x):
            return jnp.trace(jax.jacfwd(current_density_fn)(x))
            
        # vmap over batch
        div_J = jax.vmap(div_J_fn)(pde_points)
        
        # Source term handling? 
        # Typically source is injected at electrodes (Boundary Condition) 
        # or as volumetric source. For EIT, usually Boundary Current Injection.
        # So internal source is 0. Div(J) = 0 inside.
        pde_residual = jnp.mean(div_J**2)
        
        # 2. Data Mismatch
        # Evaluate V at electrodes
        # electrodes (N, 3)
        v_pred = jax.vmap(model)(self.electrodes)
        data_loss = jnp.mean((v_pred - self.measurements)**2)
        
        # 3. Prior Regularization (Pseudo-CT Synergy)
        reg_loss = 0.0
        if self.sigma_prior_fn is not None:
            sigma_pred = jax.vmap(model.get_sigma)(pde_points)
            sigma_prior = jax.vmap(self.sigma_prior_fn)(pde_points)
            # Penalize deviation
            reg_loss = jnp.mean((sigma_pred - sigma_prior)**2)
            
        # Total Loss
        # Weights should be tunable
        lambda_data = 100.0
        lambda_reg = 1.0
        
        return pde_residual + lambda_data * data_loss + lambda_reg * reg_loss
