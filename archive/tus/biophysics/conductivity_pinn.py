"""
Conductivity PDE Loss for Physics-Informed Neural Networks (PINNs).
Standardizes PDE operators for the Poisson equation: Div(sigma * Grad(V)) = -I_source.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

try:
    import jinns
    from jinns.loss import PDELoss
except ImportError:
    # Fallback for environments where jinns is not installed
    class PDELoss:
        def __init__(self, *args, **kwargs):
            pass

class ConductivityPDELoss(PDELoss):
    """
    Computes the PDE residual for the static current equation:
    Div(sigma(x) * Grad(V(x))) = -I_source(x)
    
    Attributes:
        sigma_fn: Function sigma(x) -> scalar or tensor conductivity.
                  Can be a Neural Network model (Equinox module).
        source_fn: Function source(x) -> scalar current density.
    """
    def __init__(
        self,
        u,
        geometry_boundaries,
        loss_weights,
        grad_fn,
        divergence_fn,
        sigma_fn,
        source_fn=None,
        name="ConductivityPDELoss"
    ):
        """
        Args:
            u: The PINN model function u(x, params). Computes Potential V.
            grad_fn: jinns.loss.grad or compatible.
            divergence_fn: jinns.loss.divergence or compatible.
            sigma_fn: Conductivity function sigma(x). Returns scalar or (3,3) tensor.
                      If optimizing sigma (EIT), this is the model being learned.
            source_fn: Source term I(x). Returns scalar.
        """
        super().__init__(name=name)
        self.u = u
        self.geometry = geometry_boundaries
        self.loss_weights = loss_weights
        self.grad_fn = grad_fn # Computes Grad(V)
        self.divergence_fn = divergence_fn # Computes Div(Vector)
        self.sigma_fn = sigma_fn
        self.source_fn = source_fn
        
    def evaluate(self, model, batch):
        """
        Main evaluation wrapper.
        model: The object containing parameters.
               For Forward Problem: model contains V parameters. sigma_fn is fixed.
               For Inverse Problem (EIT): model might contain both V and Sigma parameters.
        """
        # Batch unpacking depends on JINNS or custom trainer
        # We assume batch['pde_domain'] exists
        if 'pde_domain' in batch:
            pde_loss = self.pde_residual(model, batch['pde_domain'])
        else:
            pde_loss = 0.0
            
        # Boundary conditions would be handled here or via separate BC losses in Jinns
        return self.loss_weights.get('pde', 1.0) * pde_loss

    def pde_residual(self, model, points):
        """
        Computes |Div(sigma Grad V) + I|^2
        points: (batch_size, 3) -> [x, y, z] (Assuming static problem, no t)
        """
        # 1. Compute Gradient of Potential: Grad(V)
        # self.grad_fn(self.u) returns function that takes (model, x) -> (3,)
        # Note: jinns.loss.grad typically vmaps internally or expects vmapped Model?
        # If we use our GB200Trainer, 'model' is an equinox Module.
        # 'self.u' might be 'model' itself or a wrapper?
        
        # If running within JINNS, self.u is typically passed in init.
        # But if we use Equinox, 'model' passed to evaluate IS 'u'.
        # Let's assume 'model' implements the call for V.
        
        # We need to construct the Current Density Vector field J = sigma * Grad V
        # The divergence operator needs a function f(x) -> vector
        
        def current_density_fn(x):
            # x is a single point (3,) usually for autograd, or batched if vmap
            # JINNS operators usually expect function of (t, x) or x
            
            # Evaluate Grad V at x
            # We need to compute gradients w.r.t x.
            # model(x) -> V
            
            # We create a wrapper to differentiate w.r.t x using the CURRENT model params
            def v_fn(x_point):
                return model(x_point)
                
            grad_V = jax.grad(v_fn)(x) # (3,)
            
            # Evaluate Sigma at x
            # If sigma_fn is an Equinox model (Inverse), we assume it's callable
            # If sigma_fn is fixed function, just call it
            if isinstance(self.sigma_fn, eqx.Module):
                # If sigma is part of the optimized model, it might be inside 'model'?
                # For standard EIT, we usually invert for sigma. 
                # Let's assume self.sigma_fn IS the sigma model.
                sigma_val = self.sigma_fn(x)
            else:
                sigma_val = self.sigma_fn(x)
                
            # Compute J = -sigma * Grad V (Ohm's Law J = sigma E = -sigma Grad V)
            # Typically PDE is Div(sigma Grad V) = -I.
            # So term is sigma * Grad V.
            return sigma_val * grad_V
            
        # 2. Compute Divergence of Current Density: Div(J)
        # divergence_fn differentiates the vector field w.r.t x
        # Returns scalar
        
        # JINNS divergence_fn expects u(params, x). 
        # Here we defined current_density_fn closing over 'model'.
        # We need to apply divergence to this function.
        
        # Manual Divergence via trace of Jacobian or simple loop
        # Div J = dJx/dx + dJy/dy + dJz/dz
        div_J = jnp.trace(jax.jacfwd(current_density_fn)(points)) 
        # jacfwd w.r.t x gives (3,3). Trace is divergence.
        # Note: 'points' here is a single point if using jax.vmap outside?
        # PINN residuals are usually vmapped over the batch.
        
        # Optimizing: pde_residual is usually called insde a vmap in JINNS.
        # So 'points' is (3,).
        
        # 3. Source Term
        if self.source_fn:
            I_source = self.source_fn(points)
        else:
            I_source = 0.0
            
        # Residual: Div(sigma Grad V) + I_source = 0
        res = div_J + I_source 
        
        return jnp.abs(res)**2

