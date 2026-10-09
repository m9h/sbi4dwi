import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
from typing import NamedTuple, Any, Tuple, Optional

# We assume PulseInterpreter is in the same package
try:
    from .pulse_interpreter import PulseInterpreter
except ImportError:
    from pulse_interpreter import PulseInterpreter

class BlochSystem(diffrax.ODETerm):
    """
    Bloch-Torrey Equation as a Diffrax ODETerm.
    
    dM/dt = M x Omega - R(M - M0)
    
    where Omega = [0, 0, 2*pi * (G(t) . r)] (assuming standard Gz-based encoding)
    """
    t1: float
    t2: float
    gamma: float

    def __init__(self, t1: float, t2: float, gamma: float = 2 * jnp.pi * 42.57e6):
        self.t1 = t1
        self.t2 = t2
        self.gamma = gamma

    def vector_field(self, t, y, args):
        """
        Computes the time derivative of Magnetization M.
        """
        interpreter, r = args
        
        # 1. Get Gradients (Hz/m)
        grads_hz = interpreter.control.evaluate(t)
        
        # 2. Compute Omega (rad/s)
        # Pulseq gradients are in Hz/m.
        # Frequency offset f = G_hz . r (Hz)
        # Omega = 2 * pi * f (rad/s)
        # Note: We ignore 'self.gamma' here because the input is ALREADY in Hz (gamma coded).
        # Standard MRI assumption: Gradients modulate Bz field.
        
        f_hz = jnp.dot(grads_hz, r)
        omega_z = 2 * jnp.pi * f_hz
        
        # Precession vector (along Z)
        omega_vec = jnp.array([0.0, 0.0, omega_z])
        
        dM_precess = jnp.cross(y, omega_vec)
        
        # 3. Relaxation
        Mx, My, Mz = y
        dMx_relax = -Mx / self.t2
        dMy_relax = -My / self.t2
        dMz_relax = -(Mz - 1.0) / self.t1
        
        dM_relax = jnp.array([dMx_relax, dMy_relax, dMz_relax])
        
        return dM_precess + dM_relax

def simulate_magnetization(t1: float, t2: float, pulse_interpreter: PulseInterpreter, r: Optional[jax.Array] = None):
    """
    Solves the Bloch equation.
    
    Args:
        t1, t2: Relaxation times (s)
        pulse_interpreter: Control
        r: Position vector (m). Defaults to [1mm, 0, 0].
    """
    if r is None:
        r = jnp.array([0.001, 0.0, 0.0]) # 1 mm isotropic
    
    bloch = BlochSystem(t1=t1, t2=t2)
    solver = diffrax.Tsit5()
    
    t0 = 0.0
    t1_end = pulse_interpreter.t_grid[-1]
    y0 = jnp.array([0.0, 0.0, 1.0])
    args = (pulse_interpreter, r)
    
    # Heuristic step size
    dt0 = 1e-4
    
    saveat = diffrax.SaveAt(t1=True)
    adjoint = diffrax.RecursiveCheckpointAdjoint()
    
    sol = diffrax.diffeqsolve(
        bloch,
        solver,
        t0=t0,
        t1=t1_end,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        adjoint=adjoint,
        max_steps=1000000
    )
    
    return sol.ys[-1]

if __name__ == "__main__":
    # Sanity Check
    import os
    import pypulseq as pp
    
    print("--- Bloch Simulator Sanity Check (Fixed Units) ---")
    
    # 1. Create Mock
    seq_path = "test_bloch.seq"
    if not os.path.exists(seq_path):
        system = pp.Opts(max_grad=30, grad_unit='mT/m', max_slew=100, slew_unit='T/m/s')
        seq = pp.Sequence(system)
        # Use a strong kick to ensure we see something if r were big?
        # Actually standard gradients.
        gx = pp.make_trapezoid(channel='x', flat_area=100, flat_time=10e-3, system=system)
        seq.add_block(gx)
        seq.write(seq_path)
    
    pi = PulseInterpreter(seq_path)
    
    # 2. Simulate
    # Position at 1mm x
    r_test = jnp.array([0.001, 0.0, 0.0]) 
    
    print(f"Time span: {pi.t_grid[-1]} s")
    print(f"Position: {r_test} m")
    
    M_final = simulate_magnetization(1.0, 0.1, pi, r=r_test)
    
    print(f"Final Magnetization: {M_final}")
    print("--- Check Complete ---")
