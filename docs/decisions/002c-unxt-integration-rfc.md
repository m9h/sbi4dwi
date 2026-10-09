# 2c. RFC: Integration of `unxt` for Physical Unit Safety

## 1. Motivation
Current implementations of `dmipy-jax` rely on implicit units (e.g., $s/mm^2$, $\mu m$, $ms$). This leads to:
- **Ambiguity**: Is `delta` in seconds or milliseconds?
- **Manual Conversions**: `q_mag = q_mag * 1e3` (mm⁻¹ to m⁻¹) or `1e-9` scaling for diffusivities.
- **Silent Errors**: Adding a diffusivity ($m^2/s$) to a length ($m$) raises no error in raw JAX.

`unxt` (2025) provides zero-overhead unit safety within the JAX graph, allowing us to define quantities explicitly.

## 2. Proposed Design

### 2.1 Dependencies
Add `unxt` to `pyproject.toml`:
```toml
dependencies = [
    "unxt>=0.1.0",
    ...
]
```

### 2.2 Constants (`dmipy_jax.constants`)
Replace raw floats with `unxt.Quantity`.

```python
import unxt as u

# Gyromagnetic ratio for Hydrogen-1
GYRO_MAGNETIC_RATIO = u.Quantity(267.513e6, "rad * s^-1 * T^-1")
```

### 2.3 Acquisition Scheme
The `JaxAcquisition` class should accept and store quantities.

```python
@dataclass
class JaxAcquisition:
    bvalues: u.Quantity["time / length^2"]  # s/mm² or s/m²
    gradient_directions: jnp.ndarray        # Dimensionless (unit vector)
    delta: u.Quantity["time"]               # s or ms
    Delta: u.Quantity["time"]               # s or ms
    
    def __post_init__(self):
        # Enforce SI units internally ? 
        # Or keep as Quantities (unxt handles conversion)
        pass

    @property
    def qvalues(self) -> u.Quantity["1/length"]:
        # unxt handles the math and unit propagation
        tau = self.Delta - self.delta / 3.0
        q_mag = jnp.sqrt(self.bvalues / tau) / (2 * jnp.pi)
        return q_mag
```

### 2.4 Signal Models
Models will define their parameters with expected units. `equinox` fields remains `Any` or `u.Quantity`.

**Example: C1Stick (Updated)**

```python
class C1Stick(eqx.Module):
    lambda_par: u.Quantity["length^2 / time"]

    def __init__(self, lambda_par: u.Quantity = None):
        self.lambda_par = lambda_par

    def __call__(self, acquisition: JaxAcquisition, **kwargs):
        # Extract quantities
        b = acquisition.bvalues
        g = acquisition.gradient_directions
        d_par = kwargs.get('lambda_par', self.lambda_par)

        # Computation (Units track automatically)
        # exponent = -b * d_par * (g . mu)^2
        # Units: [s/m^2] * [m^2/s] * [1] = [1] (Dimensionless)
        
        exponent = -b * d_par * jnp.dot(g, kwargs['mu'])**2
        
        # unxt.exp requires dimensionless input
        return u.exp(exponent)
```

## 3. Migration Strategy

### Phase 1: Dual Support (Transition)
- Update `JaxAcquisition` to allow *both* floats (legacy) and `Quantity`.
- If floats are passed, assume SI units (m, s) or existing convention (mm for bvals?). *Decision: Enforce SI (m, s) for unxt usage.*

### Phase 2: Internal Usage
- Port `constants.py`.
- Port `dmipy_jax/signal_models/` one by one (starting with `cylinder_models.py` and `gaussian_models.py`).

### Phase 3: Public API
- Expose `unxt` types in type hints.
- Recommend users pass `u.Quantity(1000, "s/mm^2")` instead of `1000`.

## 4. `system` vs `pvec` integration
`unxt` often works with a system of units. We will standardise on **SI** for internal calculations to avoid precision issues with `mm` vs `m` if not carefully handled, although `unxt` should handle scaling factors correctly.

## 5. Benefits
- **Self-Documenting Code**: `lambda_par: Quantity["m^2/s"]` is clearer than `lambda_par: float`.
- **Bug Prevention**: `q * radius` will fail if units don't cancel to dimensionless (radians).
- **JAX Compatible**: `unxt` is designed to be JIT-compatible and essentially evaporates at runtime.
