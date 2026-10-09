# 2b. Equinox Architecture Strategy

Date: 2026-01-20

## Status

Accepted

## Context

In standard `dmipy`, models are implemented as standard Python classes. While intuitive, this approach is fundamentally incompatible with JAX's Just-In-Time (JIT) compilation mechanism.

When `jax.jit` traces a function, it treats standard Python objects as static (compile-time) constants if they are not invalid, or fails if they contain mutable state that JAX cannot track. This leads to two major issues:
1.  **Re-compilation**: Every new instance of a class triggers a re-compile, destroying performance.
2.  **Tracer Errors**: `jax.jit` cannot see inside standard Python lists or dictionaries if they are not registered as Pytrees, failing when gradients are required.

We need a way to treat our physical models (Balls, Sticks, Zeppelins) as "data" that JAX can traverse, differentiate, and vectorize freely.

## Decision

We wrap all physical models and acquisition schemas in `equinox.Module`.

### The Pattern
Instead of:
```python
class Cylinder:
    def __init__(self, mu):
        self.mu = mu
```

We use:
```python
import equinox as eqx

class Cylinder(eqx.Module):
    mu: float

    def __call__(self, ...):
        ...
```

### Why Equinox?
`equinox` automatically registers any class inheriting from `eqx.Module` as a JAX Pytree. This means:
1.  **Pytree Safety**: JAX natively understands the structure. It can differentiate with respect to `self.mu` without any extra boilerplate.
2.  **JIT Compatibility**: Instances can be passed into JIT-compiled functions without triggering re-compilation (provided the structure is stable).
3.  **Vmap-ability**: We can `jax.vmap` over lists of Modules (e.g., a population of Cylinders) to simulate them in parallel.

## Consequences

*   **Positive**: We achieve near-native C++ performance for complex tissue models via XLA compilation.
*   **Positive**: Models are purely functional and immutable, reducing state-related bugs.
*   **Negative**: We lose standard `__setattr__` mutability. To update a parameter, we must use `eqx.tree_at` to create a new instance with the changed value (copy-on-write).
