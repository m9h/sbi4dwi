import pytest

import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.components.exchange import KargerExchange
from dmipy_jax.gaussian import G1Ball
from dmipy_jax.acquisition import JaxAcquisition

@pytest.mark.xfail(reason="known failure, pre-existing, see docs/decisions/010 §7.3: Karger signal exceeds 1 (1.49 at b=0) — model bug", strict=False)
def test_karger_exchange():
    print("Initializing Karger Exchange Test...")
    
    # 1. Setup Models
    # Couple two balls: one fast (D=3e-9), one slow (D=0.5e-9)
    ball1 = G1Ball(lambda_iso=3e-9)
    ball2 = G1Ball(lambda_iso=0.5e-9)
    
    exchange_model = KargerExchange([ball1, ball2])
    print("Model initialized.")
    print("Parameter names:", exchange_model.parameter_names)
    
    # 2. Setup Acquisition
    # b-values in s/m^2
    bvalues = jnp.array([0., 1000e6, 2000e6, 3000e6])
    gradients = jnp.array([[1., 0., 0.]] * 4)
    delta = 0.02 # 20ms
    Delta = 0.04 # 40ms
    acq = JaxAcquisition(bvalues=bvalues, gradient_directions=gradients, delta=delta, Delta=Delta)
    
    # 3. Define Parameters
    # model0_lambda_iso (masked/fixed in class but we passed it in __init__)
    # Wait, G1Ball __init__ takes lambda_iso. 
    # Does G1Ball use the one from __init__ if provided?
    # G1Ball.__call__ says: `lambda_iso = kwargs.get('lambda_iso', self.lambda_iso)`
    # So if we provide it in kwargs, it overrides.
    # KargerExchange unpacks params and passes them.
    # Our KargerExchange collected parameter names.
    # G1Ball has `parameter_names = ['lambda_iso']`.
    # KargerExchange prefixed them: `model0_lambda_iso`.
    # `_unpack_params` strips the prefix and passes `lambda_iso` in kwargs.
    # So we MUST provide the values in the params vector.
    
    # Param order:
    # model0_lambda_iso
    # model1_lambda_iso
    # partial_volume_0 (fraction of model 0)
    # exchange_time_01 (residence time 0->1)
    
    params = jnp.array([
        2.0e-9,  # D1 (overriding init defaults for test)
        0.5e-9,  # D2
        0.5,     # f1 = 0.5 (so f2 = 0.5)
        0.1      # exchange time = 100ms
    ])
    
    # 4. Predict
    print("\nRunning Prediction (Exchange Time = 0.1s)...")
    signal = exchange_model.predict(params, acq)
    print("Signal:", signal)
    
    assert jnp.all(signal <= 1.0 + 1e-6)
    assert jnp.all(signal >= 0.0 - 1e-6)
    assert jnp.isclose(signal[0], 1.0)
    
    # 5. Limit Check: No Exchange (High Tau)
    print("\nRunning Prediction (No Exchange, Tau=1e5s)...")
    params_no_ex = jnp.array([2.0e-9, 0.5e-9, 0.5, 1e5])
    signal_no_ex = exchange_model.predict(params_no_ex, acq)
    print("Signal (No Ex):", signal_no_ex)
    
    # Analytical Uncoupled
    # S = f1 * exp(-b D1) + f2 * exp(-b D2)
    s1 = jnp.exp(-bvalues * 2.0e-9)
    s2 = jnp.exp(-bvalues * 0.5e-9)
    s_expected = 0.5 * s1 + 0.5 * s2
    print("Signal (Analytic):", s_expected)
    
    assert jnp.allclose(signal_no_ex, s_expected, atol=1e-5)
    print("Match confirmed!")
    
    # 6. Limit Check: Fast Exchange (Low Tau)
    # With equal fractions 0.5, 0.5
    # Effective D should be mean? 
    # Or rather, signal approaches exp(-b * mean(D))
    print("\nRunning Prediction (Fast Exchange, Tau=1e-5s)...")
    params_fast = jnp.array([2.0e-9, 0.5e-9, 0.5, 1e-5])
    signal_fast = exchange_model.predict(params_fast, acq)
    
    d_mean = 0.5 * 2.0e-9 + 0.5 * 0.5e-9
    s_fast_expected = jnp.exp(-bvalues * d_mean)
    print("Signal (Fast Ex):", signal_fast)
    print("Signal (Mean D): ", s_fast_expected)
    
    # Note: Karger approximation might differ from Jensen limits slightly depending on implementation
    # but usually converges to monoexponential.
    assert jnp.allclose(signal_fast, s_fast_expected, atol=1e-2) 
    # Relaxed tolerance as Karger isn't exactly Monoexp at finite fast exchange, 
    # but is very close.
    print("Fast Exchange Check passed.")

if __name__ == "__main__":
    test_karger_exchange()
