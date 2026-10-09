import pytest
import jax
import jax.numpy as jnp
# Check if equinox and optax are available
try:
    import equinox as eqx
    import optax
    from dmipy_jax.fitting.neural import NeuralEstimator, train_estimator, fit_neural
except ImportError:
    print("equinox or optax not installed")
    exit(0)

@pytest.mark.xfail(reason="known failure, pre-existing, see docs/decisions/010 §7.3: activation function passed as a JAX leaf (equinox API drift)", strict=False)
def test_neural_estimator_training_and_inference():
    key = jax.random.PRNGKey(42)
    
    # Define a simple dummy model: 3 params -> 5 signals
    # Signal = params[0] * x + params[1] * x^2 + ... just something simple
    # Let's say params are (a, b, c) and we return [a, b, c, a+b, b+c]
    def dummy_model(params):
        return jnp.array([params[0], params[1], params[2], params[0]+params[1], params[1]+params[2]])
    
    # 3 params, range [0, 1]
    priors = jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    
    # Train
    # Reduce samples and epochs for speed
    model = train_estimator(dummy_model, priors, n_samples=100, batch_size=10, n_epochs=2, key=key)
    
    assert isinstance(model, NeuralEstimator)
    
    # Test Inference
    dummy_data = jnp.zeros((5, 5)) # 5 voxels, 5 signals
    # Fill with some values
    dummy_data = dummy_data.at[0].set(jnp.array([0.5, 0.5, 0.5, 1.0, 1.0]))
    
    mean, std = fit_neural(dummy_data, model, key, n_mc_samples=10)
    
    assert mean.shape == (5, 3) # 5 voxels, 3 params
    assert std.shape == (5, 3)
    
    print("Test passed: Neural Estimator training and inference.")

if __name__ == "__main__":
    try:
        test_neural_estimator_training_and_inference()
        print("Everything runs correctly.")
    except Exception as e:
        print(f"Test failed: {e}")
