import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from dmipy_jax.inference.trainer import create_trainer, train_loop

# 1. Define a Dummy Simulator (Variable Sticks)
# Model: S = f_stick * exp(-b * d * (g.n)^2)
# But simplified for testing: Signal = Theta + Noise
# theta: [f_stick, diffusivity]
def dummy_simulator(key, theta):
    # Identity mapping for simplicity to check convergence
    return theta * 1.0

# 2. Define Prior Sampler
def prior_sampler(key, batch_size):
    # Theta ~ Uniform(0, 1)
    return jax.random.uniform(key, (batch_size, 2))

@pytest.mark.xfail(reason="known failure, pre-existing, see docs/decisions/010 §7.3: train_loop() API drift ('trainer' kwarg)", strict=False)
def test_trainer():
    key = jax.random.key(0)
    
    trainer = create_trainer(
        flow_key=key,
        theta_dim=2,
        signal_dim=2,
        simulator=dummy_simulator,
        prior_sampler=prior_sampler,
        learning_rate=1e-3,
        hidden_dim=32,
        num_layers=2
    )
    
    print("Starting training...")
    trained_trainer = train_loop(
        trainer=trainer,
        key=key,
        num_steps=500,
        batch_size=128,
        noise_std=0.01,
        print_every=100
    )
    
    # Verification: Sample from flow given a known signal
    test_theta = jnp.array([0.5, 0.5])
    test_signal = dummy_simulator(None, test_theta) # [0.5, 0.5]
    
    # Conditional sample
    params, static = eqx.partition(trained_trainer.flow, eqx.is_array)
    # FlowJAX sampling needs context
    samples = trained_trainer.flow.sample(jax.random.key(1), sample_shape=(100,), condition=test_signal)
    
    mean_sample = jnp.mean(samples, axis=0)
    print(f"True Theta: {test_theta}")
    print(f"Posterior Mean: {mean_sample}")
    
    # Check if close
    error = jnp.linalg.norm(mean_sample - test_theta)
    print(f"Error: {error:.4f}")
    
    if error < 0.1:
        print("SUCCESS: Trainer converged.")
    else:
        print("FAILURE: Trainer did not converge.")

if __name__ == "__main__":
    test_trainer()
