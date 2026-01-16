import jax.numpy as jnp
from typing import Callable

def simpson_integral(y: jnp.ndarray, h: float) -> float:
    """Standard Simpson's 1/3 rule for a grid with 0 at boundaries."""
    # Ensure we include the zero boundaries for the reduced wavefunction u(r)
    y_full = jnp.concatenate([jnp.array([0.0]), y, jnp.array([0.0])])
    n = y_full.shape[0]
    weights = jnp.ones(n)
    weights = weights.at[1:-1:2].set(4.0)
    weights = weights.at[2:-2:2].set(2.0)
    return (h / 3.0) * jnp.sum(weights * y_full)

def compute_expectation_value(u: jnp.ndarray, r: jnp.ndarray, h: float, 
                              operator: Callable[[jnp.ndarray], jnp.ndarray]) -> float:
    """Computes <u|Op|u> using Simpson integration."""
    y = (u**2) * operator(r)
    return simpson_integral(y, h)
