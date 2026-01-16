import jax.numpy as jnp

def simpson_rule(y_interior: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """General Simpson's 1/3 rule for a function zeroed at boundaries.
    
    Args:
        y_interior: The values of the integrand at interior grid points.
        h: Grid spacing.
    """
    # Standardize to odd number of points (e.g., 1999 interior + 2 boundaries = 2001)
    y_full = jnp.concatenate([jnp.array([0.0]), y_interior, jnp.array([0.0])])
    n = y_full.shape[0]
    
    weights = jnp.ones(n)
    weights = weights.at[1:-1:2].set(4.0)
    weights = weights.at[2:-2:2].set(2.0)
    
    return (h / 3.0) * jnp.sum(weights * y_full)
