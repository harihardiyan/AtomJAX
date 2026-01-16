import jax.numpy as jnp

def simpson_integral(u_interior: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """Performs numerical integration using Simpson's 1/3 rule.
    
    Args:
        u_interior: Wavefunction at interior grid points.
        h: Grid spacing.
        
    Returns:
        The integral of u^2 over the radial coordinate.
    """
    # Reconstruct the full wavefunction by adding zero boundaries at r=0 and r=max
    # If interior is 1999 pts, u_full becomes 2001 pts (Odd number required for Simpson)
    u_full = jnp.concatenate([jnp.array([0.0]), u_interior, jnp.array([0.0])])
    n = u_full.shape[0]
    
    # Simpson's rule weights: [1, 4, 2, 4, ..., 2, 4, 1]
    weights = jnp.ones(n)
    weights = weights.at[1:-1:2].set(4.0)
    weights = weights.at[2:-2:2].set(2.0)
    
    # Integrate u^2 (Normalization integral)
    return (h / 3.0) * jnp.sum(weights * (u_full**2))
