import jax.numpy as jnp
from atomjax.integration import simpson_integral

def calculate_expectation_rinv(u: jnp.ndarray, r: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """Calculates the expectation value <1/r>."""
    # Normalize the wavefunction first
    norm_sq = simpson_integral(u, h)
    u_norm = u / jnp.sqrt(norm_sq)
    
    # Compute the integrand for <1/r>
    integrand = (u_norm**2) / r
    
    # We reuse simpson_integral logic by passing the weighted integrand
    # Note: simpson_integral expects u, so we pass sqrt(integrand) effectively
    u_eff = jnp.sqrt(jnp.maximum(integrand, 0.0))
    return simpson_integral(u_eff, h)
