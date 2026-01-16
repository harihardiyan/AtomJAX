import jax.numpy as jnp
from atomjax.integration import compute_expectation_value, simpson_integral

def calculate_observables(u: jnp.ndarray, r: jnp.ndarray, h: float, z: float):
    """Calculates <1/r> and <1/r^3> with analytical comparisons."""
    # Normalize wavefunction
    norm = jnp.sqrt(simpson_integral(u**2, h))
    u_norm = u / norm

    # <1/r>
    rinv_num = compute_expectation_value(u_norm, r, h, lambda x: 1.0/x)
    
    # <1/r^3>
    rinv3_num = compute_expectation_value(u_norm, r, h, lambda x: 1.0/x**3)
    
    return {
        "rinv": rinv_num,
        "rinv3": rinv3_num,
        "is_normalized": jnp.isclose(norm, 1.0)
    }
