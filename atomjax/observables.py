import jax.numpy as jnp
from atomjax.integration import simpson_rule

def calculate_observables(u: jnp.ndarray, r: jnp.ndarray, h: jnp.ndarray, z: float, n: int, ell: int):
    """Calculates physical observables and analytical comparisons.
    
    Matches the original logic for 2p state analytical comparisons.
    """
    # 1. Normalization
    norm_factor = simpson_rule(u**2, h)
    u_norm = u / jnp.sqrt(norm_factor)
    
    # 2. Expectation value <1/r>
    rinv_integrand = (u_norm**2) * (1.0 / r)
    rinv_num = simpson_rule(rinv_integrand, h)
    
    # 3. Expectation value <1/r^3>
    rinv3_integrand = (u_norm**2) * (1.0 / r**3)
    rinv3_num = simpson_rule(rinv3_integrand, h)
    
    # 4. Analytical Comparison (Specific to n=2, l=1 as per original script)
    rinv_ana = jnp.where((n == 2) & (ell == 1), z / 4.0, jnp.nan)
    rinv3_ana = jnp.where((n == 2) & (ell == 1), (z**3) / 24.0, jnp.nan)
    
    return {
        "rinv_num": rinv_num,
        "rinv_ana": rinv_ana,
        "rinv3_num": rinv3_num,
        "rinv3_ana": rinv3_ana,
        "norm": norm_factor
    }
