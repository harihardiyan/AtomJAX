import jax
import jax.numpy as jnp
from atomjax.solvers import solve_hydrogen_state
from atomjax.observables import calculate_observables

def run_analysis():
    z_min, z_max = 1, 20
    n, ell = 2, 1
    zs = jnp.arange(z_min, z_max + 1, dtype=jnp.float64)

    @jax.jit
    def analyze_single_z(z):
        e_num, u, grid = solve_hydrogen_state(z, n, ell)
        obs = calculate_observables(u, grid.r, grid.h, z, n, ell)
        
        # Mapping results to match original script's array structure
        return jnp.array([
            z, float(n), float(ell), e_num, 
            -0.5 * (z**2) / (n**2), # E_ana
            obs["norm"], obs["rinv_num"], obs["rinv_ana"],
            obs["rinv3_num"], obs["rinv3_ana"]
        ])

    data = jax.vmap(analyze_single_z)(zs)
    print("Z-Batch Analysis Result (Z, n, l, E_num, E_ana, Norm, <1/r>_num, <1/r>_ana...):")
    print(data)

if __name__ == "__main__":
    run_analysis()
