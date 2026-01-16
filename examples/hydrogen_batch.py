import jax
import jax.numpy as jnp
from atomjax.solvers import solve_hydrogen_state
from atomjax.observables import calculate_observables

def main():
    print("Running AtomJAX Batch Solver...")
    
    # Vectorizing over Z (Atomic Charges from 1 to 10)
    zs = jnp.arange(1.0, 11.0)
    
    @jax.jit
    def solve_single(z):
        e, u, grid = solve_hydrogen_state(z, n=2, ell=1)
        obs = calculate_observables(u, grid.r, grid.h, z)
        return e, obs['rinv']

    # Batch execution using vmap
    energies, rinvs = jax.vmap(solve_single)(zs)
    
    for i, z in enumerate(zs):
        print(f"Z={z:.0f} | Energy: {energies[i]:.6f} Ha | <1/r>: {rinvs[i]:.6f}")

if __name__ == "__main__":
    main()
