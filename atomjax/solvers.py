import jax
import jax.numpy as jnp
from atomjax.hamiltonian import build_hamiltonian

@jax.jit
def count_nodes(u: jnp.ndarray) -> int:
    """Calculates the number of radial nodes (sign changes) in the wavefunction."""
    s = jnp.sign(u)
    s = jnp.where(s == 0.0, 1.0, s)
    return jnp.sum((s[1:] * s[:-1]) < 0.0)

def solve_hydrogen_state(z, n, ell, n_points=2001, r_max=100.0):
    """Solves for a specific (n, ell) state using nodal selection."""
    h_mat, grid = build_hamiltonian(z, ell, n_points, r_max)
    evals, evecs = jnp.linalg.eigh(h_mat)
    
    # Target nodes for state (n, ell) is n - ell - 1
    target_nodes = n - ell - 1
    
    # Compute nodes for all calculated eigenvectors
    nodes = jax.vmap(count_nodes)(evecs.T)
    
    # Calculate analytical energy for state selection guidance
    e_ana = -0.5 * (z**2) / (n**2)
    
    # Masking logic: Find state with correct node count and minimal energy deviation
    is_correct_node = (nodes == target_nodes)
    score = jnp.abs(evals - e_ana) + jnp.where(is_correct_node, 0.0, 1e6)
    
    idx = jnp.argmin(score)
    
    # Return as JAX arrays to maintain JIT/vmap compatibility (avoid float() casting)
    return evals[idx], evecs[:, idx], grid
