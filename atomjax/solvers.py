import jax
from jax import jit, vmap
import jax.numpy as jnp
from typing import Tuple
from atomjax.hamiltonian import build_hamiltonian, Grid
from atomjax.constants import DEFAULT_N_POINTS, DEFAULT_R_MAX

@jit
def count_nodes(u: jnp.ndarray) -> int:
    """Calculates the number of sign changes in the wavefunction."""
    s = jnp.sign(u)
    s = jnp.where(s == 0.0, 1.0, s)
    return jnp.sum((s[1:] * s[:-1]) < 0.0)

def solve_hydrogen_state(
    z: float, 
    n: int, 
    ell: int, 
    n_points: int = DEFAULT_N_POINTS, 
    r_max: float = DEFAULT_R_MAX
) -> Tuple[float, jnp.ndarray, Grid]:
    """
    Solves for a specific (n, ell) state using eigenvalue decomposition.
    """
    h_mat, grid = build_hamiltonian(z, ell, n_points, r_max)
    evals, evecs = jnp.linalg.eigh(h_mat)
    
    # Target nodes for state (n, ell) is n - ell - 1
    target_nodes = n - ell - 1
    
    # Vectorized node counting across all eigenvectors
    node_counts = vmap(count_nodes)(evecs.T)
    
    # Analytical energy for guidance
    e_ana = -0.5 * (z**2) / (n**2)
    
    # Selection logic: prioritize node count, then energy proximity
    mask = (node_counts == target_nodes)
    score = jnp.abs(evals - e_ana) + jnp.where(mask, 0.0, 1e9)
    
    idx = jnp.argmin(score)
    return evals[idx], evecs[:, idx], grid
