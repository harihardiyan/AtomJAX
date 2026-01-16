import jax.numpy as jnp
from atomjax.hamiltonian import build_hamiltonian

def test_hamiltonian_symmetry():
    """Ensure the Hamiltonian matrix is Hermitian (Symmetric)."""
    z, ell, n_pts = 1.0, 0, 100
    h_mat, _ = build_hamiltonian(z, ell, n_pts, 20.0)
    assert jnp.allclose(h_mat, h_mat.T)

def test_grid_scaling():
    """Ensure grid scales inversely with Z."""
    h1, grid1 = build_hamiltonian(1.0, 0, 100, 50.0)
    h2, grid2 = build_hamiltonian(2.0, 0, 100, 50.0)
    assert jnp.isclose(grid1.r.max(), grid2.r.max() * 2)
