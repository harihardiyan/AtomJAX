import jax.numpy as jnp
import pytest
from atomjax.solvers import solve_hydrogen_state

def test_hydrogen_ground_state():
    """Verify n=1, l=0 energy is -0.5 Ha for Z=1."""
    e_num, _, _ = solve_hydrogen_state(z=1.0, n=1, ell=0)
    e_ana = -0.5
    # FD4 with 2001 points should be very accurate
    assert jnp.abs(e_num - e_ana) < 1e-9

def test_node_counting():
    """Verify n=2, l=0 has exactly 1 radial node (n-l-1)."""
    from atomjax.solvers import count_nodes
    _, u, _ = solve_hydrogen_state(z=1.0, n=2, ell=0)
    assert count_nodes(u) == 1
