import jax.numpy as jnp
import pytest
from atomjax.solvers import solve_hydrogen_state

def test_ground_state_energy():
    """Test if Hydrogen (Z=1, n=1, l=0) energy is -0.5 Ha."""
    energy, _, _ = solve_hydrogen_state(z=1.0, n=1, ell=0)
    assert jnp.isclose(energy, -0.5, atol=1e-7)

def test_virial_theorem_ish():
    """Basic sanity check for 2p state."""
    z = 1.0
    n = 2
    expected_e = -0.5 * (z**2) / (n**2)
    energy, _, _ = solve_hydrogen_state(z=z, n=n, ell=1)
    assert jnp.isclose(energy, expected_e, atol=1e-7)
