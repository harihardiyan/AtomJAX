import jax.numpy as jnp
from typing import Tuple, NamedTuple

class Grid(NamedTuple):
    r: jnp.ndarray
    h: float
    z: float
    ell: int

def build_hamiltonian(z: float, ell: int, n_points: int, r_max: float) -> Tuple[jnp.ndarray, Grid]:
    """Builds the radial Hamiltonian matrix using 4th-order finite difference."""
    # Scale grid by Z to maintain resolution across different atoms
    r_full = jnp.linspace(0.0, r_max / z, n_points + 1)
    h = r_full[1] - r_full[0]
    r = r_full[1:-1]
    n = r.shape[0]

    
    # FD4 second derivative coefficients
    # (-1/12, 4/3, -5/2, 4/3, -1/12) / h^2
    c0 = -30.0 / (12.0 * h**2)
    c1 =  16.0 / (12.0 * h**2)
    c2 =  -1.0 / (12.0 * h**2)

    # Kinetic operator T = -0.5 * D2
    main_diag = jnp.full(n, -0.5 * c0)
    off_1 = jnp.full(n-1, -0.5 * c1)
    off_2 = jnp.full(n-2, -0.5 * c2)

    h_mat = (
        jnp.diag(main_diag) +
        jnp.diag(off_1, k=1) + jnp.diag(off_1, k=-1) +
        jnp.diag(off_2, k=2) + jnp.diag(off_2, k=-2)
    )

    # Ghost-point Boundary Condition at r=0
    parity = (-1.0)**(ell + 1)
    h_mat = h_mat.at[0, 0].add(-0.5 * c2 * parity)

    # Potential V = ell(ell+1)/(2r^2) - Z/r
    r_safe = jnp.maximum(r, 1e-14)
    v_pot = 0.5 * ell * (ell + 1) / (r_safe**2) - z / r_safe
    h_mat += jnp.diag(v_pot)

    return h_mat, Grid(r=r, h=h, z=z, ell=ell)
