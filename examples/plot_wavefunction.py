import matplotlib.pyplot as plt
from atomjax.solvers import solve_hydrogen_state

def main():
    z = 1.0
    states = [(1, 0), (2, 0), (2, 1)] # (n, l)
    
    plt.figure(figsize=(10, 6))
    
    for n, ell in states:
        e, u, grid = solve_hydrogen_state(z, n, ell)
        # Normalize for plotting
        u_plot = u / jnp.max(jnp.abs(u))
        plt.plot(grid.r, u_plot, label=f"n={n}, l={ell} (E={e:.4f} Ha)")
    
    plt.xlim(0, 20)
    plt.title(f"Radial Wavefunctions for Z={z}")
    plt.xlabel("r (a.u.)")
    plt.ylabel("u(r) [normalized max]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    import jax.numpy as jnp
    main()
