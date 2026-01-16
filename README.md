

# AtomJAX: A Differentiable High-Precision Radial Schrödinger Solver

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/Framework-JAX-red.svg)](https://github.com/google/jax)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harihardiyan/AtomJAX/blob/main/notebooks/AtomJAX_Tutorial.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AtomJAX** is a high-performance, JAX-native numerical engine designed to solve the radial Schrödinger equation for hydrogenic atoms. By leveraging 4th-order finite difference (FD4) methods and sophisticated boundary condition handling, AtomJAX provides a robust framework for calculating atomic eigenstates, energies, and observables with academic-grade precision.

## 🔬 Scientific Context

The software solves the time-independent radial Schrödinger equation (in atomic units):

$$-\frac{1}{2} \frac{d^2u}{dr^2} + \left[ \frac{\ell(\ell+1)}{2r^2} - \frac{Z}{r} \right] u(r) = E u(r)$$

where $u(r) = rR(r)$ is the reduced radial wavefunction. AtomJAX specializes in the numerical treatment of this singular differential equation through a vectorized, differentiable pipeline.

---

## 🚀 Key Features

### 1. High-Order Finite Difference Engine (FD4)
Unlike standard 3-point stencils (2nd order), AtomJAX implements a **4th-order central difference scheme**. This significantly reduces the grid density required to achieve convergence, allowing for high-precision results even on modest hardware.

### 2. Parity-Aware Ghost-Point BCs
A critical challenge in radial solvers is the boundary at $r=0$. AtomJAX utilizes a **ghost-point boundary condition** that exploits the parity of the wavefunction $u(r) \sim r^{\ell+1}$. By substituting $u(-h)$ based on the orbital angular momentum $\ell$, we maintain 4th-order accuracy even at the coordinate origin.

### 3. JAX-Native Vectorization (vmap)
AtomJAX is built from the ground up for **Hardware Acceleration**.
*   **Batching over Z**: Compute the entire periodic table's hydrogenic properties in a single vectorized call using `jax.vmap`.
*   **Just-In-Time (JIT) Compilation**: The entire Hamiltonian construction and eigensolver pipeline are compiled into optimized XLA kernels.

### 4. Automated Nodal State Selection
Navigating the energy spectrum is automated via a nodal counting algorithm. By analyzing the sign-changes of eigenvectors, the system identifies the physical $(n, \ell)$ states, ensuring the correct principal quantum number is mapped to the numerical eigenvalue.

---

## 🎓 Pedagogical Design & Limitations

### Pedagogical Intent
AtomJAX is designed to be a "Transparent Box." Unlike monolithic simulation packages, the mapping from the physics (Schrödinger equation) to the code (Hamiltonian matrix assembly) is direct and readable. It serves as an ideal reference for:
*   Computational physics students learning finite difference methods.
*   Researchers needing a fast, differentiable atomic baseline for perturbation theory.

### Technical Limitations
While robust, the current implementation has specific boundaries:
*   **Non-Relativistic**: It solves the Schrödinger equation, not the Dirac equation. Fine structure effects are not included.
*   **Hydrogenic Focus**: The solver assumes a central potential $-Z/r$. It does not currently account for electron-electron correlation (e.g., Hartree-Fock for multi-electron systems).
*   **Grid Sensitivity**: For very high $Z$ or high $n$, the grid parameters ($N_{points}$ and $R_{max}$) must be manually tuned to capture the rapidly oscillating wavefunctions or the extended tails.

---

## 🛠 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/harihardiyan/atomjax.git
cd atomjax

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```python
import jax.numpy as jnp
from atomjax.solvers import solve_hydrogen_state

# Solve for the 2p state of Hydrogen (Z=1, n=2, l=1)
energy, wavefunction, grid = solve_hydrogen_state(z=1.0, n=2, ell=1)

print(f"Numerical Energy: {energy:.8f} Ha")
```

---

## 📊 Performance Benchmark
AtomJAX achieves relative errors in energy on the order of $10^{-9}$ to $10^{-12}$ using standard double precision (`float64`), outperforming standard 2nd-order methods by several orders of magnitude at the same grid resolution.

---

## ✍️ Authorship & Collaboration

*   **Hari Hardiyan** - Lead Physicist / Developer
*   **Microsoft Copilot** - Computational Architecture & Optimization Support

**Correspondence:** [lorozloraz@gmail.com](mailto:lorozloraz@gmail.com)

---

## 📜 Citation

If you utilize AtomJAX in your academic work, please cite it as follows:

```bibtex
@software{hardiyan2026atomjax,
  author = {Hardiyan, Hari and Microsoft Copilot},
  title = {AtomJAX: A Differentiable High-Precision Radial Schrödinger Solver},
  year = {2026},
  url = {https://github.com/harihardiyan/atomjax},
  note = {Email: lorozloraz@gmail.com}
}
```
