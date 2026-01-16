import jax

# Enable float64 for high-precision scientific computing
jax.config.update("jax_enable_x64", True)

DEFAULT_N_POINTS = 2001
DEFAULT_R_MAX = 100.0
