# KT_backgrounds.py
"""
Collection of background fields (T(t,x), v(t,x)) for the generalised KT solver.
All functions are decorated with @jit(nopython=True) so they can be called
from within Numba‑compiled functions.

To use a background, import the desired functions and assign them to the
solver's global T_func / v_func before running the simulation.
"""

import numpy as np
from numba import jit

# -------------------------------------------------------------------
# 1. Constant rest frame (default)
# -------------------------------------------------------------------
@jit(nopython=True, cache=False)
def T_const(t, x):
    return 0.3 * np.ones_like(x)

@jit(nopython=True, cache=False)
def v_zero(t, x):
    return np.zeros_like(x)

# -------------------------------------------------------------------
# 2. Uniform boost
# -------------------------------------------------------------------
@jit(nopython=True, cache=False)
def T_const_boost(t, x):
    return 0.3 * np.ones_like(x)

@jit(nopython=True, cache=False)
def v_const_boost(t, x):
    return 0.5 * np.ones_like(x)

# -------------------------------------------------------------------
# 3. Gaussian temperature bump (zero velocity)
# -------------------------------------------------------------------
@jit(nopython=True, cache=False)
def T_gauss_bump(t, x):
    return 0.3 + 0.05 * np.exp(-x**2 / 200.0)

@jit(nopython=True, cache=False)
def v_zero_for_bump(t, x):
    return np.zeros_like(x)

# -------------------------------------------------------------------
# 4. Mild tanh shear (small amplitude, constant T)
# -------------------------------------------------------------------
@jit(nopython=True, cache=False)
def T_const_shear(t, x):
    return 0.3 * np.ones_like(x)

@jit(nopython=True, cache=False)
def v_tanh_shear(t, x):
    return 0.05 * np.tanh(x / 10.0)

# -------------------------------------------------------------------
# 5. Full non‑trivial: sinusoidal T + tanh shear
# -------------------------------------------------------------------
@jit(nopython=True, cache=False)
def T_sine(t, x):
    return 0.3 + 0.03 * np.sin(2.0 * np.pi * x / 50.0)

@jit(nopython=True, cache=False)
def v_tanh(t, x):
    return 0.4 * np.tanh(x / 10.0)

# -------------------------------------------------------------------
# 6. Travelling Gaussian temperature pulse (t‑dependent example)
# -------------------------------------------------------------------
@jit(nopython=True, cache=False)
def T_travelling(t, x):
    return 0.3 + 0.05 * np.exp(-((x - 0.5*t)**2) / 100.0)

@jit(nopython=True, cache=False)
def v_zero_travelling(t, x):
    return np.zeros_like(x)