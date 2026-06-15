"""Exact solution of the 1D wave equation on [-L, L] with periodic BCs."""
import numpy as np

def exact_wave(x: np.ndarray, t: np.ndarray,
               L: float = 1.0, c: float = 1.0) -> np.ndarray:
    """
    u(t,x) = 0.5 * [cos(3π c t/L) sin(3π x/L) − cos(π c t/L) sin(π x/L)]
    Returns (Nt, Nx) array.
    """
    tt, xx = np.meshgrid(t, x, indexing='ij')
    return 0.5 * (
        np.cos(3.0 * np.pi * c * tt / L) * np.sin(3.0 * np.pi * xx / L)
        - np.cos(np.pi * c * tt / L) * np.sin(np.pi * xx / L)
    )

def ic_wave(x: np.ndarray, L: float = 1.0):
    """Returns (u0, ut0) for the wave equation IC."""
    u0  = np.sin(np.pi * x / L) * np.cos(2.0 * np.pi * x / L)
    ut0 = np.zeros_like(x)
    return u0, ut0
