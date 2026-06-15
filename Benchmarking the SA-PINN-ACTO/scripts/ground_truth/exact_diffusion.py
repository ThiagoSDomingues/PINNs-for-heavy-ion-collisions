"""
Exact solution of the 1D diffusion equation u_t = c u_xx on all of R,
evaluated via Gaussian-kernel convolution on a periodic domain [-L, L]
using image-charge wrapping with K image pairs.
"""
import numpy as np

def exact_diffusion(x: np.ndarray, t: np.ndarray,
                    c: float = 0.1, L: float = 1.0,
                    f=None, K: int = 20) -> np.ndarray:
    """
    Parameters
    ----------
    x  : (Nx,) spatial grid
    t  : (Nt,) time grid  (t[0] should be > 0 for the convolution formula)
    c  : diffusion coefficient
    L  : half-domain length
    f  : callable f(y) for the IC; defaults to Gaussian bump
    K  : number of image pairs for periodic wrapping

    Returns
    -------
    u : (Nt, Nx)
    """
    if f is None:
        sigma = 0.15 * L
        f = lambda y: np.exp(-0.5 * (y / sigma) ** 2)

    Nx, Nt = len(x), len(t)
    dy = x[1] - x[0]
    y  = x.copy()
    u  = np.zeros((Nt, Nx))

    for ti, tt in enumerate(t):
        if tt == 0.0:
            u[ti] = f(x)
            continue
        kernel_sum = np.zeros(Nx)
        for k in range(-K, K + 1):
            shift = 2.0 * k * L
            diff  = x[:, None] - (y[None, :] + shift)       # (Nx, Nx)
            G     = np.exp(-diff**2 / (4.0 * c * tt)) / np.sqrt(4.0 * np.pi * c * tt)
            kernel_sum += G @ f(y) * dy
        u[ti] = kernel_sum

    return u

def ic_diffusion_gaussian(x: np.ndarray, L: float = 1.0):
    """Default IC: Gaussian bump centered at 0."""
    sigma = 0.15 * L
    return np.exp(-0.5 * (x / sigma) ** 2)
