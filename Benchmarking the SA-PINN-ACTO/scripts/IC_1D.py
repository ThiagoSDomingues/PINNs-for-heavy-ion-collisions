import numpy as np
import torch


#def ic_burgers(x, L):
#    N = x.shape[0]
#    t_vals = x[:, 0:1]
#    x_vals = x[:, 1:2]

    # Initial condition u(t=0,x)
#    u = -torch.sin(torch.pi * x_vals / L)

#    return u

# ── Burgers ──────────────────────────────────────────────────────────────────
def ic_burgers(x_tensor: torch.Tensor, L: float) -> torch.Tensor:
    """u(0,x) = -sin(π x / L).  x_tensor: (N,1)."""
    return -torch.sin(torch.pi * x_tensor / L)

# ── Wave ──────────────────────────────────────────────────────────────────────
def ic_wave_u0(x_tensor: torch.Tensor, L: float) -> torch.Tensor:
    """u(0,x) = sin(πx/L) cos(2πx/L)."""
    return (torch.sin(torch.pi * x_tensor / L)
            * torch.cos(2.0 * torch.pi * x_tensor / L))

def ic_wave_ut0(x_tensor: torch.Tensor, L: float) -> torch.Tensor:
    """∂_t u(0,x) = 0."""
    return torch.zeros_like(x_tensor)

# ── Diffusion ────────────────────────────────────────────────────────────────
def ic_diffusion(x_tensor: torch.Tensor, L: float, sigma_frac: float = 0.15) -> torch.Tensor:
    """u(0,x) = Gaussian bump."""
    sigma = sigma_frac * L
    return torch.exp(-0.5 * (x_tensor / sigma) ** 2)

# ── Euler / Sod ───────────────────────────────────────────────────────────────
def ic_sod(x_tensor: torch.Tensor, x0: float = 0.0):
    """
    Returns (rho, v, p) each (N,1) tensors.
    Standard Sod: left=(1,0,1), right=(0.125,0,0.1).
    """
    mask = (x_tensor < x0).float()
    rho = 1.0 * mask + 0.125 * (1.0 - mask)
    v   = torch.zeros_like(x_tensor)
    p   = 1.0 * mask + 0.1   * (1.0 - mask)
    return rho, v, p