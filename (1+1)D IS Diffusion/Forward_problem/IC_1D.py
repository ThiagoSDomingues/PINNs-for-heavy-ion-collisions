"""
IC_1D.py - Initial conditions for the Israel-Stewart PINN
==========================================================
"""

import torch
import numpy as np
from BDNK_IS_Functions import (
    T_func,
    alpha_from_n_func,
    IS_IC_from_BDNK_IC_func,
) 

# ================================================
# Main IC function
# ================================================

def IC_IS(x: torch.Tensor, L: float):
    """
    Compute the Israel-Stewart initial conditions at t = 0.
    
    Parameters
    ----------
    
    x : torch.Tensor, shape (N, 2)
        Collocation points; first column is t (all zeros), second is x.     
    L : float
        Characteristic length scale of the domain.
    
    """
    x = x.clone().detach().requires_grad_(True)

    # ============================
    # BDNK IC
    # ============================
    N_c, N_f = 3, 3
    
    # --- unpack coordinates -----------------------------------
    # x is the PINN collocation tensor, shaped (N, 2)
    N = x.shape[0]
    # Column 0 is time (= 0 for ICs), column 1 is the spatial coordinate.
    t_vals = x[:, 0:1] # shape (N, 1), all zeros
    x_vals = x[:, 1:2] # shape (N, 1), the spatial grid

    # Initial condition n(t=0,x)
    p, q, r = 0.2, 7.0, 1.0  # Initial condition parameters

    # First setup:
    n = (p * torch.exp(- (q * x_vals / L)**2) + r)*1e-3

    # --- background temperature -----------------------------------
    # In the probe approximation the temperature T and fluid velocity v are 
    # external background fields, not evolved by the equations.
    # T_func and v_func are the same functions used in IC_BDNK; we call them
    # here so the IS IC is defined on exactly the same background as BDNK.

    T = T_func(t_vals, x_vals)   # shape (N, 1), e.g. constant T = .3 GeV
    alpha = alpha_from_n_func(n, T)

    # Initial condition J0(t=0,x)
    d, f, g = 0.05, 10.0, 1.05      # Initial condition parameters

    # First setup:
    J0 = (d * torch.exp(- (f * x_vals / L)**2) + g)*1e-3

    # ============================
    # FROM BDNK IC, BUILD IS IC
    # ============================

    alpha_2, scriptJ = IS_IC_from_BDNK_IC_func(t_vals, x_vals, alpha, J0)

    # Add a causality condition here for alpha_2
    
    return scriptJ, alpha_2 
