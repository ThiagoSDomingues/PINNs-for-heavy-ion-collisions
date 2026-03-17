"""
IC_IS.py - Initial conditions for the Israel-Stewart PINN
==========================================================

Physical background
-------------------

n   : charge density
J   : spatial component of the dissipative flux in the LRF
α   : reduced chemical potential α = μ/T 
T   : temperature (background field, set by T_func)
σ   : charge conductivity (from RTA)
τ_J : Israel-Stewart relaxation time  
"""

import torch
from IS_Functions import (
    T_func,
    v_func,
    alpha_from_n_func,
    n_from_alpha_func,
    sigma_func,
    tau_J_func,
    dalpha_dx_func,
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
    
    # --- Step 1: unpack coordinates -----------------------------------
    # x is the PINN collocation tensor, shaped (N, 2)
    # Column 0 is time (= 0 for ICs), column 1 is the spatial coordinate.
    t_vals = x[:, 0:1] # shape (N, 1), all zeros
    x_vals = x[:, 1:2] # shape (N, 1), the spatial grid

    # --- Step 2: background temperature and velocity -----------------------------------
    # In the probe approximation the temperature T and fluid velocity v are 
    # external background fields, not evolved by the equations.
    # T_func and v_func are the same functions used in IC_BDNK; we call them
    # here so the IS IC is defined on exactly the same background as BDNK.
    
    T = T_func(t_vals, x_vals)   # shape (N, 1), e.g. constant T = 1.0
    v = v_func(t_vals, x_vals)   # shape (N, 1), e.g. constant v = 0.0 (LRF)
    
    # --- Step 3: density initial condition n(t=0, x)
    
    
    return n_IC, J_IC 
