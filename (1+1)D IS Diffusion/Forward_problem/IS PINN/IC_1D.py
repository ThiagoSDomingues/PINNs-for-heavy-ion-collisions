import numpy as np
import torch
from BDNK_IS_Functions import *

def IC_IS(x, L):
    x = x.clone().detach().requires_grad_(True)
    
    # ============================
    # BDNK IC
    # ============================
    N_c, N_f = 3, 3
    N = x.shape[0]
    t_vals = x[:, 0:1]
    x_vals = x[:, 1:2]

    # Initial condition n(t=0,x)
    p, q, r = 0.2, 7.0, 1.0      # Initial condition parameters
    
    # First setup:
    n = (p * torch.exp(- (q * x_vals / L)**2) + r)*1e-3
    
    T = T_func(t_vals, x_vals)
    
    alpha = alpha_from_n_func(n, T)
    
    dx = x_vals[1] - x_vals[0]
    alpha_padded = torch.cat([alpha[:1], alpha, alpha[-1:]], dim=0)
    N_x = -(alpha_padded[2:] - alpha_padded[:-2]) / (2.0 * dx)

    
    # Initial condition J0(t=0,x)
    d, f, g = 0.05, 10.0, 1.05      # Initial condition parameters

    # First setup:
    J0 = (d * torch.exp(- (f * x_vals / L)**2) + g)*1e-3

    # ============================
    # FROM BDNK IC, BUILD IS IC
    # ============================

    alpha_2, scriptJ = IS_IC_from_BDNK_IC_func(t_vals, x_vals, alpha, J0)
    
    return scriptJ, alpha_2
