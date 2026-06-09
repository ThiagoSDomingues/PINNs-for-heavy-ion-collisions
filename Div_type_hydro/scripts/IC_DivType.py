import numpy as np
import torch
from DivType_Functions import T_func, alpha_from_n_func

def IC_DivType(x, L):
    """
    Initial conditions for the divergence‑type PINN.
    Returns (scriptJ_ic, alpha_ic) at t=0.
    Generates different setups.
    """
    N = x.shape[0] 
#    x = x.clone().detach().requires_grad_(True)
    t_vals = x[:, 0:1]
    x_vals = x[:, 1:2]

    # Initial charge density n(t=0,x)
    p, qq, r = 0.2, 7.0, 1.0    # Initial condition parameters
    
    # First setup:
    n_init = p * torch.exp(- (qq * x_vals / L)**2) + r

    # Second setup:
    # sharpness = 60
    # n_init = (1.1 - 0.1*torch.tanh(sharpness*((4*x_vals/L)**2-1)))

    # Third setup:
    #n_init = 1e-3*(p * torch.exp(- (qq * x_vals / L)**2) + r)

    T = T_func(t_vals, x_vals)

    # Obtain alpha from the equation of state
    alpha_init = alpha_from_n_func(n_init)

#    dx = x_vals[1] - x_vals[0]
#    alpha_padded = torch.cat([alpha_init[:1], alpha_init, alpha_init[-1:]], dim=0)
#    N_x_init = -(alpha_padded[2:] - alpha_padded[:-2]) / (2.0 * dx)
    

    # Initial diffusive flux scriptJ(t=0,x)
    d, f, g = 0.05, 10.0, 1.05 # Initial condition parameters

    # First setup:
    scriptJ_init = d * torch.exp(- (f * x_vals / L)**2) + g

    # Second setup:
    #scriptJ_init = torch.ones_like(x_vals)*g

    # Third setup:
    #scriptJ_init = 1e-3 * torch.ones_like(x_vals)*g

    return scriptJ_init, alpha_init
