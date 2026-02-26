# Author: OptimusThi

import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import numpy as np

# QCD-like equation of state (Conformal (since ε = 3P))
N_c = 3 # colors
N_f = 3 # 3 flavors


def T_func(t, x):
    return 0.3 * torch.ones_like(x)          # This option enabled for arbitrary function backgrounds (first and second setups)
    #return _bicubic_sample_tx(t, x, _T_tab) # This option enabled for BDNK backgrounds (third setup)

def v_func(t, x):
    return 0.0 * torch.ones_like(x)          # This option enabled for arbitrary function bakgrounds (first and second setups)
    #return _bicubic_sample_tx(t, x, _v_tab) # This option enabled for BDNK backgrounds (third setup)
    
# Lorentz factor
def gamma_func(v):
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.float64, device=device)
    return 1.0 / torch.sqrt(1 - v**2)

### QCD thermodynamics
# alpha = mu / T -> chemical potential divided by temperature
def alpha_from_n_func(n, T):
    a = T**3 * N_c * N_f
    b = torch.sqrt(
        2187 * n**2 * T**12 * N_c**4 * N_f**4
        + 4 * torch.pi**2 * a**6
    )
    c = 81 * n * a**2
    d = torch.pow(torch.sqrt(torch.tensor(3.0, dtype=n.dtype, device=n.device)) * b + c, 1/3)
    
    term1 = (3/2)**(1/3) * torch.pi**(2/3) * d / a
    term2 = 2**(1/3) * 3**(2/3) * torch.pi**(4/3) * a / d
    
    return term1 - term2

# Number density from alpha
def n_from_alpha_func(alpha, T):
    term1 = alpha / 27
    term2 = (alpha ** 3) / (243 * torch.pi ** 2)
    return N_c * N_f * T**3 * (term1 + term2)
