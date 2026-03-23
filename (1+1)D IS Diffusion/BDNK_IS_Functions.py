import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import numpy as np

N_c = 3
N_f = 3
C_B = 1/(4*np.pi)

def T_func(t, x):
    return 0.3 * torch.ones_like(x)          # This option enabled for arbitrary function backgrounds (first and second setups)
    #return _bicubic_sample_tx(t, x, _T_tab) # This option enabled for IS backgrounds (third setup)

def v_func(t, x):
    return 0.0 * torch.ones_like(x)          # This option enabled for arbitrary function backgrounds (first and second setups)
    #return _bicubic_sample_tx(t, x, _v_tab) # This option enabled for IS backgrounds (third setup)
    
def gamma_func(v):
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.float64, device=device)
    return 1.0 / torch.sqrt(1 - v**2)

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

def n_from_alpha_func(alpha, T):
    term1 = alpha / 27
    term2 = (alpha ** 3) / (243 * torch.pi ** 2)
    return N_c * N_f * T**3 * (term1 + term2)

def N_x_func(alpha, x):
    grad = torch.autograd.grad(
        alpha, x,
        grad_outputs=torch.ones_like(alpha),
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )[0]

    return -grad[:, -1:]

def mu_func(alpha, T):
    return alpha * T

def pressure_func(alpha, T):
    pi = torch.pi
    mu = mu_func(alpha, T)
    term1 = (2*(N_c**2 - 1) + 3.5 * N_c * N_f) * pi**2 * T**4 / 90
    term2 = N_c * N_f * mu**2 * T**2 / 54
    term3 = N_c * N_f * mu**4 / (972 * pi**2)
    return term1 + term2 + term3

def sigma_func(alpha, T):
    mu = mu_func(alpha, T)
    n = n_from_alpha_func(alpha, T)
    P = pressure_func(alpha, T)
    eps = 3*P
    return C_B * n * (1/3 * 1/torch.tanh(alpha) - n*T/(eps+P)) / (T**2)

def lambd_func(sigma):
    cch = 0.5
    return sigma/(cch**2)

def N_0_func(lambd, sigma, T, J0, n, N_x, v):
    gamma = gamma_func(v)
    num = -J0 + gamma * n + (sigma - lambd) * T * gamma**2 * v * N_x
    denom = sigma * T + (lambd - sigma) * T * gamma**2
    return num / denom

def Jx_func(n, sigma, lambd, T, N_x, N_0, v):
    Nx = N_x
    gamma = gamma_func(v)
    return (gamma * n * v
            + sigma * T * Nx
            + gamma**2 * T * (sigma - lambd) * v**2 * Nx
            + gamma**2 * T * (sigma - lambd) * v * N_0)

def J0_func(T, v, alpha, alpha_t, x):
    gamma = gamma_func(v)
    sigma = sigma_func(alpha, T)
    lambd = lambd_func(sigma)
    
    alpha_x = torch.autograd.grad(
        alpha, x,
        grad_outputs=torch.ones_like(alpha),
        create_graph=True
    )[0][:, 1:2]
    
    n = n_from_alpha_func(alpha, T)
    
    term1 = n * gamma
    term2 = (sigma * T - gamma**2 * T * (sigma - lambd)) * alpha_t
    term3 = gamma**2 * T * (sigma - lambd) * v * alpha_x

    return term1 + term2 - term3


# ==========================================
# IS FUNCTIONS AND BDNK TO IS FUNCTIONS
# ==========================================

def tauJ_func(alpha, T):
    sigma = sigma_func(alpha, T)
    n = n_from_alpha_func(alpha, T)

    return 12 * sigma * T / n

def IS_IC_from_BDNK_IC_func(t_vals, x_vals, alpha_1, J0_1):
    v = v_func(t_vals, x_vals)
    T = T_func(t_vals, x_vals)
    n_1 = n_from_alpha_func(alpha_1, T)
    gamma = gamma_func(v)
    sigma_1 = sigma_func(alpha_1, T)
    lambd_1 = lambd_func(sigma_1)
    N_x_1 = N_x_func(alpha_1, x_vals)
    N_0_1 = N_0_func(lambd_1, sigma_1, T, J0_1, n_1, N_x_1, v)
    
    alpha_1_t = -N_0_1
    alpha_1_x = -N_x_1

    n_2 = n_1 + lambd_1 * T * gamma * (alpha_1_t + v * alpha_1_x)
    alpha_2 = alpha_from_n_func(n_2, T)
    scriptJ = -sigma_1 * T * gamma * (v * alpha_1_t + alpha_1_x)

    return alpha_2, scriptJ
