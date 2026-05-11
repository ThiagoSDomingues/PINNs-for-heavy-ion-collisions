import torch
from BDNK_IS_Functions import T_func, alpha_from_n_func

def IC_DivType(x, L):
    """
    Initial conditions for the divergence‑type PINN.
    Returns (q_ic, alpha_ic) at t=0.
    Uses the same Gaussian ansatz as the BDNK‑IS runs.
    """
    x = x.clone().detach().requires_grad_(True)
    t_vals = x[:, 0:1]
    x_vals = x[:, 1:2]

    T = T_func(t_vals, x_vals)

    # Initial charge density n(0,x)
    p, qq, r = 0.2, 7.0, 1.0   # qq to avoid conflict with the flux variable
    n_init = (p * torch.exp(- (qq * x_vals / L)**2) + r) * 1e-2

    # Obtain alpha from the equation of state
    alpha_init = alpha_from_n_func(n_init, T)

    # Initial diffusive flux q(0,x)
    d, f, g = 0.05, 10.0, 1.05
    q_init = (d * torch.exp(- (f * x_vals / L)**2) + g) * 1e-2

    return q_init, alpha_init
