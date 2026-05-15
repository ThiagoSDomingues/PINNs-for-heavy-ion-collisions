import torch
from BDNK_IS_Functions import n_from_alpha_func

# Background fields (constant in this setup)
def T_func(t, x):
    return 0.3 * torch.ones_like(x)

def v_func(t, x):
    return torch.zeros_like(x)

# Constitutive constants
sigmaT = 1.0          # σ T
lambda_val = 1.0      # λ = τ_J/(σ T)

def pde_residual(alpha, q, tx):
    """
    PDE residuals for the divergence‑type diffusion in 1+1D.
    alpha, q : (N,1) tensors
    tx       : (N,2) tensor, columns (t, x)
    Returns  : (N,2) tensor with (R1, R2)
    """
    t = tx[:, 0:1]
    x = tx[:, 1:2]
    n = n_from_alpha_func(alpha, T_func(t, x))

    def grad(u):
        g = torch.autograd.grad(
            u, tx,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        return g

    # R1: charge conservation
    g_n = grad(n)
    g_q = grad(q)
    R1 = g_n[:, 0:1] + g_q[:, 1:2]            # ∂_t n + ∂_x q

    # R2: relaxation equation
    lambda_q = lambda_val * q
    g_lq = grad(lambda_q)
    g_alpha = grad(alpha)
    R2 = g_lq[:, 0:1] + g_alpha[:, 1:2] + (1.0 / sigmaT) * q

    return torch.cat([R1, R2], dim=1)
