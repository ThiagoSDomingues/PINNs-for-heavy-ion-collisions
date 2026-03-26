import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
import torch.autograd

from BDNK_Functions import *


# ============================================================
# SigmaPINN: learns σ₂(n₂) that minimises the IS loss
#
# Architecture: takes n₂ as input (scalar), outputs σ₂ (scalar)
# 3 hidden layers × 50 neurons × tanh, inspired by SA_PINN_ACTO
# but without any SA or ACTO components.
# ============================================================

class SigmaPINN(nn.Module):
    def __init__(self, Nl=3, Nn=50, lb_n=None, ub_n=None):
        super().__init__()

        # Input normalisation bounds (set after you know the n2 range)
        # BUG 6 FIX: IS_loss was indented inside the class as a broken
        #   pseudo-method.  It is now a proper standalone function below.
        # BUG 8 FIX: we store the n2 scale so inputs are normalised to [-1,1]
        #   before entering the network, matching the SA_PINN_ACTO pattern.
        if lb_n is not None:
            self.register_buffer('lb_n', torch.as_tensor(lb_n, dtype=torch.get_default_dtype()))
            self.register_buffer('ub_n', torch.as_tensor(ub_n, dtype=torch.get_default_dtype()))
        else:
            self.lb_n = None
            self.ub_n = None

        # Base network: n₂ (scalar) → σ₂ (positive scalar)
        self.net = nn.Sequential()
        self.net.add_module('Linear_layer_1', nn.Linear(1, Nn))
        self.net.add_module('Tanh_layer_1',   nn.Tanh())
        for num in range(2, Nl + 1):
            self.net.add_module(f'Linear_layer_{num}', nn.Linear(Nn, Nn))
            self.net.add_module(f'Tanh_layer_{num}',   nn.Tanh())
        self.net.add_module('Linear_layer_final', nn.Linear(Nn, 1))

    # Helper: scale n₂ from physical units to [-1, 1]
    def _scale(self, n2):
        if self.lb_n is not None and self.ub_n is not None:
            return 2.0 * (n2 - self.lb_n) / (self.ub_n - self.lb_n) - 1.0
        return n2    # no scaling if bounds not set

    def forward(self, n2):
        """
        n2 : (N, 1) tensor — IS number density
        Returns σ₂ : (N, 1) tensor — positive conductivity via softplus
        """
        n2_scaled = self._scale(n2)
        raw = self.net(n2_scaled)
        # Softplus ensures σ₂ > 0 everywhere (conductivity must be positive)
        return nn.functional.softplus(raw)


# ============================================================
# IS loss function (standalone, outside the class)
#
# BUG 6 FIX (continued): moved out of the class entirely.
# BUG 7 FIX: n2, scriptJ, alpha2 are recomputed INSIDE the loss from
#   (bdnk_model, t, x) so that autograd can differentiate them w.r.t.
#   (t, x).  Passing pre-computed detached tensors would break the graph.
# ============================================================

def IS_loss(sigma_model, bdnk_model, t, x):
    """
    Compute the IS residual loss by:
      1. Running the frozen BDNK model to get alpha1, J0.
      2. Applying the matching conditions to get n2, scriptJ, alpha2.
      3. Running sigma_model(n2) to get sigma2.
      4. Evaluating the IS PDE residuals R1, R2.

    Parameters
    ----------
    sigma_model : SigmaPINN — the network being trained
    bdnk_model  : PINN_BDNK_1D — the frozen, trained BDNK PINN
    t           : (N, 1) tensor, requires_grad=True
    x           : (N, 1) tensor, requires_grad=True

    Returns
    -------
    loss   : scalar tensor
    R1_sq  : (N, 1) detached — charge conservation residual squared
    R2_sq  : (N, 1) detached — IS relaxation residual squared
    """

    # ── Step 1: BDNK solution at (t, x) ─────────────────────────────────
    # BUG 7 FIX: we call bdnk_model(tx) here (with gradient flowing through
    #   t and x) instead of using pre-computed detached tensors.  This
    #   ensures that when we differentiate n2, scriptJ, alpha2 w.r.t. (t,x)
    #   in Step 3, the graph exists.
    tx       = torch.cat([t, x], dim=1)
    out_bdnk = bdnk_model(tx)          # [J0_bdnk, alpha1]
    alpha1   = out_bdnk[:, 1:2]        # BDNK fugacity

    # ── Step 2: Matching conditions → IS fields ──────────────────────────
    # alpha2, scriptJ, n2 are all differentiable functions of (t, x) because
    # alpha1 = bdnk_model(t, x) carries the graph from Step 1.
    alpha2, scriptJ, n2, n1 = alpha2_and_scriptJ_from_alpha_and_J0(alpha1, t, x)

    T = T_func(t, x)

    # ── Step 3: σ₂(n₂) from the SigmaPINN ───────────────────────────────
    sigma2 = sigma_model(n2)                    # σ₂(n₂)  — (N, 1), positive

    # Relaxation time:  τ_J = 12 σ₂ T / n₂
    tau2   = 12.0 * sigma2 * T / (n2.abs().clamp(min=1e-8))

    # ── Step 4: IS PDE residuals ─────────────────────────────────────────
    # Gradients of the IS fields w.r.t. (t, x)
    def grad_tx(u, wrt):
        return torch.autograd.grad(
            u, wrt,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

    dn2_dt      = grad_tx(n2,      t)   # ∂_t n₂
    dJ_dx       = grad_tx(scriptJ, x)   # ∂_x J
    dJ_dt       = grad_tx(scriptJ, t)   # ∂_t J
    dalpha2_dx  = grad_tx(alpha2,  x)   # ∂_x α₂

    # R1 = ∂_t n₂ + ∂_x J  (charge conservation)
    # We use n2 instead of J^0 = γn₂ because v=0 in the background,
    # so J^0 = n₂ and J^x = scriptJ in the LRF.
    R1 = dn2_dt + dJ_dx

    # R2 = τ_J ∂_t J + J + σ₂ T ∂_x α₂  (IS relaxation, LRF v=0)
    # This is the correct simplified form for v = 0 (no extra τ/(σT) term).
    # BUG 9 NOTE: the paper's full R2 has an extra
    #   σ₂T/2 · J · ∂_ν[τ_J u^ν / (σ₂T)]
    # term.  For v=0 and constant T, τ_J/(σT) = 12/n₂ so
    #   ∂_t[τ_J/(σT)] = -12/n₂² · ∂_t n₂
    # This term is NOT zero in general (it was zero in is_pinn_fixed.py
    # because that code incorrectly dropped it).  We include it here.
    tau_over_sigT    = tau2 / (sigma2 * T + 1e-12)            # 12/n₂
    d_tau_sigT_dt    = grad_tx(tau_over_sigT, t)               # ∂_t[τ/(σT)]
    extra_term       = 0.5 * sigma2 * T * scriptJ * d_tau_sigT_dt

    R2 = tau2 * dJ_dt + scriptJ + extra_term + sigma2 * T * dalpha2_dx

    loss  = torch.mean(R1**2 + R2**2)

    return loss, R1.detach()**2, R2.detach()**2


# ============================================================
# Gradient-strength helper (for diagnostic plots)
# ============================================================

@torch.no_grad()
def gradient_strength(bdnk_model, t, x):
    """
    Returns a measure of the local gradient strength at collocation points,
    defined as |∂_x alpha1| + |∂_t alpha1|.  Used to bin residuals by
    how far from equilibrium each point is.
    """
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    tx = torch.cat([t, x], dim=1)

    with torch.enable_grad():
        out    = bdnk_model(tx)
        alpha1 = out[:, 1:2]
        a_t = torch.autograd.grad(alpha1.sum(), t, create_graph=False)[0]
        a_x = torch.autograd.grad(alpha1.sum(), x, create_graph=False)[0]

    return (a_t.abs() + a_x.abs()).detach()
