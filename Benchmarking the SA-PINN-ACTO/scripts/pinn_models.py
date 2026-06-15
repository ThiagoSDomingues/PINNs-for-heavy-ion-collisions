import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------
# Base class (common to all variants)
# ----------------------------------------------------------------------
class BasePINN(nn.Module):
    def __init__(self, layers, lb, ub, pde_func, ic_func, bc_func, bc_type='periodic'):
        """
        layers   : list of ints, e.g. [2, 50, 50, 1] for 1D scalar output
        lb, ub   : lower/upper bounds for (t, x) – scaling to [-1,1]
        pde_func : function (model, X) returning residual tensor (N, n_out)
        ic_func  : function (x) returning IC values (N, n_out)
        bc_func  : function (t, x_boundary) returning BC values (N, n_out) for Dirichlet
        bc_type  : 'periodic' or 'dirichlet'
        """
        super().__init__()
        self.lb = torch.as_tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.as_tensor(ub, dtype=torch.float32, device=device)
        self.pde_func = pde_func
        self.ic_func = ic_func
        self.bc_func = bc_func
        self.bc_type = bc_type.lower()

        # Build network
        self.net = self._build_network(layers)

    def _build_network(self, layers):
        seq = []
        for i in range(len(layers)-2):
            seq.append(nn.Linear(layers[i], layers[i+1]))
            seq.append(nn.Tanh())
        seq.append(nn.Linear(layers[-2], layers[-1]))
        return nn.Sequential(*seq)

    def scale(self, X):
        """Map physical (t,x) to [-1,1]"""
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, X_phys):
        """Base forward pass – returns raw network output (no ACTO)"""
        Z = self.scale(X_phys)
        return self.net(Z)

    def gradients(self, y, x, retain_graph=True):
        return autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                             create_graph=True, retain_graph=retain_graph)[0]

    # ---------- Loss components ----------
    def loss_pde(self, X_colloc):
        R = self.pde_func(self, X_colloc)
        return (R**2).mean()

    def loss_ic(self, X_ic):
        u_pred = self.forward(X_ic)
        u_target = self.ic_func(X_ic[:, 1:2])   # x only
        return ((u_pred - u_target)**2).mean()

    def loss_bc(self, X_bcL, X_bcR):
        if self.bc_type == 'periodic':
            uL = self.forward(X_bcL)
            uR = self.forward(X_bcR)
            return ((uL - uR)**2).mean()
        else:  # dirichlet
            uL = self.forward(X_bcL)
            uR = self.forward(X_bcR)
            targetL = self.bc_func(X_bcL)
            targetR = self.bc_func(X_bcR)
            return ( ((uL - targetL)**2).mean() + ((uR - targetR)**2).mean() ) / 2

    def loss_total(self, X_colloc, X_ic, X_bcL, X_bcR, w_pde=1.0, w_ic=1.0, w_bc=1.0):
        return w_pde * self.loss_pde(X_colloc) + w_ic * self.loss_ic(X_ic) + w_bc * self.loss_bc(X_bcL, X_bcR)


# ----------------------------------------------------------------------
# Vanilla PINN (no ACTO, no SA)
# ----------------------------------------------------------------------
class VanillaPINN(BasePINN):
    pass


# ----------------------------------------------------------------------
# Self-Adaptive PINN (SA-PINN)
# ----------------------------------------------------------------------
class SAPINN(BasePINN):
    def __init__(self, layers, lb, ub, pde_func, ic_func, bc_func, bc_type,
                 n_colloc, n_ic, n_bc):
        super().__init__(layers, lb, ub, pde_func, ic_func, bc_func, bc_type)

        # Learnable log‑weights for each residual term
        self.log_w_pde = nn.Parameter(torch.zeros(n_colloc, device=device))
        self.log_w_ic  = nn.Parameter(torch.zeros(n_ic,    device=device))
        self.log_w_bc  = nn.Parameter(torch.zeros(n_bc,    device=device))

        # Optimiser for weights (gradient ascent)
        self.weight_optim = torch.optim.Adam([self.log_w_pde, self.log_w_ic, self.log_w_bc], lr=1e-3)

    def loss_pde(self, X_colloc):
        R = self.pde_func(self, X_colloc)
        w = torch.exp(self.log_w_pde).detach()   # detach to avoid weight update through network
        return (w * (R**2)).mean()

    def loss_ic(self, X_ic):
        u_pred = self.forward(X_ic)
        u_target = self.ic_func(X_ic[:, 1:2])
        w = torch.exp(self.log_w_ic).detach()
        return (w * ((u_pred - u_target)**2)).mean()

    def loss_bc(self, X_bcL, X_bcR):
        if self.bc_type == 'periodic':
            uL = self.forward(X_bcL)
            uR = self.forward(X_bcR)
            residual = (uL - uR)**2
        else:
            uL = self.forward(X_bcL)
            uR = self.forward(X_bcR)
            targetL = self.bc_func(X_bcL)
            targetR = self.bc_func(X_bcR)
            residual = torch.cat([(uL - targetL)**2, (uR - targetR)**2], dim=0)
        w = torch.exp(self.log_w_bc).detach()
        return (w * residual).mean()

    def update_weights(self):
        """One gradient ASCENT step for the log‑weights"""
        loss_pde = self.loss_pde(self.X_colloc_store)   # need to store colloc points
        loss_ic  = self.loss_ic(self.X_ic_store)
        loss_bc  = self.loss_bc(self.X_bcL_store, self.X_bcR_store)
        total = loss_pde + loss_ic + loss_bc
        self.weight_optim.zero_grad()
        total.backward()
        self.weight_optim.step()

    # Store training data for weight updates (simplified, you can also pass them)
    def set_training_data(self, X_colloc, X_ic, X_bcL, X_bcR):
        self.X_colloc_store = X_colloc
        self.X_ic_store = X_ic
        self.X_bcL_store = X_bcL
        self.X_bcR_store = X_bcR


# ----------------------------------------------------------------------
# ACTO PINN (exact IC/BC via output transformation)
# ----------------------------------------------------------------------
class ACTOPINN(BasePINN):
    def __init__(self, layers, lb, ub, pde_func, ic_func, bc_func, bc_type):
        super().__init__(layers, lb, ub, pde_func, ic_func, bc_func, bc_type)
        self.Lx = (ub[1] - lb[1]).item()      # domain length

    def forward(self, X_phys):
        t = X_phys[:, 0:1]
        x = X_phys[:, 1:2]

        L = self.Lx
        xi = (x + L) / (2*L)        # 0..1 coordinate

        # Raw network output
        Z = self.scale(X_phys)
        u_raw = self.net(Z)

        # Enforce IC: u(0,x) = ic_func(x)
        g = torch.exp(-10.0 * t / self.ub[0].item())   # g(0)=1
        u_ic = self.ic_func(x)

        # Enforce BC (periodic or Dirichlet)
        if self.bc_type == 'periodic':
            # periodic: subtract the jump at boundaries
            X_L = torch.cat([t, -L*torch.ones_like(x)], dim=1)
            X_R = torch.cat([t,  L*torch.ones_like(x)], dim=1)
            u_L = self.net(self.scale(X_L))
            u_R = self.net(self.scale(X_R))
            u_raw_bc = u_raw - (u_R - u_L) * xi
        else:  # dirichlet
            # blend boundary values
            bcL = self.bc_func(torch.cat([t, -L*torch.ones_like(x)], dim=1))
            bcR = self.bc_func(torch.cat([t,  L*torch.ones_like(x)], dim=1))
            blend = (1-xi)*bcL + xi*bcR
            u_raw_bc = blend + 4*xi*(1-xi)*u_raw

        # Combine IC and BC enforcement
        u_phys = g * u_ic + (1-g) * u_raw_bc
        return u_phys


# ----------------------------------------------------------------------
# SA-PINN-ACTO (self‑adaptive weights + ACTO)
# ----------------------------------------------------------------------
class SAPINN_ACTO(ACTOPINN):
    def __init__(self, layers, lb, ub, pde_func, ic_func, bc_func, bc_type, n_colloc):
        super().__init__(layers, lb, ub, pde_func, ic_func, bc_func, bc_type)
        # Only PDE residuals get adaptive weights (IC/BC are exact)
        self.log_w_pde = nn.Parameter(torch.zeros(n_colloc, device=device))
        self.weight_optim = torch.optim.Adam([self.log_w_pde], lr=1e-3)

    def loss_pde(self, X_colloc):
        R = self.pde_func(self, X_colloc)
        w = torch.exp(self.log_w_pde).detach()
        return (w * (R**2)).mean()

    def loss_ic(self, X_ic):
        return torch.tensor(0.0, device=device)   # exactly satisfied

    def loss_bc(self, X_bcL, X_bcR):
        return torch.tensor(0.0, device=device)   # exactly satisfied

    def update_weights(self, X_colloc):
        loss = self.loss_pde(X_colloc)
        self.weight_optim.zero_grad()
        loss.backward()
        self.weight_optim.step()


# ----------------------------------------------------------------------
# PDE definitions (to be passed as pde_func)
# ----------------------------------------------------------------------
def burgers_pde(model, X):
    """Burgers: u_t + u*u_x - nu*u_xx = 0"""
    X = X.requires_grad_(True)
    u = model.forward(X)
    u_t = model.gradients(u, X)[:, 0:1]
    u_x = model.gradients(u, X)[:, 1:2]
    u_xx = model.gradients(u_x, X)[:, 1:2]
    nu = 0.01 / torch.pi
    return u_t + u * u_x - nu * u_xx

def wave_pde(model, X):
    """Wave: u_tt - c^2*u_xx = 0. Need second derivatives."""
    X = X.requires_grad_(True)
    u = model.forward(X)
    # First derivatives
    grad = model.gradients(u, X)
    u_t = grad[:, 0:1]
    u_x = grad[:, 1:2]
    # Second derivatives
    u_tt = model.gradients(u_t, X)[:, 0:1]
    u_xx = model.gradients(u_x, X)[:, 1:2]
    c = 1.0
    return u_tt - c**2 * u_xx

def diffusion_pde(model, X):
    """Diffusion: u_t - D*u_xx = 0"""
    X = X.requires_grad_(True)
    u = model.forward(X)
    u_t = model.gradients(u, X)[:, 0:1]
    u_x = model.gradients(u, X)[:, 1:2]
    u_xx = model.gradients(u_x, X)[:, 1:2]
    D = 0.01
    return u_t - D * u_xx

# For Euler (Sod) – system of 3 equations, we need a separate model with 3 outputs.
# Here we provide a minimal example; a full implementation would extend BasePINN to multiple outputs.