import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
import torch.autograd

# Import your physics helper functions from wherever you keep them
from BDNK_Functions import *

class PINN_BDNK_1D(nn.Module):
    def __init__(self, Nl, Nn, lb, ub):
        super().__init__()
        # Scaling from physical units to [-1,1]
        self.register_buffer('lb', torch.as_tensor(lb, dtype=torch.get_default_dtype()))
        self.register_buffer('ub', torch.as_tensor(ub, dtype=torch.get_default_dtype()))
        self.register_buffer('sJ0', torch.tensor(1.0, dtype=torch.get_default_dtype()))
        self.register_buffer('sA',  torch.tensor(1.0, dtype=torch.get_default_dtype()))

        # Base network (produces raw candidates)
        self.net = nn.Sequential()
        self.net.add_module('Linear_layer_1', nn.Linear(2, Nn))
        self.net.add_module('Tanh_layer_1', nn.Tanh())
        for num in range(2, Nl):
            self.net.add_module(f'Linear_layer_{num}', nn.Linear(Nn, Nn))
            self.net.add_module(f'Tanh_layer_{num}', nn.Tanh())
        self.net.add_module('Linear_layer_final', nn.Linear(Nn, 2))  # [J0_raw, alpha_raw]

        # IC algebraic enforcement toggles
        self.J0_ic_func = None
        self.alpha_ic_func = None

    # Helpers
    def _scale(self, X):  # (t,x) in physical -> [-1,1]
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

    # Forward with hard IC
    def forward(self, X_phys):
        t_phys = X_phys[:, 0:1]
        x_phys = X_phys[:, 1:2]
        
        L = float(self.ub[1].item())
        xL1 = torch.full((1, 1), -L, dtype=X_phys.dtype, device=X_phys.device)
        xR1 = torch.full((1, 1),  L, dtype=X_phys.dtype, device=X_phys.device)
        X_L  = torch.cat([t_phys, xL1.expand_as(x_phys)], dim=1)
        X_R  = torch.cat([t_phys, xR1.expand_as(x_phys)], dim=1)
    
        # Raw network outputs
        X_cat = torch.cat([X_phys, X_L, X_R], dim=0)
        Z_cat = self._scale(X_cat)
        raw_cat = self.net(Z_cat)
    
        N = X_phys.shape[0]
        raw    = raw_cat[:N]
        raw_L  = raw_cat[N:2*N]
        raw_R  = raw_cat[2*N:]
    
        J0_raw,   alpha_raw   = raw[:, 0:1],   raw[:, 1:2]
        J0_raw_L, alpha_raw_L = raw_L[:, 0:1], raw_L[:, 1:2]
        J0_raw_R, alpha_raw_R = raw_R[:, 0:1], raw_R[:, 1:2]
    
        if self.J0_ic_func is None or self.alpha_ic_func is None:
            raise RuntimeError("IC functions must be set before forward pass.")
    
        # Enforce periodic boundary conditions on raw outputs
        xi = (x_phys + L) / (2.0 * L)
        J0_raw_periodic    = J0_raw    - (J0_raw_R    - J0_raw_L)    * xi
        alpha_raw_periodic = alpha_raw - (alpha_raw_R - alpha_raw_L) * xi
        
        # Enforce initial condition
        J0_ic_x    = self.J0_ic_func(x_phys)
        alpha_ic_x = self.alpha_ic_func(x_phys)
    
        g = torch.exp(-1.0 * t_phys)  # g(0)=1
        h = 1.0 - g                   # h(0)=0
    
        J0_enf    = g * J0_ic_x    + h * J0_raw_periodic
        alpha_enf = g * alpha_ic_x + h * alpha_raw_periodic
    
        # Scale back to physical units
        J0_phys    = self.sJ0 * J0_enf
        alpha_phys = self.sA  * alpha_enf
    
        return torch.cat([J0_phys, alpha_phys], dim=1)


    # Autograd utilities
    def gradients(self, y, x, retain_graph=True):
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=retain_graph
        )[0]

    # Residuals
    def pde_residual(self, x):
        x = x.requires_grad_(True)
        out   = self.forward(x)            # [J0, alpha]
        J0    = out[:, 0:1]
        alpha = out[:, 1:2]

        t  = x[:, 0:1]
        xx = x[:, 1:2]

        T     = T_func(t, xx)
        v     = v_func(t, xx)
        gamma = gamma_func(v)

        n     = n_from_alpha_func(alpha, T)
        sigma = sigma_func(alpha, T)
        lambd = lambd_func(sigma)

        g_alpha  = self.gradients(alpha, x)
        alpha_t  = g_alpha[:, 0:1]
        alpha_x  = g_alpha[:, 1:2]
        N_x      = -alpha_x

        J0_t     = self.gradients(J0, x)[:, 0:1]

        N_0 = N_0_func(lambd, sigma, T, J0, n, N_x, v)
        Jx  = Jx_func(n, sigma, lambd, T, N_x, N_0, v)
        Jx_x = self.gradients(Jx, x)[:, 1:2]

        R1 = J0_t + Jx_x
        R2 = alpha_t + N_0

        return torch.cat([R1/self.sJ0, R2/self.sA], dim=1)

    def loss_pde(self, X_colloc):
        return self.pde_residual(X_colloc)**2

    def ic_residual(self, *args, **kwargs):
        return (torch.zeros(1,1, device=self.lb.device), torch.zeros(1,1, device=self.lb.device))

    def loss_bc(self, xL, xR):
        outL = self.forward(xL); outR = self.forward(xR)
        return ((outL - outR)**2).mean()

    def loss_mass(self, t_grid, x_grid):
        device = x_grid.device
        Nt, Nx = len(t_grid), len(x_grid)
        tt, xx = torch.meshgrid(t_grid, x_grid, indexing='ij')
        tx_mass = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1).to(device)
        out = self.forward(tx_mass)
        J0_pred = out[:, 0].view(Nt, Nx)
        dx = (x_grid[1] - x_grid[0]).item()
        mass = J0_pred.sum(dim=1) * dx
        mass0 = mass[0]
        return ((mass - mass0) ** 2).mean()