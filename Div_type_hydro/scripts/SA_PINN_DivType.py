import torch
import torch.nn as nn
from DivType_Functions import pde_residual

class PINN_DivType(nn.Module):
    def __init__(self, Nl, Nn, lb, ub):
        super().__init__()
        self.register_buffer('lb', torch.as_tensor(lb, dtype=torch.float32))
        self.register_buffer('ub', torch.as_tensor(ub, dtype=torch.float32))
        self.register_buffer('sq', torch.tensor(1.0))   # scale for q
        self.register_buffer('sA', torch.tensor(1.0))   # scale for alpha

        # Build the network
        layers = [nn.Linear(2, Nn), nn.Tanh()]
        for _ in range(1, Nl - 1):
            layers.append(nn.Linear(Nn, Nn))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(Nn, 2))
        self.net = nn.Sequential(*layers)

        # IC interpolation functions (set after instantiation)
        self.q_ic_func = None
        self.alpha_ic_func = None

    def _scale(self, X):
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, X_phys):
        t_phys = X_phys[:, 0:1]
        x_phys = X_phys[:, 1:2]
        L = float(self.ub[1].item())
        xL1 = torch.full((1, 1), -L, dtype=X_phys.dtype, device=X_phys.device)
        xR1 = torch.full((1, 1),  L, dtype=X_phys.dtype, device=X_phys.device)
        X_L  = torch.cat([t_phys, xL1.expand_as(x_phys)], dim=1)
        X_R  = torch.cat([t_phys, xR1.expand_as(x_phys)], dim=1)

        # Evaluate network at interior and boundaries
        X_cat = torch.cat([X_phys, X_L, X_R], dim=0)
        Z_cat = self._scale(X_cat)
        raw_cat = self.net(Z_cat)

        N = X_phys.shape[0]
        raw    = raw_cat[:N]
        raw_L  = raw_cat[N:2*N]
        raw_R  = raw_cat[2*N:]

        q_raw,   alpha_raw   = raw[:, 0:1],   raw[:, 1:2]
        q_raw_L, alpha_raw_L = raw_L[:, 0:1], raw_L[:, 1:2]
        q_raw_R, alpha_raw_R = raw_R[:, 0:1], raw_R[:, 1:2]

        # Enforce periodicity
        xi = (x_phys + L) / (2.0 * L)
        q_raw_periodic     = q_raw     - (q_raw_R     - q_raw_L)     * xi
        alpha_raw_periodic = alpha_raw - (alpha_raw_R - alpha_raw_L) * xi

        # Enforce initial condition
        q_ic_x     = self.q_ic_func(x_phys)
        alpha_ic_x = self.alpha_ic_func(x_phys)
        g = torch.exp(-1.0 * t_phys)          # g(0)=1 → IC dominates at t=0
        h = 1.0 - g

        q_enf     = g * q_ic_x     + h * q_raw_periodic
        alpha_enf = g * alpha_ic_x + h * alpha_raw_periodic

        # Rescale to physical units
        q_phys     = self.sq * q_enf
        alpha_phys = self.sA * alpha_enf

        return torch.cat([q_phys, alpha_phys], dim=1)

    def pde_residual(self, x):
        x = x.requires_grad_(True)
        out = self.forward(x)
        q     = out[:, 0:1]
        alpha = out[:, 1:2]
        return pde_residual(alpha, q, x)

    def loss_pde(self, X_colloc):
        return self.pde_residual(X_colloc).pow(2)

    def ic_residual(self, *args, **kwargs):
        return (torch.zeros(1, 1, device=self.lb.device),
                torch.zeros(1, 1, device=self.lb.device))

    def loss_bc(self, xL, xR):
        outL = self.forward(xL)
        outR = self.forward(xR)
        return ((outL - outR)**2).mean()
