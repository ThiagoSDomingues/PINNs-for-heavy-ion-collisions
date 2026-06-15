import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
import torch.autograd


class PINN_Burgers_1D(nn.Module):
    def __init__(self, Nl, Nn, lb, ub, nu=0.01 / torch.pi, BC='periodic', ACTO=True):
        super().__init__()
        # Scaling from physical units to [-1,1]
        self.register_buffer('lb', torch.as_tensor(lb, dtype=torch.get_default_dtype()))
        self.register_buffer('ub', torch.as_tensor(ub, dtype=torch.get_default_dtype()))
        self.register_buffer('sU', torch.tensor(1.0, dtype=torch.get_default_dtype()))
        self.register_buffer('nu', torch.as_tensor(nu, dtype=torch.get_default_dtype()))

        # Boundary-condition type and ACTO toggle
        self.BC = BC.lower()
        self.ACTO = bool(ACTO)

        # Base network (produces raw candidates)
        self.net = nn.Sequential()
        self.net.add_module('Linear_layer_1', nn.Linear(2, Nn))
        self.net.add_module('Tanh_layer_1', nn.Tanh())
        for num in range(2, Nl):
            self.net.add_module(f'Linear_layer_{num}', nn.Linear(Nn, Nn))
            self.net.add_module(f'Tanh_layer_{num}', nn.Tanh())
        self.net.add_module('Linear_layer_final', nn.Linear(Nn, 1))  # [u_raw]

        # ACTO functions (scaled)
        self.u_ic_func = None
        self.u_bc_func = None

        # Soft-loss functions (physical scale)
        self.u_ic_phys_func = None
        self.u_bc_phys_func = None

    # Helpers
    def _scale(self, X):  # (t,x) in physical -> [-1,1]
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

    # Forward
    def forward(self, X_phys):
        t_phys = X_phys[:, 0:1]
        x_phys = X_phys[:, 1:2]

        L = float(self.ub[1].item())
        xL1 = torch.full((1, 1), -L, dtype=X_phys.dtype, device=X_phys.device)
        xR1 = torch.full((1, 1),  L, dtype=X_phys.dtype, device=X_phys.device)
        X_L = torch.cat([t_phys, xL1.expand_as(x_phys)], dim=1)
        X_R = torch.cat([t_phys, xR1.expand_as(x_phys)], dim=1)

        # Raw network outputs
        X_cat = torch.cat([X_phys, X_L, X_R], dim=0)
        Z_cat = self._scale(X_cat)
        raw_cat = self.net(Z_cat)

        N = X_phys.shape[0]
        raw   = raw_cat[:N]
        raw_L = raw_cat[N:2 * N]
        raw_R = raw_cat[2 * N:]

        u_raw   = raw[:, 0:1]
        u_raw_L = raw_L[:, 0:1]
        u_raw_R = raw_R[:, 0:1]

        if self.ACTO:
            if self.u_ic_func is None:
                raise RuntimeError('ACTO=True, but u_ic_func was not set.')

            xi = (x_phys + L) / (2.0 * L)

            if self.BC == 'periodic':
                # Enforce periodic boundary conditions on raw output
                u_raw_bc = u_raw - (u_raw_R - u_raw_L) * xi

            elif self.BC == 'dirichlet':
                if self.u_bc_func is None:
                    raise RuntimeError("Dirichlet BC selected with ACTO=True, but u_bc_func was not set.")

                # Enforce Dirichlet boundary conditions on raw output
                boundary_factor = 4.0 * xi * (1.0 - xi)

                u_bc_L = self.u_bc_func(X_L)
                u_bc_R = self.u_bc_func(X_R)
                u_bc_blend = (1.0 - xi) * u_bc_L + xi * u_bc_R

                u_raw_bc = u_bc_blend + boundary_factor * u_raw

            else:
                raise ValueError(f"Unsupported BC type: {self.BC}. Use 'periodic' or 'dirichlet'.")

            # Enforce initial condition
            u_ic_x = self.u_ic_func(x_phys)

            g = torch.exp(-10.0 * t_phys / self.ub[0])  # g(0)=1
            h = 1.0 - g                                 # h(0)=0

            u_enf = g * u_ic_x + h * u_raw_bc
            u_phys = self.sU * u_enf
        else:
            # No output transform logic
            u_phys = self.sU * u_raw

        return u_phys

    # Autograd utilities
    def gradients(self, y, x, retain_graph=True):
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=retain_graph
        )[0]

    # Residuals
    def pde_residual(self, x):
        x = x.requires_grad_(True)
        u = self.forward(x)

        g_u = self.gradients(u, x)
        u_t = g_u[:, 0:1]
        u_x = g_u[:, 1:2]
        u_xx = self.gradients(u_x, x)[:, 1:2]

        R = u_t + u * u_x - self.nu * u_xx

        return R / self.sU

    def loss_pde(self, X_colloc):
        return self.pde_residual(X_colloc) ** 2

    def ic_residual(self, X_ic, u_ic_target=None):
        if self.ACTO:
            return torch.zeros((X_ic.shape[0], 1), dtype=X_ic.dtype, device=X_ic.device)

        out = self.forward(X_ic)
        if u_ic_target is not None:
            target = u_ic_target
        elif self.u_ic_phys_func is not None:
            target = self.u_ic_phys_func(X_ic[:, 1:2])
        else:
            raise RuntimeError("ACTO=False, but no initial-condition target/function was provided.")
        return out - target

    def loss_ic(self, X_ic, u_ic_target=None):
        return (self.ic_residual(X_ic, u_ic_target=u_ic_target) ** 2).mean()

    def bc_residual(self, xL, xR):
        if self.ACTO:
            n = xL.shape[0]
            if self.BC == 'dirichlet':
                n = 2 * n
            return torch.zeros((n, 1), dtype=xL.dtype, device=xL.device)

        if self.BC == 'periodic':
            outL = self.forward(xL)
            outR = self.forward(xR)
            return outL - outR

        if self.BC == 'dirichlet':
            if self.u_bc_phys_func is None:
                raise RuntimeError("Dirichlet BC selected with ACTO=False, but u_bc_phys_func was not set.")
            outL = self.forward(xL)
            outR = self.forward(xR)
            bcL = self.u_bc_phys_func(xL)
            bcR = self.u_bc_phys_func(xR)
            return torch.cat([outL - bcL, outR - bcR], dim=0)

        raise ValueError(f"Unsupported BC type: {self.BC}. Use 'periodic' or 'dirichlet'.")

    def loss_bc(self, xL, xR):
        return (self.bc_residual(xL, xR) ** 2).mean()

    def loss_mass(self, t_grid, x_grid):
        device = x_grid.device
        Nt, Nx = len(t_grid), len(x_grid)
        tt, xx = torch.meshgrid(t_grid, x_grid, indexing='ij')
        tx_mass = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1).to(device)
        u_pred = self.forward(tx_mass).view(Nt, Nx)
        dx = (x_grid[1] - x_grid[0]).item()
        mass = u_pred.sum(dim=1) * dx
        mass0 = mass[0]
        return ((mass - mass0) ** 2).mean()


PINN_BDNK_1D = PINN_Burgers_1D
