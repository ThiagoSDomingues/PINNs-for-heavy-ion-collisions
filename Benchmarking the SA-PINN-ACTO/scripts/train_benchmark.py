# train_benchmark.py
import torch
import numpy as np
from pinn_models import *
from IC_1D import ic_burgers

def train_pinn(model, X_colloc, X_ic, X_bcL, X_bcR, epochs_adam=5000, epochs_lbfgs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)

    # If SA model, store data for weight updates
    if hasattr(model, 'set_training_data'):
        model.set_training_data(X_colloc, X_ic, X_bcL, X_bcR)

    # Adam stage
    for ep in range(epochs_adam):
        optimizer.zero_grad()
        loss = model.loss_total(X_colloc, X_ic, X_bcL, X_bcR)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # Update adaptive weights (SA only)
        if isinstance(model, SAPINN):
            model.update_weights()
        elif isinstance(model, SAPINN_ACTO):
            model.update_weights(X_colloc)

        if ep % 500 == 0:
            print(f"Adam epoch {ep:5d} | loss = {loss.item():.3e}")

    # L-BFGS stage (optional)
    if epochs_lbfgs > 0:
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=epochs_lbfgs, history_size=50)
        def closure():
            lbfgs.zero_grad()
            loss = model.loss_total(X_colloc, X_ic, X_bcL, X_bcR)
            loss.backward()
            return loss
        lbfgs.step(closure)
        print(f"L-BFGS final loss = {loss.item():.3e}")

    return model

# ----------------------------------------------------------------------
# Generate data for Burgers problem (t in [0,1], x in [-1,1])
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Domain
    lb = np.array([0.0, -1.0])
    ub = np.array([1.0,  1.0])
    L = 1.0

    # Training points
    n_colloc = 20000
    n_ic = 200
    n_bc = 200

    # Collocation points (random)
    X_colloc = torch.rand(n_colloc, 2, device=device) * (torch.tensor(ub) - torch.tensor(lb)) + torch.tensor(lb)

    # Initial condition points (t=0)
    t_ic = torch.zeros(n_ic, 1, device=device)
    x_ic = torch.rand(n_ic, 1, device=device) * 2 - 1
    X_ic = torch.cat([t_ic, x_ic], dim=1)

    # Boundary points (x = -1 and x = +1)
    t_bc = torch.rand(n_bc, 1, device=device)
    x_left = -torch.ones(n_bc, 1, device=device)
    x_right = torch.ones(n_bc, 1, device=device)
    X_bcL = torch.cat([t_bc, x_left], dim=1)
    X_bcR = torch.cat([t_bc, x_right], dim=1)

    # PDE, IC, BC functions
    pde_func = burgers_pde
    ic_func = lambda x: -torch.sin(torch.pi * x / L)
    bc_func = lambda X: torch.zeros_like(X[:, 0:1])   # not used for periodic

    # Choose architecture
    layers = [2, 50, 50, 1]
    model = SAPINN_ACTO(layers, lb, ub, pde_func, ic_func, bc_func, bc_type='periodic',
                        n_colloc=n_colloc)
    model.to(device)

    # Train
    model = train_pinn(model, X_colloc, X_ic, X_bcL, X_bcR,
                       epochs_adam=8000, epochs_lbfgs=1500)

    torch.save(model.state_dict(), "burgers_sa_pinn_acto.pt")