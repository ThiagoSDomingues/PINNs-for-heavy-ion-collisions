"""
is_pinn_fixed.py
================
Fixed version of the IS diffusion PINN.

BUGS FIXED (all caused NaN residuals + trivial-solution collapse):
------------------------------------------------------------------
BUG 1 — Wrong R2 formula
    Original:  R2 = tau*dJ_dt + J + 0.5*sig*T*J*d(tau_sig)/dt + sig*T*dalpha_dx
    The extra  0.5*sig*T*J*d(tau_sig)/dt  term is NOT in the IS equation.
    tau_sig = 12/n → d(tau_sig)/dt = −12/n² · dn/dt, so the coefficient
    −6·sig·T/n² ≈ −18 at n=0.1, amplifying early gradients by ×1200 and
    driving the network to the trivial fixed point J=0, n=const.
    Fix: drop the spurious term.  Correct R2: tau*dJ_dt + J + sig*T*dalpha_dx = 0

BUG 2 — n can go negative → tau_J = 12σT/n → negative / inf / NaN
    Fix: enforce n > 0 via softplus output head:  n = softplus(raw_n)
    This is the NaN source in the post-training diagnostics.

BUG 3 — Loss plateau at 0.0894 (trivial solution)
    The combined effect of BUG 1 + equal loss weights makes it cheaper
    for the network to satisfy R2=0 by setting J=0 and ignoring the IC.
    Fix: weight the IC loss higher (w_ic = 10) + fix BUG 1.

BUG 4 — LBFGS without strong_wolfe can diverge
    Fix: use line_search_fn="strong_wolfe".

BUG 5 — Diagnostics called residual() on fresh tensors after training
    With the WRONG R2 formula, any non-zero J produced NaN.
    Fix: BUG 1 + BUG 2 make the residual well-behaved everywhere.

Physics (IS diffusion, local rest frame, v=0):
    R1:  dn/dt + dJ/dx = 0
    R2:  tau_J · dJ/dt + J + sigma·T · dalpha/dx = 0
where
    alpha = 27·n / (Nc·Nf·T³)          (large-n regime)
    sigma = (5·Nc·Nf·T)/(12π) · (1/27 + alpha²/(243π²))
    tau_J = 12·sigma·T / n
"""

import os, math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
os.makedirs("pinn_results", exist_ok=True)

# ======================================================================
# Constants & thermodynamics
# ======================================================================

Nc = 3
Nf = 3
T  = 0.2
PI = math.pi

def alpha_from_n(n):
    return 27.0 * n / (Nc * Nf * T**3)

def sigma_fn(alpha):
    return (5.0 * Nc * Nf * T / (12.0 * PI)) * (1.0/27.0 + alpha**2 / (243.0 * PI**2))

def tau_J_fn(alpha, n):
    # clamp n away from 0 to prevent 1/n blow-up
    return 12.0 * sigma_fn(alpha) * T / n.clamp(min=1e-6)

# ======================================================================
# Domain & normalisation
# ======================================================================

t_min, t_max = 0.0, 5.0
x_min, x_max = -10.0, 10.0

def normalise(t, x):
    """Map physical (t,x) to [-1,1]² before feeding to the network."""
    t_n = 2.0 * (t - t_min) / (t_max - t_min) - 1.0
    x_n = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    return t_n, x_n

# ======================================================================
# Neural network
# ======================================================================

class PINN(nn.Module):
    """
    MLP: (t,x) → (n, J)

    The n output uses a softplus activation so that n > 0 always.
    This prevents tau_J = 12σT/n from blowing up or going negative.
    """

    def __init__(self, hidden: list[int]):
        super().__init__()
        dims   = [2] + hidden
        layers = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(nn.Tanh())
        self.trunk = nn.Sequential(*layers)

        # separate output heads
        self.head_n = nn.Linear(hidden[-1], 1)   # raw → softplus → n > 0
        self.head_J = nn.Linear(hidden[-1], 1)   # raw → J (can be negative)
        nn.init.xavier_normal_(self.head_n.weight)
        nn.init.xavier_normal_(self.head_J.weight)
        nn.init.zeros_(self.head_n.bias)
        nn.init.zeros_(self.head_J.bias)

    def forward(self, t, x):
        t_n, x_n = normalise(t, x)
        inp  = torch.cat([t_n, x_n], dim=1)
        feat = self.trunk(inp)
        # FIX BUG 2: softplus ensures n > 0 everywhere
        n = torch.nn.functional.softplus(self.head_n(feat))
        J = self.head_J(feat)
        return n, J

# ======================================================================
# Derivatives helper
# ======================================================================

def grad(y, x):
    """∂y/∂x via autograd (y is a column tensor, x requires grad)."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

# ======================================================================
# PDE residuals  (FIXED)
# ======================================================================

def residual(model, t, x):
    """
    Compute IS residuals at collocation points.

    Parameters
    ----------
    t, x : (N,1) tensors, will have requires_grad set internally.

    Returns
    -------
    R1 : dn/dt + dJ/dx
    R2 : tau_J · dJ/dt + J + sigma·T · dalpha/dx
    """
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)

    n, J = model(t, x)

    dn_dt = grad(n, t)
    dJ_dx = grad(J, x)
    dJ_dt = grad(J, t)
    dn_dx = grad(n, x)

    alpha       = alpha_from_n(n)
    sig         = sigma_fn(alpha)
    tau         = tau_J_fn(alpha, n)
    dalpha_dx   = grad(alpha, x)     # = (27/(Nc·Nf·T³)) · dn/dx

    # FIX BUG 1: remove the spurious 0.5*sig*T*J*dt_tau_sig term
    R1 = dn_dt + dJ_dx
    R2 = tau * dJ_dt + J + sig * T * dalpha_dx

    return R1, R2

# ======================================================================
# Loss function
# ======================================================================

def loss_fn(model, t_r, x_r, t0, x0, n0,
            w_pde=1.0, w_ic=10.0):
    """
    Total loss = w_pde · L_pde + w_ic · L_ic

    FIX BUG 3: w_ic=10 prevents the trivial J=0, n=const solution.
    """
    R1, R2 = residual(model, t_r, x_r)
    L_pde  = torch.mean(R1**2 + R2**2)

    n_pred, J_pred = model(t0, x0)
    L_ic = torch.mean((n_pred - n0)**2) + torch.mean(J_pred**2)

    total = w_pde * L_pde + w_ic * L_ic
    return total, L_pde.detach(), L_ic.detach()

# ======================================================================
# Collocation & initial-condition samplers
# ======================================================================

def collocation_points(N):
    t = torch.rand(N, 1) * (t_max - t_min) + t_min
    x = torch.rand(N, 1) * (x_max - x_min) + x_min
    return t.to(device), x.to(device)

def initial_condition(N=1000):
    x0 = torch.linspace(x_min, x_max, N).view(-1, 1)
    t0 = torch.zeros_like(x0)
    n0 = 0.1 + 0.02 * torch.exp(-x0**2 / 2.0)
    return t0.to(device), x0.to(device), n0.to(device)

# ======================================================================
# Training
# ======================================================================

def train_model(N_col=5000, hidden=None, n_adam=5000, n_lbfgs=2000,
                lr_adam=1e-3, w_pde=1.0, w_ic=10.0, verbose=True):

    if hidden is None:
        hidden = [128, 128, 128]

    t_r, x_r      = collocation_points(N_col)
    t0,  x0,  n0  = initial_condition(1000)

    model = PINN(hidden).to(device)

    # ── Phase 1: Adam with cosine annealing ────────────────────────────
    adam  = torch.optim.Adam(model.parameters(), lr=lr_adam)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                adam, T_max=n_adam, eta_min=1e-5)

    loss_history = {"total": [], "pde": [], "ic": []}

    for it in range(n_adam):
        adam.zero_grad()
        total, L_pde, L_ic = loss_fn(model, t_r, x_r, t0, x0, n0, w_pde, w_ic)
        total.backward()
        # gradient clipping prevents early explosion
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        adam.step()
        sched.step()

        loss_history["total"].append(total.item())
        loss_history["pde"].append(L_pde.item())
        loss_history["ic"].append(L_ic.item())

        if verbose and it % 500 == 0:
            print(f"  Adam {it:5d}/{n_adam}  "
                  f"total={total.item():.4e}  "
                  f"pde={L_pde.item():.4e}  "
                  f"ic={L_ic.item():.4e}  "
                  f"lr={adam.param_groups[0]['lr']:.2e}")

    # ── Phase 2: L-BFGS with strong Wolfe ─────────────────────────────
    # FIX BUG 4: use strong_wolfe line search for stability
    if n_lbfgs > 0:
        print("  Starting L-BFGS phase...")
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter        = n_lbfgs,
            max_eval        = n_lbfgs * 2,
            tolerance_grad  = 1e-9,
            tolerance_change= 1e-12,
            history_size    = 50,
            line_search_fn  = "strong_wolfe",   # FIX BUG 4
        )
        lbfgs_log = [0]

        def closure():
            lbfgs.zero_grad()
            total, L_pde, L_ic = loss_fn(
                model, t_r, x_r, t0, x0, n0, w_pde, w_ic)
            total.backward()
            if lbfgs_log[0] % 200 == 0:
                print(f"  L-BFGS step {lbfgs_log[0]:4d}  "
                      f"total={total.item():.4e}  "
                      f"pde={L_pde.item():.4e}  "
                      f"ic={L_ic.item():.4e}")
            lbfgs_log[0] += 1
            loss_history["total"].append(total.item())
            loss_history["pde"].append(L_pde.item())
            loss_history["ic"].append(L_ic.item())
            return total

        lbfgs.step(closure)

    return model, loss_history

# ======================================================================
# Evaluation on a dense grid
# ======================================================================

@torch.no_grad()
def evaluate_model(model, Nx=200, Nt=100):
    x = torch.linspace(x_min, x_max, Nx)
    t = torch.linspace(t_min, t_max, Nt)
    X, T_g = torch.meshgrid(x, t, indexing="ij")
    t_flat = T_g.flatten().view(-1, 1).to(device)
    x_flat = X.flatten().view(-1, 1).to(device)
    n_p, J_p = model(t_flat, x_flat)
    n = n_p.cpu().reshape(Nx, Nt).numpy()
    J = J_p.cpu().reshape(Nx, Nt).numpy()
    return x.numpy(), t.numpy(), n, J

# ======================================================================
# Residual diagnostics
# ======================================================================

def residual_diagnostics(model, N=3000):
    """Compute mean |R1|, |R2| on fresh collocation points."""
    t_r, x_r = collocation_points(N)
    R1, R2   = residual(model, t_r, x_r)
    r1 = R1.abs().mean().item()
    r2 = R2.abs().mean().item()
    return r1, r2

# ======================================================================
# Plots
# ======================================================================

def plot_all(x, t, n, J, loss_history, R1_grid, R2_grid, outdir="pinn_results"):
    """Generate all diagnostic figures and save them."""

    # 1. Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    iters = np.arange(len(loss_history["total"]))
    axes[0].semilogy(iters, loss_history["total"], lw=2, label="Total")
    axes[0].semilogy(iters, loss_history["pde"],   lw=1.5, ls="--", label="PDE")
    axes[0].semilogy(iters, loss_history["ic"],    lw=1.5, ls="-.", label="IC")
    axes[0].axvline(5000, color="grey", ls=":", alpha=0.7, label="Adam→L-BFGS")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Convergence curve")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogy(iters, loss_history["pde"], lw=2, color="tab:orange")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("PDE loss")
    axes[1].set_title("PDE residual loss")
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{outdir}/loss_curves.png", dpi=150)
    plt.close()

    # 2. Solution heatmaps
    # n has shape (Nx, Nt); imshow expects (rows=Nt, cols=Nx) → transpose
    # extent = [t_min, t_max, x_min, x_max], origin='lower' puts x_min at bottom
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, field, label in zip(axes, [n, J], ["$n(x,t)$", "$J(x,t)$"]):
        vmax = float(np.max(np.abs(field)))
        im = ax.imshow(
            field.T,                          # (Nt, Nx) — rows=t, cols=x
            origin="lower",
            aspect="auto",
            extent=[x[0], x[-1], t[0], t[-1]],
            cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
        )
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("$x$"); ax.set_ylabel("$t$"); ax.set_title(label)
    plt.suptitle("IS diffusion PINN — solution fields")
    plt.tight_layout()
    plt.savefig(f"{outdir}/solution_heatmap.png", dpi=150)
    plt.close()

    # 3. Residual heatmaps  (R1_grid, R2_grid have shape (Nx_res, Nt_res))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, R, label in zip(axes, [R1_grid, R2_grid],
                             ["|R₁| charge conservation",
                              "|R₂| IS relaxation"]):
        im = ax.imshow(
            np.log10(R + 1e-12).T,            # (Nt_res, Nx_res)
            origin="lower",
            aspect="auto",
            extent=[x_min, x_max, t_min, t_max],
            cmap="inferno",
        )
        plt.colorbar(im, ax=ax, label="$\\log_{10}|R|$")
        ax.set_xlabel("$x$"); ax.set_ylabel("$t$"); ax.set_title(label)
    plt.suptitle("IS diffusion PINN — PDE residuals ($\\log_{10}$)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/residual_heatmap.png", dpi=150)
    plt.close()

    # 4. Snapshots at 4 time slices
    Nt_grid = len(t)
    t_idx   = [0, Nt_grid//4, Nt_grid//2, Nt_grid-1]   # avoid -1 index for t[idx]
    colors  = plt.cm.plasma(np.linspace(0.1, 0.9, len(t_idx)))
    fig, axes = plt.subplots(2, len(t_idx), figsize=(15, 7), sharex=True)
    for col, (idx, c) in enumerate(zip(t_idx, colors)):
        t_val = t[idx]
        axes[0, col].plot(x, n[:, idx], lw=2, color=c)
        axes[0, col].set_title(f"$t = {t_val:.2f}$")
        axes[0, col].set_ylabel("$n(x,t)$")
        axes[0, col].grid(True, alpha=0.3)
        axes[1, col].plot(x, J[:, idx], lw=2, color=c)
        axes[1, col].set_xlabel("$x$")
        axes[1, col].set_ylabel("$J(x,t)$")
        axes[1, col].grid(True, alpha=0.3)
    plt.suptitle("IS diffusion PINN — snapshots")
    plt.tight_layout()
    plt.savefig(f"{outdir}/snapshots.png", dpi=150)
    plt.close()

    print(f"  Plots saved to {outdir}/")

# ======================================================================
# Convergence tests
# ======================================================================

def convergence_test_collocation(n_col_list=None, outdir="pinn_results"):
    """Error vs N_col (fix architecture, vary collocation points)."""
    if n_col_list is None:
        n_col_list = [500, 1000, 2000, 4000, 8000]
    errors = []
    losses = []
    print(f"\n{'='*55}")
    print("  Convergence test 1: Collocation points")
    print(f"{'='*55}")
    for nc in n_col_list:
        print(f"\n  N_col = {nc}")
        m, hist = train_model(N_col=nc, n_adam=3000, n_lbfgs=500, verbose=False)
        r1, r2 = residual_diagnostics(m)
        fin     = hist["total"][-1]
        errors.append((r1 + r2) / 2)
        losses.append(fin)
        print(f"    mean|R1|={r1:.3e}  mean|R2|={r2:.3e}  loss={fin:.3e}")
    np.savez(f"{outdir}/conv_col.npz",
             N_cols=n_col_list, errors=errors, losses=losses)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].loglog(n_col_list, errors, "o-", lw=2, markersize=7)
    axes[0].set_xlabel("$N_\\mathrm{col}$")
    axes[0].set_ylabel("Mean $|R|$")
    axes[0].set_title("Collocation convergence: residual")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].loglog(n_col_list, losses, "s-", lw=2, color="tab:orange", markersize=7)
    axes[1].set_xlabel("$N_\\mathrm{col}$")
    axes[1].set_ylabel("Final loss")
    axes[1].set_title("Collocation convergence: loss")
    axes[1].grid(True, which="both", alpha=0.3)
    plt.suptitle("IS-PINN: collocation point convergence")
    plt.tight_layout()
    plt.savefig(f"{outdir}/conv_col.png", dpi=150)
    plt.close()
    print(f"\n  → {outdir}/conv_col.png")

def convergence_test_architecture(outdir="pinn_results"):
    """Error vs network width/depth."""
    archs = {
        "small [32,32]"      : [32, 32],
        "medium [64,64,64]"  : [64, 64, 64],
        "large [128,128,128]": [128, 128, 128],
        "deep [64,64,64,64]" : [64, 64, 64, 64],
    }
    print(f"\n{'='*55}")
    print("  Convergence test 2: Architecture")
    print(f"{'='*55}")
    labels, errors, losses, n_params_list = [], [], [], []
    for label, hidden in archs.items():
        print(f"\n  Arch: {label}")
        m, hist = train_model(hidden=hidden, n_adam=3000, n_lbfgs=500, verbose=False)
        r1, r2   = residual_diagnostics(m)
        fin      = hist["total"][-1]
        n_par    = sum(p.numel() for p in m.parameters())
        labels.append(label.split("[")[0].strip())
        errors.append((r1 + r2) / 2)
        losses.append(fin)
        n_params_list.append(n_par)
        print(f"    params={n_par}  mean|R|={errors[-1]:.3e}  loss={fin:.3e}")
    np.savez(f"{outdir}/conv_arch.npz",
             labels=labels, errors=errors, losses=losses, n_params=n_params_list)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x_pos = np.arange(len(labels))
    axes[0].bar(x_pos, errors, color="tab:blue", alpha=0.8)
    axes[0].set_xticks(x_pos); axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Mean $|R|$"); axes[0].set_yscale("log")
    axes[0].set_title("Architecture: residual"); axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x_pos, losses, color="tab:orange", alpha=0.8)
    axes[1].set_xticks(x_pos); axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Final loss"); axes[1].set_yscale("log")
    axes[1].set_title("Architecture: loss"); axes[1].grid(axis="y", alpha=0.3)
    plt.suptitle("IS-PINN: architecture convergence")
    plt.tight_layout()
    plt.savefig(f"{outdir}/conv_arch.png", dpi=150)
    plt.close()
    print(f"\n  → {outdir}/conv_arch.png")

# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":

    # ── Train baseline model ───────────────────────────────────────────
    print("Training baseline model...")
    model, loss_history = train_model(
        N_col   = 5000,
        hidden  = [128, 128, 128],
        n_adam  = 5000,
        n_lbfgs = 2000,
        lr_adam = 1e-3,
        w_pde   = 1.0,
        w_ic    = 10.0,
    )

    # ── Save model & loss ─────────────────────────────────────────────
    torch.save(model.state_dict(), "pinn_results/model.pt")
    np.savez("pinn_results/loss.npz",
             total=loss_history["total"],
             pde  =loss_history["pde"],
             ic   =loss_history["ic"])

    # ── Evaluate on dense grid ────────────────────────────────────────
    x, t, n, J = evaluate_model(model)
    np.savez("pinn_results/solution.npz", x=x, t=t, n=n, J=J)

    # ── Residual diagnostics ──────────────────────────────────────────
    print("\nComputing residual diagnostics...")
    R1_mean, R2_mean = residual_diagnostics(model)
    print(f"  Mean |R1|: {R1_mean:.4e}")
    print(f"  Mean |R2|: {R2_mean:.4e}")

    # ── Residuals on the solution grid ────────────────────────────────
    Nx, Nt = 80, 40   # coarser grid for speed (grad computation)
    x_g  = torch.linspace(x_min, x_max, Nx)
    t_g  = torch.linspace(t_min, t_max, Nt)
    Xg, Tg = torch.meshgrid(x_g, t_g, indexing="ij")
    t_flat = Tg.flatten().view(-1, 1).to(device)
    x_flat = Xg.flatten().view(-1, 1).to(device)
    R1, R2 = residual(model, t_flat, x_flat)
    R1_grid = R1.abs().detach().cpu().reshape(Nx, Nt).numpy()
    R2_grid = R2.abs().detach().cpu().reshape(Nx, Nt).numpy()
    np.savez("pinn_results/residuals.npz",
             x=x_g.numpy(), t=t_g.numpy(), R1=R1_grid, R2=R2_grid)

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_all(x, t, n, J, loss_history, R1_grid, R2_grid)

    # ── Convergence tests ─────────────────────────────────────────────
    convergence_test_collocation()
    convergence_test_architecture()

    print("\nDone. All results in pinn_results/")
