import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import seaborn as sns

from IC_1D import *

import os, subprocess
os.environ['PATH'] = '/sw/apps/texlive/2024/bin/x86_64-linux:' + os.environ['PATH']

# Plotting style with seaborn
sns.set(style='white')
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 18,
    'axes.titlesize': 21,
    'axes.labelsize': 22,
    'legend.fontsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}',
    'figure.dpi': 300,
    'savefig.dpi': 300
})


def plot_collocation_points(X_colloc, X_ic, X_bc_L, X_bc_R, L, t_end):
    """
    Plots collocation points in (t,x), optionally with IC and BC points.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Collocation points
    Xc = X_colloc.detach().cpu().numpy()
    ax.scatter(Xc[:, 1], Xc[:, 0], s=0.5, label=r'$N_{\rm PDE}$', alpha=1, color='gray')

    # Initial condition points
    if X_ic is not None:
        Xic = X_ic.detach().cpu().numpy()
        ax.scatter(Xic[:, 1], Xic[:, 0], s=5, label=r'$N_{\rm IC}$', alpha=0.7)

    # Boundary condition points
    if X_bc_L is not None and X_bc_R is not None:
        Xbl = X_bc_L.detach().cpu().numpy()
        Xbr = X_bc_R.detach().cpu().numpy()
        ax.scatter(Xbl[:, 1], Xbl[:, 0], s=5, label=r'$N_{\rm BC,L}$', alpha=0.7)
        ax.scatter(Xbr[:, 1], Xbr[:, 0], s=5, label=r'$N_{\rm BC,R}$', alpha=0.7)

    ax.set_xlim(-L, L)
    ax.set_ylim(0, t_end)
    ax.set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    ax.set_ylabel(r'$t\,{\rm [GeV^{-1}]}$')
    ax.legend()
    ax.grid(True)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()

    if X_bc_L is None and X_bc_R is None and X_ic is None:
        ax.set_title(r'Collocation points $N_{\rm PDE}$ in $(t,x)$')

    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(prune='both'))

    plt.show()


def derivatives(y, x):
    grad = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]

    if grad is None:
        return torch.zeros_like(y), torch.zeros_like(y)

    dy_dt = grad[:, 0:1]
    dy_dx = grad[:, 1:2]
    return dy_dt, dy_dx


def plot_results(model, t_eval, x_eval, u_ic):
    model.eval()
    p = next(model.parameters())
    device, dtype = p.device, p.dtype

    # Make (t, x) grid
    tt, xx = np.meshgrid(t_eval, x_eval, indexing='ij')
    grid = np.stack([tt.flatten(), xx.flatten()], axis=1)
    grid_tensor = torch.tensor(grid, dtype=dtype, requires_grad=True).to(device)

    # PINN forward pass
    with torch.no_grad():
        out = model(grid_tensor)
    u_pred_flat = out[:, 0:1]

    # Reshape to grid
    Nt, Nx = len(t_eval), len(x_eval)
    u_pred = u_pred_flat.view(Nt, Nx).detach().cpu().numpy()

    # For time slices
    times = np.linspace(0, Nt - 1, 4, dtype=int)
    t_arr = t_eval
    xc = x_eval

    # Custom colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap('Greys_r', 256)
    vals = np.interp(np.linspace(0, 1, 256), [0.0, 1 / 3, 2 / 3, 1.0], [0.0, 0.35, 0.59, 0.81])
    cmap = cmap(vals)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(cmap)
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        'axes.titlesize': 25,
        'axes.labelsize': 25,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
    })

    # Plot: u(t,x)
    cmap = plt.get_cmap(custom_cmap)
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.48, 0.52], hspace=0.18)

    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                              width_ratios=[1.15, 0.32], wspace=0.05)
    ax_snap = fig.add_subplot(gs_top[0, 0])
    ax_leg = fig.add_subplot(gs_top[0, 1])

    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                              width_ratios=[1.05, 0.06], wspace=0.05)
    ax_heat = fig.add_subplot(gs_bot[0, 0])
    cax = fig.add_subplot(gs_bot[0, 1])

    cmap = plt.get_cmap(custom_cmap)
    for i, ti in enumerate(times):
        ax_snap.plot(xc, u_pred[ti],
                     color=cmap(i / (len(times) - 1)), ls='--', lw=2.5,
                     label=fr'$t={t_arr[ti]:.2f}\,[\mathrm{{GeV^{{-1}}}}]$')
    ax_snap.set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    ax_snap.set_ylabel(r'$u$')

    ax_leg.axis('off')
    h, l = ax_snap.get_legend_handles_labels()
    ax_leg.legend(h, l, loc='center', ncol=1, frameon=False,
                  handlelength=1.2, handletextpad=0.5)

    pcm = ax_heat.pcolormesh(xc, t_arr, u_pred, shading='auto', cmap='gist_heat')
    ax_heat.set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    ax_heat.set_ylabel(r'$t\,{\rm [GeV^{-1}]}$')
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label(r'$u$')

    plt.show()


from matplotlib.ticker import LogLocator


def plot_pde_residuals(model, t_eval, x_eval):
    model.eval()
    p = next(model.parameters())
    device, dtype = p.device, p.dtype

    t_eval = np.asarray(t_eval, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    Nt, Nx = len(t_eval), len(x_eval)
    tt, xx = np.meshgrid(t_eval, x_eval, indexing='ij')
    tx = np.column_stack([tt.ravel(), xx.ravel()])
    tx_tensor = torch.tensor(tx, dtype=dtype, device=device, requires_grad=True)

    with torch.set_grad_enabled(True):
        R = model.pde_residual(tx_tensor)

    res_abs = np.clip(np.abs(R.detach().cpu().numpy().reshape(Nt, Nx)), 1e-14, None)

    plt.rcParams.update({
        'axes.titlesize': 39,
        'axes.labelsize': 36,
        'xtick.labelsize': 34,
        'ytick.labelsize': 34,
    })

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.5), constrained_layout=True)
    im = ax.pcolormesh(x_eval, t_eval, res_abs, shading='auto', cmap='viridis', norm=LogNorm())
    ax.set_title("Burgers' residual", pad=13)
    ax.set_xlabel(r'$x\,[\mathrm{GeV^{-1}}]$')
    ax.set_ylabel(r'$t\,[\mathrm{GeV^{-1}}]$')

    ax.text(
        0.028, 0.957, rf'$\langle R^2 \rangle = $ {np.mean(res_abs**2):.2e}',
        color='black', fontsize=28, fontweight='bold', ha='left', va='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.93, edgecolor='none', pad=8.0)
    )

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.12, pad=0.14, ticks=LogLocator(numticks=3))
    cbar.ax.tick_params(labelsize=24)

    plt.show()


def lbfgs_inner_curve(all_inner_per_epoch):
    xs, ys = [], []
    for e, inner in enumerate(all_inner_per_epoch, start=1):
        m = len(inner)
        if m == 0:
            continue
        x = np.linspace(e - 1, e, m, endpoint=False)
        xs.append(x)
        ys.append(np.asarray(inner))
        xs.append(np.array([e]))
        ys.append(np.array([inner[-1]]))
    return np.concatenate(xs), np.concatenate(ys)


def plot_combined_loss_history(adam_losses, lbfgs_hist):
    plt.rcParams.update({
        'axes.titlesize': 29,
        'axes.labelsize': 28,
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'legend.fontsize': 24,
    })

    adam_losses = np.asarray(adam_losses)
    xs_lbfgs, ys_lbfgs = lbfgs_inner_curve(lbfgs_hist['all_inner_per_epoch'])
    ys_plot = np.asarray(ys_lbfgs, dtype=float).copy()
    if ys_plot.size > 0:
        ys_plot[:3] = np.nan
        ys_plot[-10:] = np.nan
        # We hide the first three values because they are almost always much larger than the following, which generates confusion and obstructs the plot
        # We also hide the last ten in case LBFGS blew up, since we keep the best model anyway, so the blow-up is just distracting
    n_iters = ys_plot.size
    x_iters = np.arange(1, n_iters + 1)

    fig = plt.figure(figsize=(14, 4))
    gs = GridSpec(1, 2, width_ratios=[3.5, 1.5], wspace=0.15)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    # Adam panel
    ax1.plot(np.arange(len(adam_losses)), adam_losses, lw=1, color='black')
    ax1.set_xlim(0, len(adam_losses))
    ax1.set_xlabel('Adam epoch')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.annotate('Adam stage', (0.45, 0.98), xytext=(0, -5),
                 textcoords='offset points', xycoords='axes fraction',
                 ha='center', va='top', fontsize=27)

    # L-BFGS panel (inner losses spread across epoch buckets)
    ax2.plot(x_iters, ys_plot, lw=2, color='black')
    ax2.set_xlim(1, n_iters)
    ax2.set_xlabel('L-BFGS iteration')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.annotate('L-BFGS stage', (0.5, 0.98), xytext=(0, -5),
                 textcoords='offset points', xycoords='axes fraction',
                 ha='center', va='top', fontsize=27)

    # trim spines to show the “two-panel” feel
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')

    # epoch ticks 0..N on the L-BFGS panel
    if n_iters > 0:
        ax2.set_xticks([0, n_iters - 1])

    plt.show()


# ==========================
# GROUND TRUTH PLOTS (scalar fields)
# ==========================
def heatmap_ground_truth(x, t, u, title="Analytical solution", 
                         xlabel="x", ylabel="t", label="u(t, x)", cmap="RdBu_r"):
    """
    Generic heatmap for scalar field u(t,x).
    u : 2D array (Nt, Nx)
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    pcm = ax.pcolormesh(x, t, u, shading="auto", cmap=cmap)
    plt.colorbar(pcm, ax=ax, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def time_slices_ground_truth(x, t, u, title="Analytical solution at selected times",
                             xlabel="x", ylabel="u", indices=None):
    """
    Generic time slices for scalar field.
    indices : list of indices into t array; default [0, Nt//4, Nt//2, 3*Nt//4, -1]
    """
    Nt = len(t)
    if indices is None:
        indices = [0, Nt//4, Nt//2, 3*Nt//4, Nt-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx in indices:
        ax.plot(x, u[idx, :], label=f"t={t[idx]:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

# ==========================
# GROUND TRUTH PLOTS (Euler/Sod – three variables)
# ==========================
def heatmap_sod_ground_truth(x, t, rho, v, p, 
                             title="Sod shock tube solution"):
    """
    Plot heatmaps for density, velocity, pressure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    data = [(rho, r'Density $\rho$', 'viridis'),
            (v,   r'Velocity $u$',   'RdBu_r'),
            (p,   r'Pressure $p$',   'plasma')]
    for ax, (field, label, cmap) in zip(axes, data):
        im = ax.pcolormesh(x, t, field, shading='auto', cmap=cmap)
        ax.set_ylabel(r'$t$')
        ax.set_title(label)
        fig.colorbar(im, ax=ax)
    axes[-1].set_xlabel(r'$x$')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def time_slices_sod_ground_truth(x, t, rho, v, p, indices=None):
    """
    Time slices for density, velocity, pressure (three subplots).
    """
    Nt = len(t)
    if indices is None:
        indices = [0, Nt//4, Nt//2, 3*Nt//4, Nt-1]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(indices)))
    
    for ax, field, label in zip(axes, [rho, v, p], 
                                ['Density', 'Velocity', 'Pressure']):
        for idx, col in zip(indices, colors):
            ax.plot(x, field[idx, :], color=col, label=f"t={t[idx]:.2f}")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', fontsize=10)
    axes[-1].set_xlabel(r'$x$')
    fig.suptitle("Sod shock tube – selected times", fontsize=14)
    plt.tight_layout()
    plt.show()