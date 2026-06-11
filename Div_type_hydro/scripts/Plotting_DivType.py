# Plotting_DivType.py

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.ticker import LogLocator
import matplotlib.cm as cm
import matplotlib.transforms as mtransforms

import seaborn as sns

from DivType_Functions import (
    T_func,
    n_func,
)

# ============================================================
# GLOBAL STYLE
# ============================================================

sns.set(style='white')

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 18,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}',
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# ============================================================
# CUSTOM COLORMAP
# ============================================================

def custom_colormap():
    cmap = cm.get_cmap("Greys_r", 256)

    vals = np.interp(
        np.linspace(0, 1, 256),
        [0.0, 1/3, 2/3, 1.0],
        [0.0, 0.35, 0.59, 0.81]
    )

    cmap = cmap(vals)

    return ListedColormap(cmap)

# ============================================================
# AUTOGRAD DERIVATIVES
# ============================================================

def derivatives(y, x):

    grad = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]

    if grad is None:
        return torch.zeros_like(y), torch.zeros_like(y)

    return grad[:, 0:1], grad[:, 1:2]

# ============================================================
# GENERIC SNAPSHOT + HEATMAP PANEL
# ============================================================

def plot_field_panel(
    field,
    x_eval,
    t_eval,
    ylabel,
    cmap_heat='gist_heat'
):

    Nt = len(t_eval)

    times = np.linspace(0, Nt-1, 4, dtype=int)

    cmap = plt.get_cmap(custom_colormap())

    fig = plt.figure(figsize=(9, 7), constrained_layout=True)

    outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[0.48, 0.52],
        hspace=0.18
    )

    gs_top = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[0],
        width_ratios=[1.15, 0.32],
        wspace=0.05
    )

    ax_snap = fig.add_subplot(gs_top[0, 0])
    ax_leg  = fig.add_subplot(gs_top[0, 1])

    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[1],
        width_ratios=[1.05, 0.06],
        wspace=0.05
    )

    ax_heat = fig.add_subplot(gs_bot[0, 0])
    cax     = fig.add_subplot(gs_bot[0, 1])

    # --------------------------------------------------------

    for i, ti in enumerate(times):

        ax_snap.plot(
            x_eval,
            field[ti],
            color=cmap(i/(len(times)-1)),
            ls='--',
            lw=2.5,
            label=fr'$t={t_eval[ti]:.2f}\,[\mathrm{{GeV^{{-1}}}}]$'
        )

    ax_snap.set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    ax_snap.set_ylabel(ylabel)

    ax_leg.axis('off')

    h, l = ax_snap.get_legend_handles_labels()

    ax_leg.legend(
        h,
        l,
        loc='center',
        frameon=False,
        handlelength=1.2,
        handletextpad=0.5
    )

    # --------------------------------------------------------

    pcm = ax_heat.pcolormesh(
        x_eval,
        t_eval,
        field,
        shading='auto',
        cmap=cmap_heat
    )

    ax_heat.set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    ax_heat.set_ylabel(r'$t\,{\rm [GeV^{-1}]}$')

    cb = fig.colorbar(pcm, cax=cax)

    cb.set_label(ylabel)

    plt.show()

# ============================================================
# MAIN RESULTS
# ============================================================

def plot_results_divtype(model, t_eval, x_eval):

    model.eval()

    p = next(model.parameters())

    device = p.device
    dtype  = p.dtype

    tt, xx = np.meshgrid(t_eval, x_eval, indexing='ij')

    tx = np.column_stack([
        tt.ravel(),
        xx.ravel()
    ])

    tx_t = torch.tensor(
        tx,
        dtype=dtype,
        device=device,
        requires_grad=True
    )

    with torch.no_grad():

        out = model(tx_t)

    q_pred = out[:, 0:1]
    alpha_pred = out[:, 1:2]

    Nt = len(t_eval)
    Nx = len(x_eval)

    q_grid = q_pred.view(Nt, Nx).cpu().numpy()

    alpha_grid = alpha_pred.view(Nt, Nx).cpu().numpy()

    # --------------------------------------------------------
    # Compute density
    # --------------------------------------------------------

    T_grid = T_func(
        torch.tensor(tt, dtype=dtype, device=device),
        torch.tensor(xx, dtype=dtype, device=device)
    )

    n_grid = n_func(
        torch.tensor(alpha_grid, dtype=dtype, device=device)
    ).cpu().numpy()

    # --------------------------------------------------------
    # Plot fields
    # --------------------------------------------------------

    plot_field_panel(
        q_grid,
        x_eval,
        t_eval,
        ylabel=r'$\mathcal{J}$'
    )

    plot_field_panel(
        alpha_grid,
        x_eval,
        t_eval,
        ylabel=r'$\alpha$'
    )

    plot_field_panel(
        n_grid,
        x_eval,
        t_eval,
        ylabel=r'$n\,{\rm [GeV^3]}$'
    )

# ============================================================
# PDE RESIDUALS
# ============================================================

def plot_pde_residuals_divtype(model, t_eval, x_eval):

    model.eval()

    p = next(model.parameters())

    device = p.device
    dtype  = p.dtype

    tt, xx = np.meshgrid(t_eval, x_eval, indexing='ij')

    tx = np.column_stack([
        tt.ravel(),
        xx.ravel()
    ])

    tx_t = torch.tensor(
        tx,
        dtype=dtype,
        device=device,
        requires_grad=True
    )

    with torch.enable_grad():

        R = model.pde_residual(tx_t)

    R1 = R[:, 0].detach().cpu().numpy().reshape(
        len(t_eval),
        len(x_eval)
    )

    R2 = R[:, 1].detach().cpu().numpy().reshape(
        len(t_eval),
        len(x_eval)
    )

    labels = [
        r"$|R_1|$",
        r"$|R_2|$"
    ]

    data = [
        np.abs(R1),
        np.abs(R2)
    ]

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(11, 5.5),
        sharey=True,
        constrained_layout=True
    )

    for i, (ax, lab, res) in enumerate(zip(axs, labels, data), start=1):

        res = np.clip(res, 1e-14, None)

        im = ax.pcolormesh(
            x_eval,
            t_eval,
            res,
            shading='auto',
            cmap='viridis',
            norm=LogNorm()
        )

        ax.set_title(lab)

        ax.set_xlabel(r"$x\,[\mathrm{GeV^{-1}}]$")

        if i == 1:
            ax.set_ylabel(r"$t\,[\mathrm{GeV^{-1}}]$")

        ax.text(
            0.03,
            0.95,
            rf"$\langle R_{i}^{2} \rangle =$ {np.mean(res**2):.2e}",
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=20,
            bbox=dict(
                facecolor='white',
                alpha=0.9,
                edgecolor='none'
            )
        )

        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation='horizontal',
            fraction=1,
            pad=0.06,
            ticks=LogLocator(numticks=3)
        )

        cbar.ax.tick_params(labelsize=18)

    plt.show()

# ============================================================
# LOSS HISTORY
# ============================================================

def plot_loss_history(adam_hist, lbfgs_hist):

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(14, 4),
        sharey=True
    )

    axs[0].plot(adam_hist, color='black')

    axs[0].set_yscale('log')

    axs[0].set_xlabel('Adam epoch')

    axs[0].set_ylabel('Loss')

    axs[0].grid(True)

    axs[0].set_title('Adam stage')

    axs[1].plot(lbfgs_hist, color='black')

    axs[1].set_yscale('log')

    axs[1].set_xlabel('L-BFGS iteration')

    axs[1].grid(True)

    axs[1].set_title('L-BFGS stage')

    plt.tight_layout()

    plt.show()

#def plot_results_divtype(model, t_eval, x_eval):
#    model.eval()
#    device = next(model.parameters()).device
#    dtype  = next(model.parameters()).dtype

#    tt, xx = np.meshgrid(t_eval, x_eval, indexing='ij')
#    tx = np.column_stack([tt.ravel(), xx.ravel()])
#    tx_t = torch.tensor(tx, dtype=dtype, device=device)

#    with torch.no_grad():
#        out = model(tx_t)
#    q_grid     = out[:, 0].view(len(t_eval), len(x_eval)).cpu().numpy()
#    alpha_grid = out[:, 1].view(len(t_eval), len(x_eval)).cpu().numpy()

    # Compute n from alpha
#    T_t = torch.tensor(tt, dtype=dtype, device=device)
#    n_grid = n_from_alpha_func(torch.tensor(alpha_grid, dtype=dtype, device=device),
#                               T_func(T_t, torch.tensor(xx, dtype=dtype, device=device))).cpu().numpy()

    # Plot fields
#    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
#    fields = [q_grid, alpha_grid, n_grid]
#    titles = [r'$q$', r'$\alpha$', r'$n$']
#    for i in range(3):
#        im = axs[0, i].pcolormesh(x_eval, t_eval, fields[i], shading='auto', cmap='gist_heat')
#        plt.colorbar(im, ax=axs[0, i])
#        axs[0, i].set_title(titles[i])
#        axs[0, i].set_xlabel('x')
#        axs[0, i].set_ylabel('t')
        # Snapshot lines
#        times_idx = np.linspace(0, len(t_eval)-1, 4, dtype=int)
#        for ti in times_idx:
#            axs[1, i].plot(x_eval, fields[i][ti], label=f't={t_eval[ti]:.1f}')
#        axs[1, i].set_xlabel('x')
#        axs[1, i].legend()
#    plt.tight_layout()
#    plt.show()


#def plot_pde_residuals_divtype(model, t_eval, x_eval):
#    model.eval()
#    device = next(model.parameters()).device
#    dtype  = next(model.parameters()).dtype

#    tt, xx = np.meshgrid(t_eval, x_eval, indexing='ij')
#    tx = np.column_stack([tt.ravel(), xx.ravel()])
#    tx_t = torch.tensor(tx, dtype=dtype, device=device, requires_grad=True)

#    with torch.enable_grad():
#        R = model.pde_residual(tx_t)
#    R1 = R[:, 0].detach().cpu().numpy().reshape(len(t_eval), len(x_eval))
#    R2 = R[:, 1].detach().cpu().numpy().reshape(len(t_eval), len(x_eval))

#    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
#    for ax, res, name in zip(axs, [R1, R2], ['R1 (charge)', 'R2 (relax)']):
#        im = ax.pcolormesh(x_eval, t_eval, np.abs(res), shading='auto',
#                           norm=LogNorm(), cmap='inferno')
#        plt.colorbar(im, ax=ax)
#        ax.set_title(name)
#        ax.set_xlabel('x')
#        ax.set_ylabel('t')
#    plt.tight_layout()
#    plt.show()
