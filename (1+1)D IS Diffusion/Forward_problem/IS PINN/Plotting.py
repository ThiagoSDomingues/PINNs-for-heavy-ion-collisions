import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
import seaborn as sns

from IC_1D import *
from BDNK_IS_Functions import *

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

def plot_results(model, t_eval, x_eval, alpha_ic, scriptJ_ic):
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
    scriptJ_pred_flat    = out[:, 0:1]
    alpha_pred_flat = out[:, 1:2]
    
    # Compute N_x
    grid_tensor.requires_grad_(True)
    out2 = model(grid_tensor)
    alpha_pred_flat_2 = out2[:, 1:2]
    _, alpha_x_flat = derivatives(alpha_pred_flat_2, grid_tensor)
    N_x_pred_flat = -alpha_x_flat

    # Reshape to grid
    Nt, Nx = len(t_eval), len(x_eval)
    scriptJ_pred    = scriptJ_pred_flat.view(Nt, Nx).detach().cpu().numpy()
    alpha_pred = alpha_pred_flat.view(Nt, Nx).detach().cpu().numpy()
    N_x_pred   = N_x_pred_flat.view(Nt, Nx).detach().detach().cpu().numpy()

    # Evaluate temperature, n, sigma, lambda on grid using Functions.py
    T_pred = T_func(torch.tensor(tt, dtype=dtype), torch.tensor(xx, dtype=dtype)).detach().cpu().numpy()
    v_pred = v_func(torch.tensor(tt, dtype=dtype), torch.tensor(xx, dtype=dtype)).detach().cpu().numpy()
    n_pred = n_from_alpha_func(torch.tensor(alpha_pred, dtype=dtype), torch.tensor(T_pred, dtype=dtype)).detach().cpu().numpy()
    sigma_pred = sigma_func(torch.tensor(alpha_pred, dtype=dtype), torch.tensor(T_pred, dtype=dtype)).detach().cpu().numpy()
    lambd_pred = lambd_func(torch.tensor(sigma_pred, dtype=dtype)).detach().cpu().numpy()

    # For time slices
    times  = np.linspace(0, Nt-1, 4, dtype=int)
    t_arr  = t_eval
    xc     = x_eval

    # Custom colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap("Greys_r", 256)
    vals = np.interp(np.linspace(0, 1, 256), [0.0, 1/3, 2/3, 1.0],[0.0, 0.35, 0.59, 0.81])
    cmap = cmap(vals)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(cmap)
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FuncFormatter
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams.update({
        'axes.titlesize': 25,
        'axes.labelsize': 25,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
    })

    # Plot 1: n(t,x)
    cmap = plt.get_cmap(custom_cmap)
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.48, 0.52], hspace=0.18)
    
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                              width_ratios=[1.15, 0.32], wspace=0.05)
    ax_snap = fig.add_subplot(gs_top[0, 0])
    ax_leg  = fig.add_subplot(gs_top[0, 1])
    
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                              width_ratios=[1.05, 0.06], wspace=0.05)
    ax_heat = fig.add_subplot(gs_bot[0, 0])
    cax     = fig.add_subplot(gs_bot[0, 1])
    
    cmap = plt.get_cmap(custom_cmap)
    for i, ti in enumerate(times):
        ax_snap.plot(xc, n_pred[ti],
                     color=cmap(i/(len(times)-1)), ls='--', lw=2.5,
                     label=fr'$t={t_arr[ti]:.2f}\,[\mathrm{{GeV^{{-1}}}}]$')
    ax_snap.set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    ax_snap.set_ylabel(r'$n\,{\rm [GeV^{3}]}$')
    ax_snap.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax_leg.axis('off')
    h, l = ax_snap.get_legend_handles_labels()
    ax_leg.legend(h, l, loc='center', ncol=1, frameon=False,
                  handlelength=1.2, handletextpad=0.5)
    
    pcm = ax_heat.pcolormesh(xc, t_arr, n_pred, shading='auto', cmap='gist_heat')
    ax_heat.set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    ax_heat.set_ylabel(r'$t\,{\rm [GeV^{-1}]}$')
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label(r'$n\,{\rm [GeV^{3}]}$')
    cb.ax.ticklabel_format(style='sci', scilimits=(0,0))

    fig.canvas.draw()
    
    offset = cb.ax.yaxis.get_offset_text()
    txt = offset.get_text()
    offset.set_visible(False)
    
    cb.ax.text(3.63, -0.18, txt, transform=cb.ax.transAxes, ha='right', va='bottom', fontsize=offset.get_fontsize())
    
    plt.show()

    # Plot 2: J(t,x)
    cmap = plt.get_cmap(custom_cmap)
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.48, 0.52], hspace=0.18)
    
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                              width_ratios=[1.15, 0.32], wspace=0.05)
    ax_snap = fig.add_subplot(gs_top[0, 0])
    ax_leg  = fig.add_subplot(gs_top[0, 1])
    
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                              width_ratios=[1.05, 0.06], wspace=0.05)
    ax_heat = fig.add_subplot(gs_bot[0, 0])
    cax     = fig.add_subplot(gs_bot[0, 1])
    
    cmap = plt.get_cmap(custom_cmap)
    for i, ti in enumerate(times):
        ax_snap.plot(xc, scriptJ_pred[ti],
                     color=cmap(i/(len(times)-1)), ls='--', lw=2.5,
                     label=fr'$t={t_arr[ti]:.2f}\,[\mathrm{{GeV^{{-1}}}}]$')
    ax_snap.set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    ax_snap.set_ylabel(r'$J^{0}\,{\rm [GeV^{3}]}$')
    ax_snap.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax_leg.axis('off')
    h, l = ax_snap.get_legend_handles_labels()
    ax_leg.legend(h, l, loc='center', ncol=1, frameon=False,
                  handlelength=1.2, handletextpad=0.5)
    
    pcm = ax_heat.pcolormesh(xc, t_arr, scriptJ_pred, shading='auto', cmap='gist_heat')
    ax_heat.set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    ax_heat.set_ylabel(r'$t\,{\rm [GeV^{-1}]}$')
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label(r'$J^{0}\,{\rm [GeV^{3}]}$')
    cb.ax.ticklabel_format(style='sci', scilimits=(0,0))

    fig.canvas.draw()
    
    offset = cb.ax.yaxis.get_offset_text()
    txt = offset.get_text()
    offset.set_visible(False)
    
    cb.ax.text(3.63, -0.18, txt, transform=cb.ax.transAxes, ha='right', va='bottom', fontsize=offset.get_fontsize())
    
    plt.show()

    # Plot 3: alpha(t,x)
    cmap = plt.get_cmap(custom_cmap)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for i, ti in enumerate(times):
        axs[0].plot(xc, alpha_pred[ti], color=cmap(i/(len(times)-1)), lw=2, ls='--', label=f'$t={t_arr[ti]:.3f}\,[\mathrm{{GeV^{{-1}}}}]$')
    axs[0].legend()
    axs[0].set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    axs[0].set_ylabel(r'$\alpha$')
    pcm = axs[1].pcolormesh(xc, t_arr, alpha_pred, shading='auto', cmap='gist_heat')
    fig.colorbar(pcm, ax=axs[1], label=r'$\alpha$')
    axs[1].set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    axs[1].set_ylabel(r'$t\,{\rm [GeV^{-1}]}$')
    plt.tight_layout()
    plt.show()

    # Plot 4: sigma(t,x)
    cmap = plt.get_cmap(custom_cmap)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for i, ti in enumerate(times):
        axs[0].plot(xc, sigma_pred[ti], color=cmap(i/(len(times)-1)), lw=2, ls='--', label=f'$t={t_arr[ti]:.3f}\,[\mathrm{{GeV^{{-1}}}}]$')
    axs[0].legend()
    axs[0].set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    axs[0].set_ylabel(r'$\sigma\,{\rm [GeV]}$')
    pcm = axs[1].pcolormesh(xc, t_arr, sigma_pred, shading='auto', cmap='gist_heat')
    fig.colorbar(pcm, ax=axs[1], label=r'$\sigma\,{\rm [GeV]}$')
    axs[1].set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    axs[1].set_ylabel(r'$t\,{\rm [GeV^{-1}]}$')
    plt.tight_layout()
    plt.show()

    # Plot 5: T(t,x)
    cmap = plt.get_cmap(custom_cmap)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for i, ti in enumerate(times):
        axs[0].plot(xc, T_pred[ti], color=cmap(i/(len(times)-1)), lw=2, ls='--', label=f'$t={t_arr[ti]:.3f}\,[\mathrm{{GeV^{{-1}}}}]$')
    axs[0].legend()
    axs[0].set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    axs[0].set_ylabel(r'$T\,{\rm [GeV]}$')
    pcm = axs[1].pcolormesh(xc, t_arr, T_pred, shading='auto', cmap='gist_heat')
    fig.colorbar(pcm, ax=axs[1], label=r'$T\,{\rm [GeV]}$')
    axs[1].set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    axs[1].set_ylabel(r'$t\,{\rm [GeV^{-1}]}$')
    plt.tight_layout()
    plt.show()

    # Plot 6: v(t,x)
    cmap = plt.get_cmap(custom_cmap)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for i, ti in enumerate(times):
        axs[0].plot(xc, v_pred[ti], color=cmap(i/(len(times)-1)), lw=2, ls='--', label=f'$t={t_arr[ti]:.3f}\,[\mathrm{{GeV^{{-1}}}}]$')
    axs[0].legend()
    axs[0].set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    axs[0].set_ylabel(r'$v$')
    pcm = axs[1].pcolormesh(xc, t_arr, v_pred, shading='auto', cmap='gist_heat')
    fig.colorbar(pcm, ax=axs[1], label=r'$v$')
    axs[1].set_xlabel(r'$x\,{\rm [GeV^{-1}]}$')
    axs[1].set_ylabel(r'$t\,{\rm [GeV^{-1}]}$')
    plt.tight_layout()
    plt.show()

    # Plot 7: Mass conservation check
    fig, ax = plt.subplots(figsize=(14, 5))
    
    dx = xc[1] - xc[0]
    mass_scriptJ = scriptJ_pred.sum(axis=1) * dx
    
    if mass_scriptJ[0] == 0:
        mass_scriptJ_variation = (mass_scriptJ - mass_scriptJ[0])
        ax.plot(t_arr, mass_scriptJ_variation, '-o', ms=1, lw=1)
        ax.set_ylabel(r'Variation of $\int \mathcal{J}\,dx$')
    else:
        mass_scriptJ_percent_variation = (mass_scriptJ - mass_scriptJ[0]) / mass_scriptJ[0]
        ax.plot(t_arr, mass_scriptJ_percent_variation, '-o', ms=1, lw=2, color='black')
        ax.set_ylabel(r'$\Delta \int \mathcal{J}\,dx$')
    
    ax.set_xlabel(r'$t\,{\rm [GeV^{-1}]}$')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

from matplotlib.colors import LogNorm
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

    def grad(u):
        return torch.autograd.grad(
            u, tx_tensor, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

    with torch.set_grad_enabled(True):
        out   = model(tx_tensor)     # [scriptJ, alpha]
        scriptJ    = out[:, 0:1]
        alpha = out[:, 1:2]

        t = tx_tensor[:, 0:1]
        x = tx_tensor[:, 1:2]

        T     = T_func(t, x)
        n     = n_from_alpha_func(alpha, T)
        sigma = sigma_func(alpha, T)
        tauJ  = tauJ_func(alpha, T)

        n_t       = grad(n)[:, 0:1]
        alpha_x   = grad(alpha)[:, 1:2]

        g_scriptJ = grad(scriptJ)
        scriptJ_t = g_scriptJ[:, 0:1]
        scriptJ_x = g_scriptJ[:, 1:2]

        tauJ_sigmaT_t = grad(tauJ / (sigma * T))[:, 0:1]

        sn = n_from_alpha_func(model.sA.to(device=device, dtype=dtype), T).abs().clamp_min(1e-12)
        sJ = model.sscriptJ.to(device=device, dtype=dtype).abs().clamp_min(1e-12)

        R1 = (n_t + scriptJ_x) / sn
        R2 = (
            tauJ * scriptJ_t
            + scriptJ
            + 0.5 * sigma * T * scriptJ * tauJ_sigmaT_t
            + sigma * T * alpha_x
        ) / sJ

    def to_grid(R):
        return R.detach().cpu().numpy().reshape(Nt, Nx)

    labels = [r"$|R_1|$", r"$|R_2|$"]
    data   = [to_grid(R1), to_grid(R2)]

    plt.rcParams.update({
        "axes.titlesize": 39,
        "axes.labelsize": 36,
        "xtick.labelsize": 34,
        "ytick.labelsize": 34,
    })

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.5), sharey=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.00, wspace=0.035, hspace=0.00)

    for i, (ax, lab, res) in enumerate(zip(axs, labels, data), start=1):
        res_abs = np.clip(np.abs(res), 1e-14, None)
        im = ax.pcolormesh(x_eval, t_eval, res_abs, shading='auto',
                           cmap='viridis', norm=LogNorm())
        ax.set_title(lab, pad=13)
        ax.set_xlabel(r"$x\,[\mathrm{GeV^{-1}}]$")
        if i == 1:
            ax.set_ylabel(r"$t\,[\mathrm{GeV^{-1}}]$")
        else:
            ax.tick_params(labelleft=False)

        ax.text(
            0.028, 0.957, rf"$\langle R^2_{{{i}}} \rangle = $ {np.mean(res_abs**2):.2e}",
            color='black', fontsize=36, fontweight='bold', ha='left', va='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.93, edgecolor='none', pad=8.0)
        )

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=1, pad=0.06, ticks=LogLocator(numticks=3))
        cbar.ax.tick_params(labelsize=37)

        fig.canvas.draw()

        import matplotlib.transforms as mtransforms

        fig.canvas.draw()

        shift_pts = 13
        offset_unit = fig.dpi_scale_trans

        xticks = ax.get_xticks()
        x0, x1 = ax.get_xlim()
        xmid = 0.5 * (x0 + x1)

        for val, lbl in zip(xticks, ax.get_xticklabels()):
            sgn = 1 if val < xmid else (-1 if val > xmid else 0)
            dx = (sgn * shift_pts) / 72.0
            lbl.set_transform(lbl.get_transform() +
                              mtransforms.ScaledTranslation(dx, 0, offset_unit))

        yticks = ax.get_yticks()
        y0, y1 = ax.get_ylim()
        ymid = 0.5 * (y0 + y1)

        for val, lbl in zip(yticks, ax.get_yticklabels()):
            sgn = 1 if val < ymid else (-1 if val > ymid else 0)
            dy = (sgn * shift_pts) / 72.0
            lbl.set_transform(lbl.get_transform() +
                              mtransforms.ScaledTranslation(0, dy, offset_unit))

    plt.show()

def lbfgs_inner_curve(all_inner_per_epoch):
    xs, ys = [], []
    for e, inner in enumerate(all_inner_per_epoch, start=1):
        m = len(inner)
        if m == 0:
            continue
        x = np.linspace(e-1, e, m, endpoint=False)
        xs.append(x)
        ys.append(np.asarray(inner))
        xs.append(np.array([e]))
        ys.append(np.array([inner[-1]]))
    return np.concatenate(xs), np.concatenate(ys)

def plot_combined_loss_history(adam_losses, lbfgs_hist):
    plt.rcParams.update({
        "axes.titlesize": 29,
        "axes.labelsize": 28,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 24,
    })

    adam_losses = np.asarray(adam_losses)
    xs_lbfgs, ys_lbfgs = lbfgs_inner_curve(lbfgs_hist["all_inner_per_epoch"])
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
    ax1.set_xlabel("Adam epoch")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)#
    ax1.annotate("Adam stage", (0.45, 0.98), xytext=(0, -5),
                 textcoords="offset points", xycoords="axes fraction",
                 ha="center", va="top", fontsize=27)

    # L-BFGS panel (inner losses spread across epoch buckets)
    ax2.plot(x_iters, ys_plot, lw=2, color='black')
    ax2.set_xlim(1, n_iters)
    ax2.set_xlabel("L-BFGS iteration")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)#
    ax2.annotate("L-BFGS stage", (0.5, 0.98), xytext=(0, -5),
                 textcoords="offset points", xycoords="axes fraction",
                 ha="center", va="top", fontsize=27)

    # trim spines to show the “two-panel” feel
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    # epoch ticks 0..N on the L-BFGS panel
    if n_iters > 0:
        ax2.set_xticks([0, n_iters-1])

    plt.show()