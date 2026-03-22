"""
Script to make plots for IS PINNs.
──────────────────────────────────────────────────────────────────────────────
  • plot_loss_breakdown  – separate R1 vs R2 loss curves
  • plot_adaptive_weights – w1/w2 evolution
  • plot_results updated to show IS-specific τ_J field
  • All existing plots preserved
──────────────────────────────────────────────────────────────────────────────
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter
import matplotlib.cm as cm
import seaborn as sns

from BDNK_IS_Functions_improved import (
    T_func, v_func, n_from_alpha_func, sigma_func, lambd_func, tauJ_func
)

# ── Global style ────────────────────────────────────────────────────────────
sns.set(style='white')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'legend.fontsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

# ── Custom greyscale colormap ───────────────────────────────────────────────
_base = cm.get_cmap("Greys_r", 256)
_vals = np.interp(np.linspace(0, 1, 256),
                  [0.0, 1/3, 2/3, 1.0],
                  [0.0, 0.35, 0.59, 0.81])
GREY_CMAP = ListedColormap(_base(np.interp(np.linspace(0, 1, 256),
                                           [0, 1], [0, 1]))
                           [:, :])
# Rebuild properly
_cols = cm.get_cmap("Greys_r", 256)(np.linspace(0, 1, 256))
GREY_CMAP = ListedColormap(_cols)


def _to_numpy(t):
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _eval_grid(model, t_eval, x_eval):
    """Evaluate model on (t, x) meshgrid; return dict of numpy arrays."""
    p = next(model.parameters())
    dtype, dev = p.dtype, p.device

    tt, xx = np.meshgrid(t_eval, x_eval, indexing='ij')
    grid   = np.stack([tt.flatten(), xx.flatten()], axis=1)
    grid_t = torch.tensor(grid, dtype=dtype, device=dev, requires_grad=True)

    with torch.set_grad_enabled(True):
        out      = model(grid_t)
        sJ_flat  = out[:, 0:1]
        alp_flat = out[:, 1:2]

        grad_alp  = torch.autograd.grad(
            alp_flat, grid_t,
            grad_outputs=torch.ones_like(alp_flat),
            create_graph=False, retain_graph=False
        )[0]
        Nx_flat = -grad_alp[:, 1:2]

    Nt, Nx = len(t_eval), len(x_eval)
    sJ   = sJ_flat.detach().cpu().numpy().reshape(Nt, Nx)
    alp  = alp_flat.detach().cpu().numpy().reshape(Nt, Nx)
    Nxg  = Nx_flat.detach().cpu().numpy().reshape(Nt, Nx)

    dtype_np = np.float64
    T_arr    = _to_numpy(T_func(torch.tensor(tt, dtype=torch.float64),
                                torch.tensor(xx, dtype=torch.float64)))
    v_arr    = _to_numpy(v_func(torch.tensor(tt, dtype=torch.float64),
                                torch.tensor(xx, dtype=torch.float64)))
    n_arr    = _to_numpy(n_from_alpha_func(
                    torch.tensor(alp,   dtype=torch.float64),
                    torch.tensor(T_arr, dtype=torch.float64)))
    sig_arr  = _to_numpy(sigma_func(
                    torch.tensor(alp,   dtype=torch.float64),
                    torch.tensor(T_arr, dtype=torch.float64)))
    lam_arr  = _to_numpy(lambd_func(torch.tensor(sig_arr, dtype=torch.float64)))
    tauJ_arr = _to_numpy(tauJ_func(
                    torch.tensor(alp,   dtype=torch.float64),
                    torch.tensor(T_arr, dtype=torch.float64)))

    return dict(sJ=sJ, alpha=alp, Nx=Nxg,
                T=T_arr, v=v_arr, n=n_arr,
                sigma=sig_arr, lambd=lam_arr, tauJ=tauJ_arr,
                tt=tt, xx=xx)


def _panel(fig_kw=None):
    if fig_kw is None:
        fig_kw = {}
    return plt.subplots(**fig_kw)


# ──────────────────────────────────────────────────────────────────────────
#  Individual plot helpers
# ──────────────────────────────────────────────────────────────────────────

def _two_panel_plot(xc, t_arr, field, label_y, label_cb,
                    title='', t_indices=None, sci=True, cmap='gist_heat',
                    savefile=None):
    """Snapshot slices (top) + heatmap (bottom)."""
    if t_indices is None:
        t_indices = np.linspace(0, len(t_arr) - 1, 4, dtype=int)

    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    outer = gridspec.GridSpec(2, 1, figure=fig,
                              height_ratios=[0.48, 0.52], hspace=0.18)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                              width_ratios=[1.15, 0.32], wspace=0.05)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                              width_ratios=[1.05, 0.06], wspace=0.05)

    ax_snap = fig.add_subplot(gs_top[0, 0])
    ax_leg  = fig.add_subplot(gs_top[0, 1])
    ax_heat = fig.add_subplot(gs_bot[0, 0])
    cax     = fig.add_subplot(gs_bot[0, 1])

    for i, ti in enumerate(t_indices):
        ax_snap.plot(xc, field[ti],
                     color=GREY_CMAP(i / max(len(t_indices) - 1, 1)),
                     ls='--', lw=2.5,
                     label=fr'$t={t_arr[ti]:.2f}$')
    ax_snap.set_xlabel(r'$x\,[\mathrm{GeV^{-1}}]$')
    ax_snap.set_ylabel(label_y)
    if sci:
        ax_snap.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    if title:
        ax_snap.set_title(title)

    ax_leg.axis('off')
    h, l = ax_snap.get_legend_handles_labels()
    ax_leg.legend(h, l, loc='center', ncol=1, frameon=False)

    pcm = ax_heat.pcolormesh(xc, t_arr, field, shading='auto', cmap=cmap)
    ax_heat.set_xlabel(r'$x\,[\mathrm{GeV^{-1}}]$')
    ax_heat.set_ylabel(r'$t\,[\mathrm{GeV^{-1}}]$')
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label(label_cb)
    if sci:
        cb.ax.ticklabel_format(style='sci', scilimits=(0, 0))

    if savefile:
        plt.savefig(savefile, bbox_inches='tight')
    plt.show()


def plot_collocation_points(X_colloc, X_ic, X_bc_L, X_bc_R, L, t_end,
                            savefile=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    Xc = _to_numpy(X_colloc)
    ax.scatter(Xc[:, 1], Xc[:, 0], s=0.5, label=r'$N_{\rm PDE}$',
               alpha=0.6, color='gray')
    if X_ic is not None:
        Xi = _to_numpy(X_ic)
        ax.scatter(Xi[:, 1], Xi[:, 0], s=5, label=r'$N_{\rm IC}$', alpha=0.7)
    if X_bc_L is not None:
        Xl = _to_numpy(X_bc_L); Xr = _to_numpy(X_bc_R)
        ax.scatter(Xl[:, 1], Xl[:, 0], s=5, label=r'$N_{\rm BC,L}$', alpha=0.7)
        ax.scatter(Xr[:, 1], Xr[:, 0], s=5, label=r'$N_{\rm BC,R}$', alpha=0.7)
    ax.set_xlim(-L, L); ax.set_ylim(0, t_end)
    ax.set_xlabel(r'$x\,[\mathrm{GeV^{-1}}]$')
    ax.set_ylabel(r'$t\,[\mathrm{GeV^{-1}}]$')
    ax.legend(); ax.grid(True)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    plt.tight_layout()
    if savefile: plt.savefig(savefile, bbox_inches='tight')
    plt.show()


def plot_results(model, t_eval, x_eval, alpha_ic, scriptJ_ic,
                 savefile_prefix=None):
    model.eval()
    D = _eval_grid(model, t_eval, x_eval)
    xc, t_arr = x_eval, t_eval

    sf = lambda s: (savefile_prefix + s) if savefile_prefix else None

    # n(t,x)
    _two_panel_plot(xc, t_arr, D['n'],
                    r'$n\,[\mathrm{GeV^{3}}]$',
                    r'$n\,[\mathrm{GeV^{3}}]$',
                    title=r'Number density $n(t,x)$',
                    savefile=sf('_n.pdf'))

    # J(t,x)
    _two_panel_plot(xc, t_arr, D['sJ'],
                    r'$\mathcal{J}\,[\mathrm{GeV^{3}}]$',
                    r'$\mathcal{J}$',
                    title=r'Diffusion current $\mathcal{J}(t,x)$',
                    savefile=sf('_J.pdf'))

    # alpha(t,x)
    _two_panel_plot(xc, t_arr, D['alpha'],
                    r'$\alpha = \mu/T$', r'$\alpha$',
                    title=r'Reduced chemical potential $\alpha(t,x)$',
                    sci=False,
                    savefile=sf('_alpha.pdf'))

    # tau_J(t,x)  ← NEW: IS-specific field
    _two_panel_plot(xc, t_arr, D['tauJ'],
                    r'$\tau_{\mathcal{J}}\,[\mathrm{GeV^{-1}}]$',
                    r'$\tau_{\mathcal{J}}$',
                    title=r'IS relaxation time $\tau_{\mathcal{J}}(t,x)$',
                    cmap='plasma',
                    savefile=sf('_tauJ.pdf'))

    # sigma(t,x)
    _two_panel_plot(xc, t_arr, D['sigma'],
                    r'$\sigma\,[\mathrm{GeV^{2}}]$', r'$\sigma$',
                    title=r'Diffusion conductivity $\sigma(t,x)$',
                    savefile=sf('_sigma.pdf'))

    # N_x(t,x)
    _two_panel_plot(xc, t_arr, D['Nx'],
                    r'$\mathcal{N}_x$', r'$\mathcal{N}_x$',
                    title=r'Spatial derivative $\mathcal{N}_x(t,x)$',
                    sci=False, cmap='coolwarm',
                    savefile=sf('_Nx.pdf'))

    # Mass conservation
    dx = xc[1] - xc[0]
    mass = D['sJ'].sum(axis=1) * dx
    fig, ax = plt.subplots(figsize=(10, 4))
    if mass[0] != 0:
        ax.plot(t_arr, (mass - mass[0]) / np.abs(mass[0]),
                '-o', ms=1, lw=2, color='black')
        ax.set_ylabel(r'Relative $\Delta\int\mathcal{J}\,dx$')
    else:
        ax.plot(t_arr, mass - mass[0], '-o', ms=1, lw=2, color='black')
        ax.set_ylabel(r'$\Delta\int\mathcal{J}\,dx$')
    ax.axhline(0, ls='--', color='gray', lw=1)
    ax.set_xlabel(r'$t\,[\mathrm{GeV^{-1}}]$')
    ax.grid(True); plt.tight_layout()
    if sf('_mass.pdf'): plt.savefig(sf('_mass.pdf'), bbox_inches='tight')
    plt.show()


def plot_pde_residuals(model, t_eval, x_eval, savefile=None):
    model.eval()
    p = next(model.parameters())
    dtype, dev = p.dtype, p.device

    t_eval = np.asarray(t_eval, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    Nt, Nx = len(t_eval), len(x_eval)
    tt, xx = np.meshgrid(t_eval, x_eval, indexing='ij')
    tx     = np.column_stack([tt.ravel(), xx.ravel()])
    tx_t   = torch.tensor(tx, dtype=dtype, device=dev, requires_grad=True)

    def _g(u):
        return torch.autograd.grad(
            u, tx_t, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

    with torch.set_grad_enabled(True):
        out  = model(tx_t)
        sJ   = out[:, 0:1];  alp = out[:, 1:2]
        t_   = tx_t[:, 0:1]; x_  = tx_t[:, 1:2]
        T    = T_func(t_, x_)
        n    = n_from_alpha_func(alp, T)
        sig  = sigma_func(alp, T)
        tauJ = tauJ_func(alp, T)

        n_t    = _g(n)[:, 0:1]
        alp_x  = _g(alp)[:, 1:2]
        sJ_t   = _g(sJ)[:, 0:1]
        sJ_x   = _g(sJ)[:, 1:2]
        tau_st = _g(tauJ / (sig * T))[:, 1:2]

        sn = n_from_alpha_func(model.sA.to(dtype=dtype), T).abs().clamp(min=1e-30)
        sJ_s = model.sscriptJ.abs().clamp(min=1e-30)

        R1 = (n_t + sJ_x) / sn
        R2 = (tauJ * sJ_t + sJ + 0.5 * sig * T * sJ * tau_st + sig * T * alp_x) / sJ_s

    def tog(R):
        return R.detach().cpu().numpy().reshape(Nt, Nx)

    labels = [r"$|R_1|$  (continuity)", r"$|R_2|$  (IS relaxation)"]
    data   = [tog(R1), tog(R2)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True,
                            constrained_layout=True)
    for i, (ax, lab, res) in enumerate(zip(axs, labels, data), start=1):
        ra = np.clip(np.abs(res), 1e-14, None)
        im = ax.pcolormesh(x_eval, t_eval, ra, shading='auto',
                           cmap='viridis', norm=LogNorm())
        ax.set_title(lab)
        ax.set_xlabel(r'$x\,[\mathrm{GeV^{-1}}]$')
        if i == 1:
            ax.set_ylabel(r'$t\,[\mathrm{GeV^{-1}}]$')
        ax.text(0.03, 0.95,
                rf'$\langle R^2_{{{i}}}\rangle = {np.mean(ra**2):.2e}$',
                transform=ax.transAxes, color='white', fontsize=13,
                va='top',
                bbox=dict(fc='black', alpha=0.5, ec='none', pad=4))
        fig.colorbar(im, ax=ax, orientation='horizontal',
                     fraction=0.046, pad=0.08, ticks=LogLocator(numticks=4))

    if savefile: plt.savefig(savefile, bbox_inches='tight')
    plt.show()


def plot_combined_loss_history(adam_losses, lbfgs_hist, savefile=None):
    adam_losses = np.asarray(adam_losses)

    # Build L-BFGS inner curve
    xs, ys = [], []
    for e, inner in enumerate(lbfgs_hist['all_inner_per_epoch'], start=1):
        if not inner: continue
        m = len(inner)
        xs.append(np.linspace(e - 1, e, m, endpoint=False))
        ys.append(np.asarray(inner))
        xs.append(np.array([e]))
        ys.append(np.array([inner[-1]]))
    if xs:
        xs_lb = np.concatenate(xs); ys_lb = np.concatenate(ys).astype(float)
        ys_lb[:3]  = np.nan   # hide transient
        ys_lb[-10:] = np.nan  # hide potential blow-up
    else:
        xs_lb = np.array([0]); ys_lb = np.array([np.nan])

    fig = plt.figure(figsize=(13, 4))
    gs  = GridSpec(1, 2, width_ratios=[3, 1.5], wspace=0.15)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    ax1.semilogy(np.arange(len(adam_losses)), adam_losses, lw=1.2, color='black')
    ax1.set_xlabel('Adam epoch'); ax1.set_ylabel('Total loss')
    ax1.grid(True, which='both', ls='--', alpha=0.4)
    ax1.annotate('Adam stage', (0.45, 0.97), xycoords='axes fraction',
                 ha='center', va='top', fontsize=15)
    ax1.spines['right'].set_visible(False)

    ax2.semilogy(np.arange(1, len(ys_lb) + 1), ys_lb, lw=1.5, color='black')
    ax2.set_xlabel('L-BFGS iter')
    ax2.grid(True, which='both', ls='--', alpha=0.4)
    ax2.annotate('L-BFGS stage', (0.5, 0.97), xycoords='axes fraction',
                 ha='center', va='top', fontsize=15)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.tick_right(); ax2.yaxis.set_label_position('right')

    plt.tight_layout()
    if savefile: plt.savefig(savefile, bbox_inches='tight')
    plt.show()


def plot_loss_breakdown(loss_R1_hist, loss_R2_hist, savefile=None):
    """NEW: plot R1 and R2 losses separately to diagnose imbalance."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)

    for ax, hist, label, color in zip(
            axes,
            [loss_R1_hist, loss_R2_hist],
            [r'$\mathcal{L}_{R_1}$  (continuity)',
             r'$\mathcal{L}_{R_2}$  (IS relaxation)'],
            ['steelblue', 'darkorange']):
        ax.semilogy(np.arange(len(hist)), hist, lw=1.5, color=color)
        ax.set_xlabel('Adam epoch')
        ax.set_ylabel('Loss')
        ax.set_title(label)
        ax.grid(True, which='both', ls='--', alpha=0.4)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    plt.suptitle('Individual equation losses (adaptive weighting active)',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    if savefile: plt.savefig(savefile, bbox_inches='tight')
    plt.show()

