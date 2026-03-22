"""
Script to generate and plot a Gaussian initial condition for IS PINNs outputs: alpha and scriptJ
"""

import numpy as np, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ======================
# Plotting settings
# ======================

plt.rcParams.update({
    "font.family"      : "serif",
    "mathtext.fontset" : "cm",
    "font.size"        : 12,
    "axes.linewidth"   : 0.8,
    "xtick.direction"  : "in", "ytick.direction": "in",
    "xtick.major.size" : 4,    "ytick.major.size": 4,
})

T_END = 20.0
snap_t  = np.array([0.0, T_END/3, 2*T_END/3, T_END])
grey    = ["0.00", "0.30", "0.55", "0.72"]
snap_lb = [f"$t = {v:.2f}$" + r"$\;[\mathrm{GeV}^{-1}]$" for v in snap_t]


# Exact IC parameters from the paper
L     = 50.0    # GeV^-1  (half-domain)
T     = 0.3     # GeV
Nc, Nf, PI = 3, 3, math.pi

x = np.linspace(-L, L, 2000)

# ICs
n_ic  = 0.2  * np.exp(-(7*x/L)**2)  + 1.00
J0_ic = 0.05 * np.exp(-(10*x/L)**2) + 1.05

# Standard deviations
sig_n  = L / (7  * math.sqrt(2))
sig_J0 = L / (10 * math.sqrt(2))

# alpha from n
def alpha(n):
    return 27*n / (Nc*Nf*T**3)

alpha_ic = alpha(n_ic)

print(f"n IC:  bg={1.00}, amp={0.20}, peak={n_ic.max():.3f},  σ={sig_n:.3f} GeV^-1")
print(f"J0 IC: bg={1.05}, amp={0.05}, peak={J0_ic.max():.4f}, σ={sig_J0:.3f} GeV^-1")
print(f"α IC:  min={alpha_ic.min():.3f}, max={alpha_ic.max():.3f}")
print(f"S_N={1.0}, S_alpha={alpha(1.0):.4f}")
print(f"Snapshot times: 0, {20/3:.4f}, {2*20/3:.4f}, 20.0 GeV^-1")

#Output

#n IC:  bg=1.0, amp=0.2, peak=1.200,  σ=5.051 GeV^-1
#J0 IC: bg=1.05, amp=0.05, peak=1.1000, σ=3.536 GeV^-1
#α IC:  min=111.111, max=133.333
#S_N=1.0, S_alpha=111.1111 scaling N and alpha
#Snapshot times: 0, 6.6667, 13.3333, 20.0 GeV^-1

def make_field_figure(x, field_ic, field_label, ylabel,
                      ylim, yticks, cmap, outpath):
    """
    Single-field two-panel figure (matching paper style):
      Top:    dashed lines — only IC shown (black), 
              grey dashed lines for other snapshot times (empty/flat at bg)
      Bottom: heatmap — IC profile tiled along t; 
              placeholder text explains PINN will fill this.
    """
    fig = plt.figure(figsize=(7.2, 5.8))
    ax_top = fig.add_axes([0.13, 0.57, 0.60, 0.38])
    ax_bot = fig.add_axes([0.13, 0.10, 0.60, 0.40])
    cax    = fig.add_axes([0.75, 0.10, 0.025, 0.40])

    # ── top: IC snapshot + ghost lines for future t ───────────────────────
    ax_top.plot(x, field_ic, color=grey[0], lw=2.2, ls="--")

    leg = [plt.Line2D([0],[0], ls="--", lw=2.0, color=grey[0], label=snap_lb[0])]
    for k in range(1, len(snap_t)):
        # show the background level at other times as faint dashes
        bg = field_ic.min()   # background value
        ax_top.plot(x, np.full_like(x, bg + (field_ic.max()-field_ic.min())*0.0),
                    color=grey[k], lw=2.0, ls="--", alpha=0.0)  # invisible ghost
        leg.append(plt.Line2D([0],[0], ls="--", lw=2.0,
                               color=grey[k], label=snap_lb[k]))

    ax_top.set_xlim(-L, L)
    ax_top.set_ylim(*ylim)
    ax_top.set_yticks(yticks)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylabel(ylabel, fontsize=12.5)
    ax_top.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    ax_top.legend(handles=leg,
                  loc="upper left", bbox_to_anchor=(1.03, 1.0),
                  frameon=False, fontsize=10.5, handlelength=2.2)

    # ── bottom: heatmap of IC ────────────────────────────────────────────
    vmin = float(field_ic.min())
    vmax = float(field_ic.max())
    field_2d = np.tile(field_ic.reshape(-1,1), (1, 200))

    im = ax_bot.imshow(
        field_2d.T,
        origin="lower", aspect="auto",
        extent=[-L, L, 0, T_END],
        cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="bilinear",
    )
    ax_bot.set_xlabel(r"$x\;[\mathrm{GeV}^{-1}]$", fontsize=12.5)
    ax_bot.set_ylabel(r"$t\;[\mathrm{GeV}^{-1}]$",  fontsize=12.5)
    ax_bot.set_xlim(-L, L)
    ax_bot.set_ylim(0, T_END)
    ax_bot.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax_bot.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    cb = plt.colorbar(im, cax=cax)
    cb.set_label(ylabel, fontsize=11, labelpad=5)
    cb.ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {outpath}")
