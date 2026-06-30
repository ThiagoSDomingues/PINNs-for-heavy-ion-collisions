"""
analysis.py
===========
Standalone analysis and plotting script for the PINN benchmark study.

Usage
-----
    python analysis.py [--results_dir RESULTS_DIR] [--out_dir OUT_DIR]

The script expects *.npz result files produced by the training runs,
one per PINN variant per equation, with the naming convention:

    {equation}_{variant}.npz

where
    equation : wave | diffusion | burgers | euler
    variant  : Vanilla | SA | ACTO | SA_ACTO

Each .npz file must contain at least:
    t_eval              : (Nt,)       time grid used during evaluation
    x_eval              : (Nx,)       spatial grid used during evaluation
    u_grid              : (Nt,Nx)     PINN solution (scalar PDEs)
    adam_loss_history   : (N_adam,)   loss per Adam epoch
    lbfgs_inner_curve   : (N_lbfgs,) loss per L-BFGS call
    adam_time           : float       wall time of Adam stage [s]
    lbfgs_time          : float       wall time of L-BFGS stage [s]
    total_training_time : float       total wall time [s]

For Euler (Sod shock tube) the npz should also carry:
    rho_grid / v_grid / p_grid : (Nt, Nx)

Optional (enables richer L2 vs epoch plots):
    L2_history_epoch    : (N,)   L2 values sampled during training
    L2_epochs           : (N,)   corresponding epoch indices
    L2_history_walltime : (N,)   L2 values
    L2_walltimes        : (N,)   corresponding wall-clock times

Produced outputs (all saved under OUT_DIR)
------------------------------------------
For every equation:
    {eq}_snapshots.pdf       -- IC + final-time snapshot overlays (4 PINNs)
    {eq}_loss_curves.pdf     -- loss vs epoch AND vs wall time (all 4 PINNs)
    {eq}_L2_curves.pdf       -- spacetime L2 vs epoch AND vs wall time
    {eq}_L2_heatmap.pdf      -- 4-panel |error| heatmap (log scale)
    {eq}_3d_scatter.pdf      -- 3-D scatter: (log10 L2, log10 Loss, t_E)
    {eq}_time_vs_metrics.pdf -- 2-D scatter: wall time vs L2 / loss

Summary:
    summary_table.txt        -- plain-text table of final L2 and wall times
    summary_table_latex.tex  -- LaTeX tabular for the paper
"""

import os
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set(style="white")
plt.rcParams.update({
    "text.usetex"   : False,
    "font.family"   : "serif",
    "font.size"     : 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi"    : 150,
    "savefig.dpi"   : 300,
    "savefig.bbox"  : "tight",
})

# ── Constants ─────────────────────────────────────────────────────────────────
EQUATIONS = ["wave", "diffusion", "burgers", "euler"]
VARIANTS  = ["Vanilla", "SA", "ACTO", "SA_ACTO"]

LABELS = {
    "Vanilla" : "Vanilla PINN",
    "SA"      : "SA-PINN",
    "ACTO"    : "PINN-ACTO",
    "SA_ACTO" : "SA-PINN-ACTO",
}
COLORS = {
    "Vanilla" : "#4e79a7",
    "SA"      : "#f28e2b",
    "ACTO"    : "#59a14f",
    "SA_ACTO" : "#e15759",
}
LINESTYLES = {
    "Vanilla" : "--",
    "SA"      : "-.",
    "ACTO"    : ":",
    "SA_ACTO" : "-",
}
MARKERS = {
    "Vanilla" : "o",
    "SA"      : "s",
    "ACTO"    : "^",
    "SA_ACTO" : "D",
}
EQ_LABELS = {
    "wave"      : "Wave equation",
    "diffusion" : "Diffusion equation",
    "burgers"   : "Viscous Burgers'",
    "euler"     : "Euler / Sod shock",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_npz(path):
    if not os.path.isfile(path):
        warnings.warn(f"File not found: {path}")
        return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def get_exact_solution(eq, t_eval, x_eval, **kw):
    """Compute the analytical ground-truth solution on the given grid."""
    from scipy.optimize import brentq

    Nt, Nx = len(t_eval), len(x_eval)

    # ── Wave ──────────────────────────────────────────────────────────────
    if eq == "wave":
        L, c = kw.get("L", 1.0), kw.get("c", 1.0)
        tt, xx = np.meshgrid(t_eval, x_eval, indexing="ij")
        u = 0.5 * (
            np.cos(3.*np.pi*c*tt/L) * np.sin(3.*np.pi*xx/L)
            - np.cos(np.pi*c*tt/L) * np.sin(np.pi*xx/L)
        )
        return {"u": u}

    # ── Diffusion ─────────────────────────────────────────────────────────
    elif eq == "diffusion":
        c, L = kw.get("c", 0.1), kw.get("L", 1.0)
        sigma = 0.15 * L
        K     = kw.get("K", 20)
        f     = lambda y: np.exp(-0.5*(y/sigma)**2)
        dy    = x_eval[1] - x_eval[0]
        u     = np.zeros((Nt, Nx))
        for ti, tv in enumerate(t_eval):
            if tv == 0.:
                u[ti] = f(x_eval)
                continue
            ksum = np.zeros(Nx)
            for k in range(-K, K+1):
                shift = 2.*k*L
                diff  = x_eval[:, None] - (x_eval[None, :] + shift)
                G     = np.exp(-diff**2/(4.*c*tv)) / np.sqrt(4.*np.pi*c*tv)
                ksum += G @ f(x_eval) * dy
            u[ti] = ksum
        return {"u": u}

    # ── Burgers (Cole-Hopf) ───────────────────────────────────────────────
    elif eq == "burgers":
        nu, L = kw.get("nu", 0.01/np.pi), kw.get("L", 1.0)
        K     = kw.get("K", 10)
        dx    = x_eval[1] - x_eval[0]
        y     = x_eval.copy()
        psi   = np.exp(-(L/(2.*nu*np.pi))*(np.cos(np.pi*y/L)-1.))
        u     = np.zeros((Nt, Nx))
        for ti, tv in enumerate(t_eval):
            if tv == 0.:
                u[ti] = -np.sin(np.pi*x_eval/L)
                continue
            phi = np.zeros(Nx)
            for k in range(-K, K+1):
                shift = 2.*k*L
                diff  = x_eval[:, None] - (y[None, :]+shift)
                G     = np.exp(-diff**2/(4.*nu*tv)) / np.sqrt(4.*np.pi*nu*tv)
                phi  += np.sum(G*psi[None, :], axis=1)*dx
            u[ti] = -2.*nu*np.gradient(phi, dx)/phi
        return {"u": u}

    # ── Euler / Sod ───────────────────────────────────────────────────────
    elif eq == "euler":
        GAMMA = 1.4
        rL, vL, pL = 1.0, 0.0, 1.0
        rR, vR, pR = 0.125, 0.0, 0.1
        x0 = kw.get("x0", 0.0)

        def a_s(r, p): return np.sqrt(GAMMA*p/r)
        def f_wave(ps, rs, ps0):
            a0 = a_s(rs, ps0)
            if ps <= ps0:
                return (2.*a0/(GAMMA-1.))*((ps/ps0)**((GAMMA-1.)/(2.*GAMMA))-1.)
            A = 2./((GAMMA+1.)*rs)
            B = (GAMMA-1.)/(GAMMA+1.)*ps0
            return (ps-ps0)*np.sqrt(A/(ps+B))

        g      = lambda ps: f_wave(ps,rL,pL)+f_wave(ps,rR,pR)+(vR-vL)
        p_star = brentq(g, 1e-8*min(pL,pR), 10.*max(pL,pR), xtol=1e-12)
        v_star = vL - f_wave(p_star, rL, pL)

        aL        = a_s(rL, pL)
        aR        = a_s(rR, pR)
        a_star_L  = aL*(p_star/pL)**((GAMMA-1.)/(2.*GAMMA))
        rstar_L   = rL*(p_star/pL)**(1./GAMMA)
        s_head    = vL - aL
        s_tail    = v_star - a_star_L
        AR        = 2./((GAMMA+1.)*rR)
        BR        = (GAMMA-1.)/(GAMMA+1.)*pR
        rstar_R   = rR*((p_star/pR+(GAMMA-1.)/(GAMMA+1.))/
                        ((GAMMA-1.)/(GAMMA+1.)*p_star/pR+1.))
        s_shockR  = vR+aR*np.sqrt((GAMMA+1.)/(2.*GAMMA)*p_star/pR
                                   +(GAMMA-1.)/(2.*GAMMA))

        rho_arr = np.zeros((Nt,Nx))
        v_arr   = np.zeros((Nt,Nx))
        p_arr   = np.zeros((Nt,Nx))

        for ti, tv in enumerate(t_eval):
            if tv == 0.:
                rho_arr[ti] = np.where(x_eval<x0, rL, rR)
                v_arr[ti]   = np.where(x_eval<x0, vL, vR)
                p_arr[ti]   = np.where(x_eval<x0, pL, pR)
                continue
            xi = (x_eval-x0)/tv
            r, v, p = np.empty(Nx), np.empty(Nx), np.empty(Nx)

            m1=(xi<=s_head); m2=(xi>s_head)&(xi<=s_tail)
            m3=(xi>s_tail)&(xi<=v_star); m4=(xi>v_star)&(xi<=s_shockR); m5=xi>s_shockR
            r[m1]=rL; v[m1]=vL; p[m1]=pL
            r[m5]=rR; v[m5]=vR; p[m5]=pR
            r[m3]=rstar_L; v[m3]=v_star; p[m3]=p_star
            r[m4]=rstar_R; v[m4]=v_star; p[m4]=p_star

            v_fan  = 2./(GAMMA+1.)*(aL+(GAMMA-1.)/2.*vL+xi[m2])
            a_fan  = 2./(GAMMA+1.)*(aL+(GAMMA-1.)/2.*(vL-xi[m2]))
            r[m2]  = rL*(a_fan/aL)**(2./(GAMMA-1.))
            v[m2]  = v_fan
            p[m2]  = pL*(a_fan/aL)**(2.*GAMMA/(GAMMA-1.))

            rho_arr[ti]=r; v_arr[ti]=v; p_arr[ti]=p

        E_arr = p_arr/(GAMMA-1.) + 0.5*rho_arr*v_arr**2
        return {"rho": rho_arr, "v": v_arr, "p": p_arr, "E": E_arr}

    raise ValueError(f"Unknown equation: {eq}")


def extract_fields(d, eq):
    """Return primary solution fields from an npz dict."""
    if d is None:
        return None
    if eq == "euler":
        out = {}
        for key in ("rho","v","p","E"):
            k2 = key+"_grid"
            if k2 in d:
                out[key] = np.asarray(d[k2], dtype=np.float64)
        if not out and "u_grid" in d:
            out["rho"] = np.asarray(d["u_grid"], dtype=np.float64)
        return out or None
    else:
        if "u_grid" in d:
            return {"u": np.asarray(d["u_grid"], dtype=np.float64)}
        return None


def build_exact(data, eq):
    for var in VARIANTS:
        d = data[eq].get(var)
        if d is not None:
            t = np.asarray(d["t_eval"], dtype=np.float64)
            x = np.asarray(d["x_eval"], dtype=np.float64)
            kw = {}
            for k in ("L","c","nu","x0","K"):
                if k in d:
                    kw[k] = float(d[k])
            return t, x, get_exact_solution(eq, t, x, **kw)
    return None, None, None


def load_all(results_dir):
    data = {}
    for eq in EQUATIONS:
        data[eq] = {}
        for var in VARIANTS:
            path = os.path.join(results_dir, f"{eq}_{var}.npz")
            data[eq][var] = load_npz(path)
    return data


# ── Plot 1 : Snapshot overlay ─────────────────────────────────────────────────

def plot_snapshots(eq, data, exact, t_eval, x_eval, out_dir):
    if t_eval is None:
        return
    display = {"u": "u", "rho": r"$\rho$", "v": "v", "p": "p"}
    for field in [f for f in exact if f != "E"]:
        true_f = exact[field]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, (tidx, tlabel) in zip(axes,
                [(0, "IC  (t=0)"),
                 (-1, f"Final (t={t_eval[-1]:.2f})")]):
            ax.plot(x_eval, true_f[tidx], "k-", lw=2.5, label="Exact", zorder=10)
            for var in VARIANTS:
                d    = data[eq].get(var)
                flds = extract_fields(d, eq)
                if flds is None or field not in flds:
                    continue
                ax.plot(x_eval, flds[field][tidx],
                        color=COLORS[var], ls=LINESTYLES[var], lw=2,
                        label=LABELS[var])
            ax.set_xlabel("x"); ax.set_ylabel(display.get(field, field))
            ax.set_title(tlabel)
            ax.legend(frameon=False, ncol=2, fontsize=10)
            ax.spines[["top","right"]].set_visible(False)
        suffix = f"_{field}" if field != "u" else ""
        fig.suptitle(f"{EQ_LABELS[eq]} — snapshot overlay")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{eq}{suffix}_snapshots.pdf"))
        plt.close(fig)
        print(f"  saved {eq}{suffix}_snapshots.pdf")


# ── Compute metrics ───────────────────────────────────────────────────────────

def compute_metrics(eq, data, exact, t_eval, x_eval):
    if t_eval is None:
        return {}
    metrics = {}
    for var in VARIANTS:
        d = data[eq].get(var)
        if d is None:
            continue
        flds = extract_fields(d, eq)
        if flds is None:
            continue

        num2, den2 = 0., 0.
        for field, pred in flds.items():
            if field == "E":
                continue
            true = exact.get(field)
            if true is None:
                continue
            diff = pred - true
            num2 += np.trapz(np.trapz(diff**2,    x_eval, axis=1), t_eval)
            den2 += np.trapz(np.trapz(true**2,    x_eval, axis=1), t_eval)
        final_L2 = float(np.sqrt(num2/(den2+1e-60)))

        al  = np.asarray(d.get("adam_loss_history",  []), dtype=np.float64)
        ll  = np.asarray(d.get("lbfgs_inner_curve",  []), dtype=np.float64)
        at  = float(d.get("adam_time",  0.))
        lt  = float(d.get("lbfgs_time", 0.))
        tot = float(d.get("total_training_time", at+lt))

        metrics[var] = {
            "final_L2"  : final_L2,
            "wall_time" : tot,
            "adam_loss" : al,
            "lbfgs_loss": ll,
            "adam_time" : at,
            "lbfgs_time": lt,
        }
    return metrics


# ── Plot 2 : Loss curves ──────────────────────────────────────────────────────

def plot_loss_curves(eq, metrics, out_dir):
    if not metrics:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for var, m in metrics.items():
        al, ll = m["adam_loss"], m["lbfgs_loss"]
        na, nl = len(al), len(ll)
        at, lt = m["adam_time"], m["lbfgs_time"]

        # epoch axis
        if na:
            axes[0].semilogy(np.arange(na), al,
                             color=COLORS[var], ls=LINESTYLES[var], lw=1.8,
                             label=LABELS[var])
        if nl:
            axes[0].semilogy(np.arange(na, na+nl),
                             np.where(np.isfinite(ll), ll, np.nan),
                             color=COLORS[var], ls="--", lw=1.4, alpha=0.7)

        # wall-time axis
        if na:
            axes[1].semilogy(np.linspace(0, at, na), al,
                             color=COLORS[var], ls=LINESTYLES[var], lw=1.8,
                             label=LABELS[var])
        if nl:
            axes[1].semilogy(at + np.linspace(0, lt, nl),
                             np.where(np.isfinite(ll), ll, np.nan),
                             color=COLORS[var], ls="--", lw=1.4, alpha=0.7)

    for ax, xl in zip(axes, ["Epoch / Iteration", "Wall time [s]"]):
        ax.set_xlabel(xl); ax.set_ylabel("Loss")
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"{EQ_LABELS[eq]} — training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{eq}_loss_curves.pdf"))
    plt.close(fig)
    print(f"  saved {eq}_loss_curves.pdf")


# ── Plot 3 : L2 vs epoch / wall-time ─────────────────────────────────────────

def plot_L2_curves(eq, data, exact, t_eval, x_eval, metrics, out_dir):
    if not metrics or t_eval is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for var, m in metrics.items():
        d = data[eq].get(var)
        if d is None:
            continue

        L2h_e = d.get("L2_history_epoch", None)
        L2h_t = d.get("L2_history_walltime", None)
        ep    = d.get("L2_epochs", None)
        wt    = d.get("L2_walltimes", None)

        final = m["final_L2"]

        if L2h_e is not None and ep is not None:
            axes[0].semilogy(ep, L2h_e,
                             color=COLORS[var], ls=LINESTYLES[var], lw=1.8,
                             label=LABELS[var])
        else:
            axes[0].axhline(final, color=COLORS[var], ls=LINESTYLES[var],
                            lw=1.5, label=f"{LABELS[var]} (final)")

        if L2h_t is not None and wt is not None:
            axes[1].semilogy(wt, L2h_t,
                             color=COLORS[var], ls=LINESTYLES[var], lw=1.8,
                             label=LABELS[var])
        else:
            axes[1].axhline(final, color=COLORS[var], ls=LINESTYLES[var],
                            lw=1.5, label=f"{LABELS[var]} (final)")

    for ax, xl in zip(axes, ["Epoch / Iteration", "Wall time [s]"]):
        ax.set_xlabel(xl)
        ax.set_ylabel("Spacetime L2 error")
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"{EQ_LABELS[eq]} — spacetime L2 error")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{eq}_L2_curves.pdf"))
    plt.close(fig)
    print(f"  saved {eq}_L2_curves.pdf")


# ── Plot 4 : L2 heatmaps ──────────────────────────────────────────────────────

def plot_L2_heatmaps(eq, data, exact, t_eval, x_eval, out_dir):
    if t_eval is None:
        return
    for field in [f for f in exact if f != "E"]:
        true_f = exact[field]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes_flat = axes.flatten()
        for i, var in enumerate(VARIANTS):
            ax   = axes_flat[i]
            flds = extract_fields(data[eq].get(var), eq)
            if flds is None or field not in flds:
                ax.set_visible(False)
                continue
            err  = np.maximum(np.abs(flds[field] - true_f), 1e-16)
            vmin = max(err.min(), 1e-12)
            vmax = max(err.max(), vmin*10)
            im   = ax.pcolormesh(x_eval, t_eval, err, shading="auto",
                                 norm=LogNorm(vmin=vmin, vmax=vmax),
                                 cmap="viridis")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         ticks=LogLocator(numticks=4))
            ax.set_title(LABELS[var]); ax.set_xlabel("x"); ax.set_ylabel("t")
        suffix = f"_{field}" if field != "u" else ""
        fig.suptitle(f"{EQ_LABELS[eq]} — |PINN - Exact|{suffix} (log scale)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{eq}{suffix}_L2_heatmap.pdf"))
        plt.close(fig)
        print(f"  saved {eq}{suffix}_L2_heatmap.pdf")


# ── Plot 5 : 3-D scatter  (L2, Loss, t_E) ────────────────────────────────────

def plot_3d_scatter(eq, metrics, out_dir):
    if not metrics:
        return
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    for var, m in metrics.items():
        l2 = m["final_L2"]
        al, ll = m["adam_loss"], m["lbfgs_loss"]
        if len(ll):
            valid = ll[np.isfinite(ll)]
            floss = float(valid[-1]) if len(valid) else (float(al[-1]) if len(al) else np.nan)
        elif len(al):
            floss = float(al[-1])
        else:
            floss = np.nan
        ax.scatter([np.log10(max(l2,1e-16))],
                   [np.log10(max(floss,1e-30))],
                   [m["wall_time"]],
                   s=180, color=COLORS[var], marker=MARKERS[var],
                   label=LABELS[var], edgecolors="k", linewidths=0.6)
    ax.set_xlabel("log10(L2)", labelpad=8)
    ax.set_ylabel("log10(Loss)", labelpad=8)
    ax.set_zlabel("Exec. time [s]", labelpad=8)
    ax.set_title(f"{EQ_LABELS[eq]}\n3-D metric scatter")
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{eq}_3d_scatter.pdf"))
    plt.close(fig)
    print(f"  saved {eq}_3d_scatter.pdf")


# ── Plot 6 : time vs metrics  ─────────────────────────────────────────────────

def plot_time_vs_metrics(eq, metrics, out_dir):
    if not metrics:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for var, m in metrics.items():
        t_e = m["wall_time"]
        l2  = m["final_L2"]
        al, ll = m["adam_loss"], m["lbfgs_loss"]
        if len(ll):
            valid = ll[np.isfinite(ll)]
            floss = float(valid[-1]) if len(valid) else (float(al[-1]) if len(al) else np.nan)
        elif len(al):
            floss = float(al[-1])
        else:
            floss = np.nan
        kw = dict(color=COLORS[var], marker=MARKERS[var], s=140,
                  edgecolors="k", linewidths=0.6, zorder=5)
        axes[0].scatter([t_e], [l2],    label=LABELS[var], **kw)
        axes[1].scatter([t_e], [floss], label=LABELS[var], **kw)
    for ax, yl in zip(axes, ["Spacetime L2 error", "Final training loss"]):
        ax.set_yscale("log")
        ax.set_xlabel("Wall time [s]"); ax.set_ylabel(yl)
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"{EQ_LABELS[eq]} — wall time vs accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{eq}_time_vs_metrics.pdf"))
    plt.close(fig)
    print(f"  saved {eq}_time_vs_metrics.pdf")


# ── Summary table ─────────────────────────────────────────────────────────────

def write_summary_table(all_metrics, out_dir):
    # Plain text
    header = "%-22s" % "Equation"
    for v in VARIANTS:
        header += "  %-14s  %-12s" % (f"L2({v})", f"Time({v})[s]")
    sep = "-" * len(header)
    lines = [header, sep]
    for eq in EQUATIONS:
        m   = all_metrics.get(eq, {})
        row = "%-22s" % EQ_LABELS[eq]
        for v in VARIANTS:
            if v in m:
                row += "  %-14.3e  %-12.1f" % (m[v]["final_L2"], m[v]["wall_time"])
            else:
                row += "  %-14s  %-12s" % ("N/A", "N/A")
        lines.append(row)
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary_table.txt"), "w") as f:
        f.write(txt)
    print(f"\n=== Summary ===\n{txt}")

    # LaTeX
    latex = [
        r"\begin{tabular}{l" + "cc"*len(VARIANTS) + r"}",
        r"\toprule",
        "Equation & " + " & ".join(
            r"\multicolumn{2}{c}{" + LABELS[v] + r"}" for v in VARIANTS
        ) + r" \\",
        " & " + " & ".join([r"$E_{\rm rel}$ & Time~[s]"]*len(VARIANTS)) + r" \\",
        r"\midrule",
    ]
    for eq in EQUATIONS:
        m   = all_metrics.get(eq, {})
        row = EQ_LABELS[eq]
        for v in VARIANTS:
            if v in m:
                row += r" & %.2e & %.0f" % (m[v]["final_L2"], m[v]["wall_time"])
            else:
                row += r" & N/A & N/A"
        latex.append(row + r" \\")
    latex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out_dir, "summary_table_latex.tex"), "w") as f:
        f.write("\n".join(latex) + "\n")
    print(f"LaTeX table saved to {out_dir}/summary_table_latex.tex")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PINN benchmark analysis — all plots and metrics.")
    parser.add_argument("--results_dir", default="results",
                        help="Dir with {eq}_{variant}.npz files.")
    parser.add_argument("--out_dir", default="analysis_output",
                        help="Dir to save plots and tables.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading results from: {args.results_dir}")
    data = load_all(args.results_dir)

    all_metrics = {}
    for eq in EQUATIONS:
        print(f"\n{'='*58}")
        print(f"  {EQ_LABELS[eq]}")
        print(f"{'='*58}")

        t_eval, x_eval, exact = build_exact(data, eq)
        if exact is None:
            print("  No data found — skipping.")
            continue

        metrics = compute_metrics(eq, data, exact, t_eval, x_eval)
        all_metrics[eq] = metrics
        for var, m in metrics.items():
            print(f"  [{var:8s}]  L2={m['final_L2']:.3e}  wall={m['wall_time']:.1f}s")

        plot_snapshots      (eq, data, exact, t_eval, x_eval,          args.out_dir)
        plot_loss_curves    (eq, metrics,                               args.out_dir)
        plot_L2_curves      (eq, data, exact, t_eval, x_eval, metrics, args.out_dir)
        plot_L2_heatmaps    (eq, data, exact, t_eval, x_eval,          args.out_dir)
        plot_3d_scatter     (eq, metrics,                               args.out_dir)
        plot_time_vs_metrics(eq, metrics,                               args.out_dir)

    write_summary_table(all_metrics, args.out_dir)
    print(f"\nAll outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
