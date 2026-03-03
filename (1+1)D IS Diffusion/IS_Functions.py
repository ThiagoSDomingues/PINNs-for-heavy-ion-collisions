# =============================================================================
# IS_Functions.py
#
# Israel–Stewart (IS) diffusion in (1+1)D — physics functions for the PINN.
#
# This file is the IS counterpart of BDNK_Functions.py from the SA-PINN-ACTO
# repository (vchomalicastro/1-1D-BDNK-diffusion-simulations).
#
# ── Structural correspondence ──────────────────────────────────────────────
#
#   BDNK variable          IS counterpart          Role
#   ──────────────────────────────────────────────────────────────────────
#   J^0                    J^0                     conserved charge density
#   α   (= μ/T)            α   (= μ/T)             thermal chemical potential
#   N^x (= -∂_x α)         ν^x                     diffusion current (x-comp.)
#   N_0 (algebraic)        ν^0 = 0  (transverse)   [no independent ν^0 in LRF]
#   λ   (frame parameter)  τ_ν  (relaxation time)  causal regularisation
#   c_ch = √(σ/λ)          c_IS = √(σ T / (ρ τ_ν)) causal signal speed
#
# ── IS equations of motion (1+1)D, general Lorentz frame ──────────────────
#
#   ∂_t J^0  + ∂_x J^x  = 0                              (charge conservation)
#
#   Δ^{μν} [ τ_ν u^λ ∂_λ ν_ν  +  ν_μ  −  σ T Δ_μ^{\ ν} ∂_ν α ] = 0
#                                                          (IS relaxation eq.)
#
#   where the constitutive relation is
#       J^μ = n u^μ + ν^μ,    u_μ ν^μ = 0
#
#   In the local rest frame (v=0, u^μ=(1,0)) this collapses to:
#       ∂_t J^0 + ∂_x ν^x = 0
#       τ_ν ∂_t ν^x  +  ν^x  =  σ T ∂_x α
#
#   In a boosted frame (general v) the spatial projection of the relaxation
#   equation gives the equation for ν^x used in the PINN loss.
#
# ── EOS & transport ───────────────────────────────────────────────────────
#   Massless QCD gas, N_c = N_f = 3 (identical to BDNK setup).
#   Conductivity σ(α,T): same expression as BDNK_Functions.py.
#   Relaxation time:  τ_ν = σ T / (c_IS^2 * ρ_IS),
#       where ρ_IS = ∂n/∂α * T  is the enthalpy-like susceptibility and
#       c_IS ∈ (0,1) is a free causal-speed parameter (analogous to c_ch).
#   This choice ensures the IS causal front speed equals c_IS.
#
# =============================================================================

import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import numpy as np

# ── QCD parameters (same as BDNK) ─────────────────────────────────────────
N_c = 3
N_f = 3
# C_B = 1 / (4 * np.pi)   # Third setup (small-α / KSS limit)
C_B = 0.4                  # First and second setups

# ── IS causal-speed parameter (analogous to c_ch in BDNK) ─────────────────
# c_IS ∈ (0, 1).  The IS relaxation time is constructed so that the
# characteristic speed of the IS system equals c_IS.
c_IS = 0.5

# =============================================================================
# Background field interpolation  (identical to BDNK_Functions.py)
# T(t,x) and v(t,x) are background fields treated as probes.
# =============================================================================

_T_tab  = None
_v_tab  = None
_t0     = None
_dt     = None
_Nt     = None
_x0     = None
_dx     = None
_Nx     = None
_Lval   = None
_T_tmax = 20.0        # default tmax in GeV^{-1}

_EPS_TO_T_DENOM = 15.6268736   # converts energy density to T^4: ε = _EPS_TO_T_DENOM * T^4


def _BDNK_base_dir():
    """Return the absolute path to the 'BDNK Background Simulations' folder
    sitting next to this file.  Kept identical to BDNK_Functions.py so that
    IS and BDNK share the same background data."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "BDNK Background Simulations")


def _pick_BDNK_subfolder(base_dir: str, nsim: int) -> str:
    prefix = f"{nsim}_"
    candidates = [d for d in os.listdir(base_dir)
                  if d.startswith(prefix) and os.path.isdir(os.path.join(base_dir, d))]
    if not candidates:
        raise FileNotFoundError(f"No folder starting with '{prefix}' in {base_dir}")
    candidates.sort()
    return os.path.join(base_dir, candidates[0])


def setup_external_Tv(BDNK_simulation: int, L_in: float, tmax: float = _T_tmax):
    """Load background (ε, v) tables from a BDNK simulation folder and set up
    the bicubic interpolation globals.  Identical interface to BDNK_Functions.py."""
    global _T_tab, _v_tab, _t0, _dt, _Nt, _x0, _dx, _Nx, _Lval, _T_tmax

    base   = _BDNK_base_dir()
    folder = _pick_BDNK_subfolder(base, BDNK_simulation)

    ep_path = os.path.join(folder, "ep(t,x).npy")
    v_path  = os.path.join(folder, "v(t,x).npy")
    if not os.path.exists(ep_path):
        raise FileNotFoundError(ep_path)
    if not os.path.exists(v_path):
        raise FileNotFoundError(v_path)

    ep_tx = np.load(ep_path)
    v_tx  = np.load(v_path)
    if ep_tx.shape != v_tx.shape:
        raise ValueError(f"Shape mismatch: ep {ep_tx.shape} vs v {v_tx.shape}")

    Nt_data, Nx_data = ep_tx.shape
    t_axis = np.linspace(0.0, tmax, Nt_data, dtype=np.float64)

    # Drop the periodic-duplicate final column if present
    if (np.allclose(ep_tx[:, 0], ep_tx[:, -1]) and
            np.allclose(v_tx[:, 0], v_tx[:, -1])):
        ep_tx = ep_tx[:, :-1]
        v_tx  = v_tx[:, :-1]
    Nx_data = ep_tx.shape[1]

    dx_data = 2.0 * L_in / Nx_data
    x0_data = -L_in + 0.5 * dx_data

    T_tx = (ep_tx.astype(np.float64) / _EPS_TO_T_DENOM) ** 0.25

    _T_tab = torch.tensor(T_tx,  dtype=torch.float32, device=device)
    _v_tab = torch.tensor(v_tx,  dtype=torch.float32, device=device)

    _t0  = torch.tensor(float(t_axis[0]),  dtype=torch.float32, device=device)
    _dt  = torch.tensor(
        float((t_axis[-1] - t_axis[0]) / (Nt_data - 1) if Nt_data > 1 else 1.0),
        dtype=torch.float32, device=device)
    _Nt  = torch.tensor(int(Nt_data),  dtype=torch.int64, device=device)

    _x0  = torch.tensor(float(x0_data), dtype=torch.float32, device=device)
    _dx  = torch.tensor(float(dx_data), dtype=torch.float32, device=device)
    _Nx  = torch.tensor(int(Nx_data),   dtype=torch.int64, device=device)

    _Lval   = torch.tensor(float(L_in), dtype=torch.float32, device=device)
    _T_tmax = torch.tensor(float(tmax), dtype=torch.float32, device=device)


# ── Bicubic interpolation helpers (unchanged from BDNK_Functions.py) ───────

def _cubic_kernel_1d(u, a=-0.5):
    absu  = u.abs()
    absu2 = absu * absu
    absu3 = absu2 * absu
    w = torch.where(
        absu <= 1,
        (a + 2) * absu3 - (a + 3) * absu2 + 1,
        torch.where(
            (absu > 1) & (absu < 2),
            a * absu3 - 5 * a * absu2 + 8 * a * absu - 4 * a,
            torch.zeros_like(u)
        )
    )
    return w


def _wrap_x_periodic(x):
    twoL = 2.0 * _Lval
    return torch.remainder(x + _Lval, twoL) - _Lval


def _bicubic_sample_tx(t, x, tab_2d):
    """Bicubic interpolation of a 2-D table tab_2d[t_index, x_index]."""
    t = t.to(dtype=torch.float32, device=device)
    x = x.to(dtype=torch.float32, device=device)
    tb, xb = torch.broadcast_tensors(t, x)
    t_flat = tb.reshape(-1)
    x_flat = xb.reshape(-1)

    tmin = _t0
    tmax = _t0 + _dt * (_Nt.to(torch.float32) - 1.0)
    tt = torch.clamp(t_flat, min=tmin, max=tmax)
    xx = _wrap_x_periodic(x_flat)

    ft = (tt - _t0) / (_dt + 1e-12)
    fx = (xx - _x0) / (_dx + 1e-12)

    it0 = torch.floor(ft).to(torch.int64)
    ix0 = torch.floor(fx).to(torch.int64)

    dt_frac = ft - it0.to(torch.float32)
    dx_frac = fx - ix0.to(torch.float32)

    NtI = int(_Nt.item())
    NxI = int(_Nx.item())
    offsets = torch.tensor([-1, 0, 1, 2], device=device, dtype=torch.int64)

    it_neighbors = (it0.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, NtI - 1)
    ix_base      = torch.remainder(ix0, NxI).to(torch.int64)
    ix_neighbors = torch.remainder(ix_base.unsqueeze(1) + offsets.unsqueeze(0), NxI)

    wt = torch.stack([_cubic_kernel_1d(dt_frac - k) for k in (-1., 0., 1., 2.)], dim=1)
    wx = torch.stack([_cubic_kernel_1d(dx_frac - k) for k in (-1., 0., 1., 2.)], dim=1)

    tab = tab_2d.reshape(NtI * NxI)
    vals = []
    for j in range(4):
        idx_row_4 = (it_neighbors[:, j].unsqueeze(1) * NxI + ix_neighbors)
        vals.append(tab[idx_row_4])
    vals = torch.stack(vals, dim=1)

    vx  = (vals * wx.unsqueeze(1)).sum(dim=2)
    out = (vx * wt).sum(dim=1)
    return out.view_as(tb)


# =============================================================================
# Background field accessors
# (identical interface to BDNK_Functions.py — swap the commented lines for
#  BDNK-background runs, exactly as in the original)
# =============================================================================

def T_func(t, x):
    """Background temperature T(t, x)."""
    return 0.3 * torch.ones_like(x)           # Setups 1 & 2: constant T
    # return _bicubic_sample_tx(t, x, _T_tab)  # Setup 3: BDNK background


def v_func(t, x):
    """Background fluid velocity v(t, x)."""
    return 0.0 * torch.ones_like(x)           # Setups 1 & 2: rest frame
    # return _bicubic_sample_tx(t, x, _v_tab)  # Setup 3: BDNK background


def gamma_func(v):
    """Lorentz factor γ = 1/√(1−v²)."""
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.float64, device=device)
    return 1.0 / torch.sqrt(1 - v ** 2)


# =============================================================================
# Equation of State  (massless QCD gas, N_c = N_f = 3)
# Identical to BDNK_Functions.py — shared EOS.
# =============================================================================

def alpha_from_n_func(n, T):
    """Invert n(α,T) analytically to obtain α = μ/T.
    Uses the cubic root formula for the massless QCD EOS."""
    a = T ** 3 * N_c * N_f
    b = torch.sqrt(
        2187 * n ** 2 * T ** 12 * N_c ** 4 * N_f ** 4
        + 4 * torch.pi ** 2 * a ** 6
    )
    c = 81 * n * a ** 2
    d = torch.pow(torch.sqrt(torch.tensor(3.0, dtype=n.dtype, device=n.device)) * b + c,
                  1 / 3)
    term1 = (3 / 2) ** (1 / 3) * torch.pi ** (2 / 3) * d / a
    term2 = 2 ** (1 / 3) * 3 ** (2 / 3) * torch.pi ** (4 / 3) * a / d
    return term1 - term2


def n_from_alpha_func(alpha, T):
    """Baryon number density  n(α,T) for the massless QCD gas."""
    term1 = alpha / 27
    term2 = (alpha ** 3) / (243 * torch.pi ** 2)
    return N_c * N_f * T ** 3 * (term1 + term2)


def dn_dalpha_func(alpha, T):
    """
    ∂n/∂α at fixed T — the charge susceptibility.

    dn/dα = N_c N_f T³ [ 1/27 + α²/(81π²) ]

    This appears in the IS relaxation time and also in the
    'enthalpy-like susceptibility' ρ_IS = T · dn/dα used to fix τ_ν.
    There is no direct counterpart in BDNK_Functions.py because BDNK
    uses the frame parameter λ instead of a relaxation time.
    """
    return N_c * N_f * T ** 3 * (1.0 / 27.0 + alpha ** 2 / (81.0 * torch.pi ** 2))


def mu_func(alpha, T):
    """Chemical potential  μ = α T."""
    return alpha * T


def pressure_func(alpha, T):
    """Pressure P(α, T) for the massless QCD gas."""
    pi = torch.pi
    mu = mu_func(alpha, T)
    term1 = (2 * (N_c ** 2 - 1) + 3.5 * N_c * N_f) * pi ** 2 * T ** 4 / 90
    term2 = N_c * N_f * mu ** 2 * T ** 2 / 54
    term3 = N_c * N_f * mu ** 4 / (972 * pi ** 2)
    return term1 + term2 + term3


def sigma_func(alpha, T):
    """
    Baryon conductivity  σ(α, T).

    σ = C_B n [coth(α)/3 − n T/(ε+P)] / T²

    Identical expression to BDNK_Functions.py — the conductivity is a
    property of the fluid, not of the hydrodynamic framework.
    """
    n = n_from_alpha_func(alpha, T)
    P = pressure_func(alpha, T)
    eps = 3 * P   # conformal EOS: ε = 3P
    return C_B * n * (1.0 / 3.0 * 1.0 / torch.tanh(alpha) - n * T / (eps + P)) / (T ** 2)


# =============================================================================
# IS-specific transport coefficient: relaxation time τ_ν
#
# In BDNK the frame is fixed by λ = σ / c_ch².
# In IS the causal regularisation is achieved by the relaxation time:
#
#   τ_ν(α, T) = σ(α, T) · T  /  [ c_IS² · ρ_IS(α, T) ]
#
# where  ρ_IS = T · dn/dα  is the (enthalpy-normalised) charge susceptibility.
#
# This relation is derived by demanding that the linearised IS dispersion
# relation gives a causal front speed equal to c_IS (analogous to c_ch in
# BDNK).  For small α one recovers  τ_ν = σ/(c_IS² χ_q)  with χ_q = dn/dμ.
# =============================================================================

def tau_nu_func(alpha, T):
    """
    IS relaxation time  τ_ν(α, T).

    τ_ν = σ T / (c_IS² · ρ_IS),    ρ_IS = T · dn/dα

    In the LRF at small α this reduces to  τ_ν ≈ σ / (c_IS² χ_q)
    with χ_q = dn/dμ = (dn/dα)/T the charge susceptibility.

    BDNK analogue: lambd_func  (λ = σ / c_ch²), but note that λ enters
    the BDNK current algebraically while τ_ν enters the IS equations
    as a dynamical time scale.
    """
    sig    = sigma_func(alpha, T)
    rho_IS = T * dn_dalpha_func(alpha, T)   # T · dn/dα
    return sig * T / (c_IS ** 2 * rho_IS + 1e-30)


# =============================================================================
# Flux-conservative IS equations in a general Lorentz frame
#
# The IS diffusion system in (1+1)D with background fields T(t,x), v(t,x):
#
#   ∂_t J^0  +  ∂_x J^x  =  0                               (conservation)
#
#   τ_ν Π^{μν} u^λ ∂_λ ν_ν  +  ν^μ  =  σ T Δ^{μν} ∂_ν α   (IS relaxation)
#
# where Π^{μν} = Δ^{μν} = g^{μν} + u^μ u^ν is the spatial projector.
#
# The constitutive relation is:
#   J^μ = n u^μ + ν^μ
#
# In the boosted frame u^μ = γ(1, v):
#   J^0 = γ n + γ²(γ²v²−1) temporal part of ν contracted with u...
#
# For the PINN we take ν^x as an independent network output, just as
# N^x = −∂_x α is an independent output in the BDNK PINN.
# The x-component of the IS relaxation equation is the PDE residual.
#
# ── General frame equations ────────────────────────────────────────────────
#
#  J^x  =  γ n v  +  ν^x                           (trivially from J^μ=nu^μ+ν^μ)
#
#  IS relaxation (x-projection, general v):
#
#    τ_ν γ [ ∂_t ν^x + v ∂_x ν^x ]                  ← u^λ ∂_λ ν^x
#    + ν^x
#    − γ²(v ∂_t α + ∂_x α ... ) correction terms
#    = σ T [ −∂_x α + γ²(∂_t α + v ∂_x α)(−v) ]    ← σ T Δ^{xν} ∂_ν α
#
# For clarity the functions below separate the LRF case (v=0) and the
# general-frame case.
# =============================================================================

def Jx_IS_func(n, nu_x, v):
    """
    x-component of the IS conserved current.

        J^x = γ n v + ν^x

    BDNK analogue: Jx_func — but there J^x contains the BDNK diffusion
    current built from N^x and N_0, whereas here ν^x is independent.

    Parameters
    ----------
    n    : baryon density n(α, T)
    nu_x : x-component of the IS diffusion current (independent PINN output)
    v    : background fluid velocity
    """
    gamma = gamma_func(v)
    return gamma * n * v + nu_x


def J0_IS_func(n, nu_x, v):
    """
    Time-component of the IS conserved current.

        J^0 = γ n + γ² v ν^x          [from J^μ = n u^μ + ν^μ, u^0 = γ]

    Note: the transversality condition u_μ ν^μ = 0 gives
          ν^0 = v ν^x  (in 1+1D), so J^0 = γ n + γ v ν^x is NOT the
          network output; it is determined algebraically from n and ν^x.
          This function is used for cross-checks and initialisation.

    BDNK analogue: J0_func — in BDNK J^0 is the primary output of the
    network and α is the secondary; in IS J^0 and α are both primary
    outputs together with ν^x.
    """
    gamma = gamma_func(v)
    nu_0  = v * nu_x          # transversality: ν^0 = v ν^x in the boosted 1D frame
    return gamma * n + gamma * nu_0


def IS_relaxation_residual_func(nu_x, nu_x_t, nu_x_x, alpha, alpha_t, alpha_x,
                                 sigma, tau_nu, T, v):
    """
    Residual of the IS relaxation equation projected onto the x-direction.

        R_IS = τ_ν γ (∂_t ν^x + v ∂_x ν^x)
               + ν^x
               + γ² v τ_ν (v ∂_t ν^x + ∂_x ν^x)   ← second-rank correction
               − σ T [ Δ^{x0} ∂_t α + Δ^{xx} ∂_x α ]

    where the spatial projector components are:
        Δ^{x0} = γ² v
        Δ^{xx} = 1 + γ² v²

    And u^λ ∂_λ ν^x = γ (∂_t ν^x + v ∂_x ν^x).

    This simplifies in the LRF (v=0) to:
        R_IS = τ_ν ∂_t ν^x  +  ν^x  −  σ T ∂_x α

    Parameters
    ----------
    nu_x    : ν^x (PINN output)
    nu_x_t  : ∂_t ν^x
    nu_x_x  : ∂_x ν^x
    alpha   : α (PINN output)
    alpha_t : ∂_t α
    alpha_x : ∂_x α
    sigma   : conductivity σ(α, T)
    tau_nu  : relaxation time τ_ν(α, T)
    T       : background temperature
    v       : background velocity

    Returns
    -------
    Scalar residual (should be zero on the true solution).

    BDNK analogue: there is no single BDNK function for this; in BDNK the
    conservation equation ∂_t J^0 + ∂_x J^x = 0 is the only PDE residual
    because N^x ≡ −∂_x α is not independent.  In IS we have TWO PDE
    residuals: the conservation law and this relaxation equation.
    """
    gamma = gamma_func(v)

    # u^λ ∂_λ ν^x  =  γ (∂_t ν^x + v ∂_x ν^x)
    u_dot_grad_nu_x = gamma * (nu_x_t + v * nu_x_x)

    # σ T Δ^{xν} ∂_ν α  =  σ T [ γ² v ∂_t α + (1 + γ²v²) ∂_x α ]
    # (Δ^{x0} = γ²v, Δ^{xx} = 1 + γ²v²)
    sigma_T_drive = sigma * T * (gamma ** 2 * v * alpha_t
                                 + (1 + gamma ** 2 * v ** 2) * alpha_x)

    # Full IS relaxation residual
    return tau_nu * u_dot_grad_nu_x + nu_x - sigma_T_drive


def conservation_residual_func(J0_t, nu_x_x, n, nu_x, v):
    """
    Residual of the charge conservation equation:

        R_cons = ∂_t J^0 + ∂_x J^x

    where J^0 and J^x are the IS currents.

    In the PINN we take J^0 and α as primary outputs (like in BDNK), plus ν^x.
    We therefore compute ∂_t J^0 directly from the network Jacobian, and
    ∂_x J^x = ∂_x (γ n v + ν^x) = γ v (dn/dα) ∂_x α + ∂_x ν^x
              (since γ and v are background, not trained).

    In the LRF (v=0):
        R_cons = ∂_t J^0 + ∂_x ν^x = 0

    Parameters
    ----------
    J0_t   : ∂_t J^0 (from autograd on the J^0 network output)
    nu_x_x : ∂_x ν^x (from autograd on the ν^x network output)
    n      : n(α, T)  [used for consistency; J^x = γnv + ν^x]
    nu_x   : ν^x
    v      : background velocity

    Returns
    -------
    Scalar residual.
    """
    gamma = gamma_func(v)
    # ∂_x J^x = γv (∂_x n) + ∂_x ν^x
    # We pass pre-computed ∂_t J^0 directly from autograd, and ∂_x ν^x likewise.
    # The ∂_x(γnv) term should be handled externally via autograd on J^0 or
    # by passing ∂_x n explicitly; here we keep the structure parallel to
    # BDNK where J^x is differentiated via autograd on the full expression.
    return J0_t + nu_x_x   # simplest form; full version uses d_x(gamma*n*v) separately


# =============================================================================
# Convenience wrapper: compute all IS fluxes given PINN outputs
#
# The PINN outputs three fields:  (J^0,  α,  ν^x)
# (Same number as BDNK: (J^0, α, N^x), but now ν^x is dynamical.)
# =============================================================================

def IS_fluxes(J0, alpha, nu_x, T, v):
    """
    Compute all derived quantities needed for the IS loss function.

        n     = n(α, T)
        J^x   = γ n v + ν^x
        σ     = σ(α, T)
        τ_ν   = τ_ν(α, T)

    Parameters
    ----------
    J0    : PINN output — conserved charge density (shape: [N,1])
    alpha : PINN output — thermal chemical potential α = μ/T (shape: [N,1])
    nu_x  : PINN output — IS diffusion current x-component (shape: [N,1])
    T     : background temperature (shape: [N,1])
    v     : background velocity (shape: [N,1])

    Returns
    -------
    dict with keys: n, Jx, sigma, tau_nu, gamma
    """
    n      = n_from_alpha_func(alpha, T)
    sigma  = sigma_func(alpha, T)
    tau_nu = tau_nu_func(alpha, T)
    gamma  = gamma_func(v)
    Jx     = gamma * n * v + nu_x   # J^x = γnv + ν^x
    return dict(n=n, Jx=Jx, sigma=sigma, tau_nu=tau_nu, gamma=gamma)


# =============================================================================
# Gradient extraction helper
#
# BDNK uses N_x_func to compute N^x = -∂_x α via autograd.
# For IS we need several gradients explicitly; we provide a unified helper.
# =============================================================================

def grad_func(field, coord, name=""):
    """
    Compute ∂(field)/∂(coord) via autograd.

    Parameters
    ----------
    field : tensor requiring grad computation
    coord : tensor w.r.t. which we differentiate; must have requires_grad=True
    name  : optional label for debugging

    Returns
    -------
    Tensor of the same shape as field.

    BDNK analogue: N_x_func computes -∂_x α specifically; this function
    generalises it to any (field, coord) pair needed by the IS system.
    """
    g = torch.autograd.grad(
        field, coord,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    if g is None:
        raise RuntimeError(f"grad_func: gradient of {name} w.r.t. coord is None — "
                           "check that coord has requires_grad=True.")
    return g


def extract_spacetime_grads(alpha, nu_x, J0, tx):
    """
    Extract all first-order spacetime derivatives needed for the IS PDE residuals.

    The PINN input tensor tx has shape [N, 2] with tx[:,0] = t, tx[:,1] = x.
    This helper mirrors the in-line autograd calls in BDNK_Functions.py
    (see J0_func and N_x_func there).

    Parameters
    ----------
    alpha : α output of the PINN, shape [N,1]
    nu_x  : ν^x output of the PINN, shape [N,1]
    J0    : J^0 output of the PINN, shape [N,1]
    tx    : input tensor with requires_grad=True, shape [N,2]

    Returns
    -------
    dict with keys:
        alpha_t, alpha_x   — ∂_t α, ∂_x α
        nu_x_t, nu_x_x    — ∂_t ν^x, ∂_x ν^x
        J0_t               — ∂_t J^0  (used in conservation residual)

    Note: ∂_x J^x = ∂_x(γnv + ν^x).  Since γ, n, v all depend on α and x,
    the full derivative is best computed by passing J^x through autograd.
    Here we return ∂_x ν^x as the primary building block; the caller should
    add γ v dn/dα · ∂_x α if the background velocity is non-zero.
    """
    # ∂α/∂t and ∂α/∂x
    alpha_grad = torch.autograd.grad(
        alpha, tx,
        grad_outputs=torch.ones_like(alpha),
        create_graph=True, retain_graph=True, allow_unused=True
    )[0]                                  # shape [N, 2]
    alpha_t = alpha_grad[:, 0:1]
    alpha_x = alpha_grad[:, 1:2]

    # ∂ν^x/∂t and ∂ν^x/∂x
    nu_x_grad = torch.autograd.grad(
        nu_x, tx,
        grad_outputs=torch.ones_like(nu_x),
        create_graph=True, retain_graph=True, allow_unused=True
    )[0]
    nu_x_t = nu_x_grad[:, 0:1]
    nu_x_x = nu_x_grad[:, 1:2]

    # ∂J^0/∂t  (used in conservation residual)
    J0_grad = torch.autograd.grad(
        J0, tx,
        grad_outputs=torch.ones_like(J0),
        create_graph=True, retain_graph=True, allow_unused=True
    )[0]
    J0_t = J0_grad[:, 0:1]

    return dict(
        alpha_t=alpha_t, alpha_x=alpha_x,
        nu_x_t=nu_x_t,   nu_x_x=nu_x_x,
        J0_t=J0_t,
    )


# =============================================================================
# Full IS PDE residual function
#
# This is the central function called by the PINN training loop.
# It mirrors the role of the BDNK equations assembled in the notebook
# BDNKProblem - SA-PINN-ACTO.ipynb, where J^0 conservation and
# ∂_t α = -N_0 are the two PDE residuals.
#
# For IS we have TWO PDE residuals:
#   R1: charge conservation          ∂_t J^0 + ∂_x J^x = 0
#   R2: IS relaxation (x-component)  τ_ν u^λ∂_λ ν^x + ν^x = σT Δ^{xν} ∂_ν α
# =============================================================================

def IS_pde_residuals(J0, alpha, nu_x, tx, T_bg, v_bg):
    """
    Compute both IS PDE residuals given PINN outputs and background fields.

    Parameters
    ----------
    J0    : J^0 from PINN, shape [N,1], requires_grad via tx
    alpha : α   from PINN, shape [N,1], requires_grad via tx
    nu_x  : ν^x from PINN, shape [N,1], requires_grad via tx
    tx    : PINN input tensor with requires_grad=True, shape [N,2]
    T_bg  : background temperature T(t,x), shape [N,1]
    v_bg  : background velocity v(t,x),    shape [N,1]

    Returns
    -------
    R_cons : residual of charge conservation, shape [N,1]
    R_IS   : residual of IS relaxation eq.,    shape [N,1]

    BDNK analogue: in the BDNK notebook the two PDE residuals are
        R1 = ∂_t J^0 + ∂_x J^x      (same as R_cons here)
        R2 = ∂_t α + N_0             (replaced here by R_IS)
    """
    grads  = extract_spacetime_grads(alpha, nu_x, J0, tx)
    fluxes = IS_fluxes(J0, alpha, nu_x, T_bg, v_bg)

    alpha_t = grads['alpha_t'];  alpha_x = grads['alpha_x']
    nu_x_t  = grads['nu_x_t'];   nu_x_x  = grads['nu_x_x']
    J0_t    = grads['J0_t']

    sigma  = fluxes['sigma']
    tau_nu = fluxes['tau_nu']
    Jx     = fluxes['Jx']

    # ── R1: charge conservation  ∂_t J^0 + ∂_x J^x ───────────────────────
    # ∂_x J^x = ∂_x(γnv + ν^x).  We differentiate J^x through autograd.
    Jx_grad = torch.autograd.grad(
        Jx, tx,
        grad_outputs=torch.ones_like(Jx),
        create_graph=True, retain_graph=True, allow_unused=True
    )[0]
    Jx_x = Jx_grad[:, 1:2]

    R_cons = J0_t + Jx_x

    # ── R2: IS relaxation residual ─────────────────────────────────────────
    R_IS = IS_relaxation_residual_func(
        nu_x=nu_x, nu_x_t=nu_x_t, nu_x_x=nu_x_x,
        alpha=alpha, alpha_t=alpha_t, alpha_x=alpha_x,
        sigma=sigma, tau_nu=tau_nu, T=T_bg, v=v_bg,
    )

    return R_cons, R_IS


# =============================================================================
# Initial condition helpers
#
# These mirror IC_1D.py from the BDNK repository.
# For IS, the initial state specifies (J^0_0, α_0, ν^x_0).
# ν^x_0 is typically zero (no diffusion current at t=0) but can be
# set to the Navier–Stokes value  ν^x_NS = σ T ∂_x α  if desired.
# =============================================================================

def nu_x_NS_IC(alpha_IC, x, T0):
    """
    Navier–Stokes (first-order) initial condition for ν^x:

        ν^x_NS(x, 0) = σ(α_0, T_0) · T_0 · ∂_x α_0

    This is the NS limit of the IS relaxation equation (τ_ν → 0).
    Using this IC reduces the initial transient in IS evolution.

    BDNK analogue: none — in BDNK ν^x is not an independent field.
    """
    T_const = T0 * torch.ones_like(alpha_IC)
    sig     = sigma_func(alpha_IC, T_const)
    dalpha_dx = torch.autograd.grad(
        alpha_IC, x,
        grad_outputs=torch.ones_like(alpha_IC),
        create_graph=True, retain_graph=True
    )[0]
    return sig * T_const * dalpha_dx


def zero_nu_x_IC(alpha_IC):
    """Zero initial diffusion current (simplest choice)."""
    return torch.zeros_like(alpha_IC)
