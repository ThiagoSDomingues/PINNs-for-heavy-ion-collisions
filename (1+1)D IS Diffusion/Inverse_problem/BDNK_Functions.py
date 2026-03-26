import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import numpy as np

N_c = 3
N_f = 3
C_B = 1/(4*np.pi) # Third setup
#C_B = 0.4        # First and second setup

# For the BDNK background simulations
_T_tab = None
_v_tab = None
_t0 = None
_dt = None
_Nt = None
_x0 = None
_dx = None
_Nx = None
_Lval = None
_T_tmax = 20.0  # 20.0 GeV^-1 by default

_EPS_TO_T_DENOM = 15.6268736

def _BDNK_base_dir():
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
    global _T_tab, _v_tab, _t0, _dt, _Nt, _x0, _dx, _Nx, _Lval, _T_tmax

    base = _BDNK_base_dir()
    folder = _pick_BDNK_subfolder(base, BDNK_simulation)

    ep_path = os.path.join(folder, "ep(t,x).npy")
    v_path  = os.path.join(folder, "v(t,x).npy")
    if not os.path.exists(ep_path): raise FileNotFoundError(ep_path)
    if not os.path.exists(v_path):  raise FileNotFoundError(v_path)

    ep_tx = np.load(ep_path)
    v_tx  = np.load(v_path)
    if ep_tx.shape != v_tx.shape:
        raise ValueError(f"Shape mismatch: ep {ep_tx.shape} vs v {v_tx.shape}")

    Nt_data, Nx_data = ep_tx.shape

    t_axis = np.linspace(0.0, tmax, Nt_data, dtype=np.float64)
    if np.allclose(ep_tx[:, 0], ep_tx[:, -1]) and np.allclose(v_tx[:, 0], v_tx[:, -1]):
        ep_tx = ep_tx[:, :-1]
        v_tx  = v_tx[:, :-1]
    Nx_data = ep_tx.shape[1]
    
    dx_data = 2.0 * L_in / Nx_data
    x0_data = -L_in + 0.5 * dx_data

    T_tx = (ep_tx.astype(np.float64) / _EPS_TO_T_DENOM) ** 0.25

    _T_tab = torch.tensor(T_tx, dtype=torch.float32, device=device)
    _v_tab = torch.tensor(v_tx,  dtype=torch.float32, device=device)

    _t0  = torch.tensor(float(t_axis[0]), dtype=torch.float32, device=device)
    _dt  = torch.tensor(float((t_axis[-1] - t_axis[0]) / (Nt_data - 1) if Nt_data > 1 else 1.0),
                        dtype=torch.float32, device=device)
    _Nt  = torch.tensor(int(Nt_data), dtype=torch.int64, device=device)

    _x0  = torch.tensor(float(x0_data), dtype=torch.float32, device=device)
    _dx  = torch.tensor(float(dx_data), dtype=torch.float32, device=device)
    _Nx  = torch.tensor(int(Nx_data), dtype=torch.int64, device=device)

    _Lval    = torch.tensor(float(L_in), dtype=torch.float32, device=device)
    _T_tmax  = torch.tensor(float(tmax), dtype=torch.float32, device=device)

def _cubic_kernel_1d(u, a=-0.5):
    absu = u.abs()
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

    NtI = int(_Nt.item()); NxI = int(_Nx.item())
    offsets = torch.tensor([-1, 0, 1, 2], device=device, dtype=torch.int64)
    
    it_neighbors = (it0.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, NtI - 1).to(torch.int64)
    ix_base = torch.remainder(ix0, NxI).to(torch.int64)
    ix_neighbors = torch.remainder(ix_base.unsqueeze(1) + offsets.unsqueeze(0), NxI).to(torch.int64)
    
    wt = torch.stack([_cubic_kernel_1d(dt_frac - k) for k in (-1., 0., 1., 2.)], dim=1)
    wx = torch.stack([_cubic_kernel_1d(dx_frac - k) for k in (-1., 0., 1., 2.)], dim=1)
    
    tab = tab_2d.reshape(NtI * NxI)
    vals = []
    for j in range(4):
        idx_row_4 = (it_neighbors[:, j].unsqueeze(1) * NxI + ix_neighbors)
        vals.append(tab[idx_row_4])
    vals = torch.stack(vals, dim=1)
    
    vx = (vals * wx.unsqueeze(1)).sum(dim=2)
    out = (vx * wt).sum(dim=1)

    return out.view_as(tb)

def T_func(t, x):
    return 0.3 * torch.ones_like(x)          # This option enabled for arbitrary function bakgrounds (first and second setups)
    #return _bicubic_sample_tx(t, x, _T_tab) # This option enabled for BDNK backgrounds (third setup)

def v_func(t, x):
    return 0.0 * torch.ones_like(x)          # This option enabled for arbitrary function bakgrounds (first and second setups)
    #return _bicubic_sample_tx(t, x, _v_tab) # This option enabled for BDNK backgrounds (third setup)
    
def gamma_func(v):
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.float64, device=device)
    return 1.0 / torch.sqrt(1 - v**2)

def alpha_from_n_func(n, T):
    a = T**3 * N_c * N_f
    b = torch.sqrt(
        2187 * n**2 * T**12 * N_c**4 * N_f**4
        + 4 * torch.pi**2 * a**6
    )
    c = 81 * n * a**2
    d = torch.pow(torch.sqrt(torch.tensor(3.0, dtype=n.dtype, device=n.device)) * b + c, 1/3)
    
    term1 = (3/2)**(1/3) * torch.pi**(2/3) * d / a
    term2 = 2**(1/3) * 3**(2/3) * torch.pi**(4/3) * a / d
    
    return term1 - term2

def n_from_alpha_func(alpha, T):
    term1 = alpha / 27
    term2 = (alpha ** 3) / (243 * torch.pi ** 2)
    return N_c * N_f * T**3 * (term1 + term2)

def N_x_func(alpha, x):
    grad = torch.autograd.grad(
        alpha, x,
        grad_outputs=torch.ones_like(alpha),
        create_graph=True, retain_graph=True,
        allow_unused=True
    )[0]
    
    return -grad[:,1:2]

def mu_func(alpha, T):
    return alpha * T

def pressure_func(alpha, T):
    pi = torch.pi
    mu = mu_func(alpha, T)
    term1 = (2*(N_c**2 - 1) + 3.5 * N_c * N_f) * pi**2 * T**4 / 90
    term2 = N_c * N_f * mu**2 * T**2 / 54
    term3 = N_c * N_f * mu**4 / (972 * pi**2)
    return term1 + term2 + term3

def sigma_func(alpha, T):
    mu = mu_func(alpha, T)
    n = n_from_alpha_func(alpha, T)
    P = pressure_func(alpha, T)
    eps = 3*P
    return C_B * n * (1/3 * 1/torch.tanh(alpha) - n*T/(eps+P)) / (T**2)

def lambd_func(sigma):
    cch = 0.5
    return sigma/(cch**2)

def N_0_func(lambd, sigma, T, J0, n, N_x, v):
    gamma = gamma_func(v)
    num = -J0 + gamma * n + (sigma - lambd) * T * gamma**2 * v * N_x
    denom = sigma * T + (lambd - sigma) * T * gamma**2
    return num / denom

def Jx_func(n, sigma, lambd, T, N_x, N_0, v):
    Nx = N_x
    gamma = gamma_func(v)
    return (gamma * n * v
            + sigma * T * Nx
            + gamma**2 * T * (sigma - lambd) * v**2 * Nx
            + gamma**2 * T * (sigma - lambd) * v * N_0)

def J0_func(T, v, alpha, alpha_t, x):
    gamma = gamma_func(v)
    sigma = sigma_func(alpha, T)
    lambd = lambd_func(sigma)
    
    alpha_x = torch.autograd.grad(
        alpha, x,
        grad_outputs=torch.ones_like(alpha),
        create_graph=True
    )[0][:, 1:2]
    
    n = n_from_alpha_func(alpha, T)
    
    term1 = n * gamma
    term2 = (sigma * T - gamma**2 * T * (sigma - lambd)) * alpha_t
    term3 = gamma**2 * T * (sigma - lambd) * v * alpha_x

    return term1 + term2 - term3
    
def alpha2_and_scriptJ_from_alpha_and_J0(alpha1, t, x):
    """
    Compute the IS matching conditions (n2, alpha2, scriptJ) from a BDNK
    solution (alpha1) using Eq. IS_from_BDNK_1d:

        n2      = n1 + lambda(n1) * T * gamma * (∂_t + v ∂_x) alpha1
        scriptJ = -sigma(n1) * T * gamma * (v ∂_t + ∂_x) alpha1

    where  n1 = n(alpha1, T)  and  alpha2 = alpha(n2, T).

    Parameters
    ----------
    alpha1 : (N, 1) tensor  —  BDNK fugacity, WITH requires_grad=True
                                (so that autograd can differentiate it
                                 w.r.t. the (t, x) inputs below)
    t      : (N, 1) tensor  —  time coordinates,  requires_grad=True
    x      : (N, 1) tensor  —  space coordinates, requires_grad=True

    Returns
    -------
    alpha2  : (N, 1) tensor  —  IS fugacity (derived from n2)
    scriptJ : (N, 1) tensor  —  IS diffusion current  J (scalar component)
    n2      : (N, 1) tensor  —  IS number density
    n1      : (N, 1) tensor  —  BDNK number density (useful for plotting)
    """

    T      = T_func(t, x)          # background temperature field
    v      = v_func(t, x)          # background velocity field
    gamma  = gamma_func(v)         # Lorentz factor

    # ── BDNK density and transport coefficients ──────────────────────────
    # BUG 1 FIX: lambda_func(alpha1, T) → lambd_func(sigma1)
    #   The original code called a non-existent lambda_func(alpha1, T).
    #   lambd_func takes ONE argument (sigma), not two.
    n1     = n_from_alpha_func(alpha1, T)
    sigma1 = sigma_func(alpha1, T)
    lam1   = lambd_func(sigma1)    # λ(n1) = σ(n1) / c_ch²

    # ── Gradients of alpha1 w.r.t. (t, x) ───────────────────────────────
    # We differentiate alpha1 w.r.t. t and x separately so that the graph
    # stays clean.  (We could also cat [t,x] and slice, but separate grads
    # are more readable and avoid index confusion.)
    alpha_t = torch.autograd.grad(
        alpha1, t,
        grad_outputs=torch.ones_like(alpha1),
        create_graph=True, retain_graph=True
    )[0]                            # ∂_t α₁

    alpha_x = torch.autograd.grad(
        alpha1, x,
        grad_outputs=torch.ones_like(alpha1),
        create_graph=True, retain_graph=True
    )[0]                            # ∂_x α₁

    # ── Matching conditions (Eq. IS_from_BDNK_1d) ───────────────────────
    # n2 = n1 + λ(n1) T γ (∂_t + v ∂_x) α₁
    u_dot_grad_alpha = gamma * (alpha_t + v * alpha_x)   # u^ν ∂_ν α₁
    n2 = n1 + lam1 * T * u_dot_grad_alpha

    # scriptJ = −σ(n1) T γ (v ∂_t + ∂_x) α₁
    # BUG 5 FIX: the original code was missing the γ factor and the v*∂_t
    #   term (it set  scriptJ = -sigma1 * T * alpha_x, which is only correct
    #   for v = 0).  The full expression from the paper is:
    #       J = −σ₁ T γ (v ∂_t + ∂_x) α₁
    Delta_grad_alpha = gamma * (v * alpha_t + alpha_x)   # −Δ^{xν} ∂_ν α₁ · (−1)
    scriptJ = -sigma1 * T * Delta_grad_alpha

    # ── Invert n2 → alpha2 ───────────────────────────────────────────────
    # BUG 2 FIX: the original code called alpha_from_n(n2) which does not
    #   exist; the correct name is alpha_from_n_func(n, T).
    alpha2 = alpha_from_n_func(n2, T)

    return alpha2, scriptJ, n2, n1    
