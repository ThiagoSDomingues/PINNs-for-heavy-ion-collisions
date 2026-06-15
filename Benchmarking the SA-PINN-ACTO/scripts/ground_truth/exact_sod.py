"""
Exact Riemann solution for the Sod shock-tube problem (1D Euler).
Standard Sod initial conditions: left state (rho=1, v=0, p=1),
right state (rho=0.125, v=0, p=0.1), gamma=1.4, diaphragm at x=0.
"""
import numpy as np
from scipy.optimize import brentq

GAMMA = 1.4

def _sound_speed(rho, p):
    return np.sqrt(GAMMA * p / rho)

def _f(p_star, rho_L, p_L, rho_R, p_R):
    """Rankine–Hugoniot / Riemann fan pressure function."""
    a_L = _sound_speed(rho_L, p_L)
    a_R = _sound_speed(rho_R, p_R)

    # Left wave
    if p_star <= p_L:   # rarefaction
        f_L = (2.0 * a_L / (GAMMA - 1.0)) * ((p_star / p_L) ** ((GAMMA - 1.0) / (2.0 * GAMMA)) - 1.0)
    else:               # shock
        A_L = 2.0 / ((GAMMA + 1.0) * rho_L)
        B_L = (GAMMA - 1.0) / (GAMMA + 1.0) * p_L
        f_L = (p_star - p_L) * np.sqrt(A_L / (p_star + B_L))

    # Right wave
    if p_star <= p_R:   # rarefaction
        f_R = (2.0 * a_R / (GAMMA - 1.0)) * ((p_star / p_R) ** ((GAMMA - 1.0) / (2.0 * GAMMA)) - 1.0)
    else:               # shock
        A_R = 2.0 / ((GAMMA + 1.0) * rho_R)
        B_R = (GAMMA - 1.0) / (GAMMA + 1.0) * p_R
        f_R = (p_star - p_R) * np.sqrt(A_R / (p_star + B_R))

    return f_L + f_R + (0.0)   # contact: v_L* = v_R*

def exact_sod(x: np.ndarray, t: np.ndarray,
              rho_L=1.0, v_L=0.0, p_L=1.0,
              rho_R=0.125, v_R=0.0, p_R=0.1,
              x0: float = 0.0) -> dict:
    """
    Returns dict with keys 'rho', 'v', 'p', 'E', each (Nt, Nx).
    t[0] may be 0, in which case IC is returned.
    """
    a_L = _sound_speed(rho_L, p_L)
    a_R = _sound_speed(rho_R, p_R)

    # Solve for p_star, v_star
    g = lambda p: _f(p, rho_L, p_L, rho_R, p_R) + (v_R - v_L)
    p_star = brentq(g, 1e-8 * min(p_L, p_R), 10.0 * max(p_L, p_R), xtol=1e-12)

    if p_star <= p_L:
        f_L = (2.0 * a_L / (GAMMA - 1.0)) * ((p_star / p_L) ** ((GAMMA - 1.0) / (2.0 * GAMMA)) - 1.0)
    else:
        A_L = 2.0 / ((GAMMA + 1.0) * rho_L)
        B_L = (GAMMA - 1.0) / (GAMMA + 1.0) * p_L
        f_L = (p_star - p_L) * np.sqrt(A_L / (p_star + B_L))

    v_star = v_L - f_L

    Nt, Nx = len(t), len(x)
    rho_arr = np.zeros((Nt, Nx))
    v_arr   = np.zeros((Nt, Nx))
    p_arr   = np.zeros((Nt, Nx))

    for ti, tt in enumerate(t):
        if tt == 0.0:
            rho_arr[ti] = np.where(x < x0, rho_L, rho_R)
            v_arr[ti]   = np.where(x < x0, v_L,   v_R)
            p_arr[ti]   = np.where(x < x0, p_L,   p_R)
            continue

        xi = (x - x0) / tt   # self-similar variable

        rho = np.empty(Nx)
        v   = np.empty(Nx)
        p   = np.empty(Nx)

        for j in range(Nx):
            s = xi[j]

            # ---- Left rarefaction (p_star < p_L) ----
            if p_star <= p_L:
                s_head = v_L - a_L          # head of rarefaction
                a_star_L = a_L * (p_star / p_L) ** ((GAMMA - 1.0) / (2.0 * GAMMA))
                s_tail = v_star - a_star_L  # tail of rarefaction
                if s <= s_head:
                    rho[j], v[j], p[j] = rho_L, v_L, p_L
                elif s <= s_tail:
                    v_fan   = 2.0 / (GAMMA + 1.0) * (a_L + (GAMMA - 1.0) / 2.0 * v_L + s)
                    a_fan   = 2.0 / (GAMMA + 1.0) * (a_L + (GAMMA - 1.0) / 2.0 * (v_L - s))
                    rho_fan = rho_L * (a_fan / a_L) ** (2.0 / (GAMMA - 1.0))
                    p_fan   = p_L * (a_fan / a_L) ** (2.0 * GAMMA / (GAMMA - 1.0))
                    rho[j], v[j], p[j] = rho_fan, v_fan, p_fan
                elif s <= v_star:
                    rho_star_L = rho_L * (p_star / p_L) ** (1.0 / GAMMA)
                    rho[j], v[j], p[j] = rho_star_L, v_star, p_star
                else:
                    # right of contact
                    pass
            else:
                # Left shock
                A_L = 2.0 / ((GAMMA + 1.0) * rho_L)
                B_L = (GAMMA - 1.0) / (GAMMA + 1.0) * p_L
                s_shock_L = v_L - a_L * np.sqrt((GAMMA + 1.0) / (2.0 * GAMMA) * p_star / p_L + (GAMMA - 1.0) / (2.0 * GAMMA))
                rho_star_L = rho_L * ((p_star / p_L + (GAMMA - 1.0) / (GAMMA + 1.0)) /
                                      ((GAMMA - 1.0) / (GAMMA + 1.0) * p_star / p_L + 1.0))
                if s <= s_shock_L:
                    rho[j], v[j], p[j] = rho_L, v_L, p_L
                elif s <= v_star:
                    rho[j], v[j], p[j] = rho_star_L, v_star, p_star
                else:
                    pass

            # ---- Right shock (standard Sod: p_star > p_R always) ----
            A_R = 2.0 / ((GAMMA + 1.0) * rho_R)
            B_R = (GAMMA - 1.0) / (GAMMA + 1.0) * p_R
            rho_star_R = rho_R * ((p_star / p_R + (GAMMA - 1.0) / (GAMMA + 1.0)) /
                                   ((GAMMA - 1.0) / (GAMMA + 1.0) * p_star / p_R + 1.0))
            s_shock_R = v_R + a_R * np.sqrt((GAMMA + 1.0) / (2.0 * GAMMA) * p_star / p_R + (GAMMA - 1.0) / (2.0 * GAMMA))

            if s > s_shock_R:
                rho[j], v[j], p[j] = rho_R, v_R, p_R
            elif s > v_star:
                rho[j], v[j], p[j] = rho_star_R, v_star, p_star

        rho_arr[ti] = rho
        v_arr[ti]   = v
        p_arr[ti]   = p

    E_arr = p_arr / (GAMMA - 1.0) + 0.5 * rho_arr * v_arr ** 2
    return {'rho': rho_arr, 'v': v_arr, 'p': p_arr, 'E': E_arr}
