import torch

# switch between constant Lambda and RTA Lambda(α) ---
USE_RTA_LAMBDA = False          # set to True to use 12/n(α)
CONST_LAMBDA   = 500.0          # used when USE_RTA_LAMBDA = False

# Constant temperature (GeV)
T_const = 0.3
T_cube = T_const ** 3   # 0.027

# ---------- Equilibrium equation of state ----------
def P0(alpha):
    T4 = T_const ** 4
    return (95 * torch.pi**2 / 180) * T4 + (T4 / 6) * alpha**2 + (T4 / 108) * alpha**4

def dP0_dalpha(alpha):
    T4 = T_const ** 4
    return (T4 / 3) * alpha + (T4 / 27) * alpha**3

def n_func(alpha):
    """Charge density n(α) = (1/T) dP0/dα = (T³/3)α + (T³/27)α³."""
    return (T_cube / 3.0) * alpha + (T_cube / 27.0) * alpha**3

def sigma_func(alpha):
    """Conductivity σ(α) = (15T/(4π)) * (1/27 + α²/(243π²))."""
    return (15 * T_const / (4 * torch.pi)) * (1.0/27.0 + alpha**2 / (243.0 * torch.pi**2))

def sigmaT_func(alpha):
    """σT used in the source term."""
    return sigma_func(alpha) * T_const

def alpha_from_n_func(n):
    """Inverse: given n, solve the cubic for α.
       n = a α + b α³  with a = T³/3, b = T³/27.
       Solution of b α³ + a α - n = 0 (depressed cubic)."""
    a = T_cube / 3.0
    b = T_cube / 27.0
    p = a / b
    q = -n / b
    delta = (q / 2.0)**2 + (p / 3.0)**3
    sqrt_delta = torch.sqrt(delta)
    # Robust cube root that works for negative arguments
    t1 = -q/2.0 + sqrt_delta
    t2 = -q/2.0 - sqrt_delta
    u = torch.sign(t1) * torch.pow(torch.abs(t1), 1.0/3.0)
    v = torch.sign(t2) * torch.pow(torch.abs(t2), 1.0/3.0)
    return u + v    

# ---------- Background fields (constant) ----------
def T_func(t, x):
    return T_const * torch.ones_like(x)

def v_func(t, x):
    return torch.zeros_like(x) 

def lambda_func(alpha):
    """
    Λ(α) = τ_J/(σT)
    
    Two options (controlled by USE_RTA_LAMBDA):
      - False: constant Λ = CONST_LAMBDA (simplest model)
      - True : Λ(α) = 12 / n(α)   (RTA model)
    """
    if USE_RTA_LAMBDA:
        return 12.0 / (n_func(alpha) + 1e-30)
    else:
        # constant lambda: return a tensor of the same shape as alpha
        return torch.full_like(alpha, CONST_LAMBDA)    
