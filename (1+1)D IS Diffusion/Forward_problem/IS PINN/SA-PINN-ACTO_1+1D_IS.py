import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Physical constants
# =========================
L = 10.0
t_max = 5.0

kappa = 0.5
tau_q = 0.2
P_e = 1/3

# =========================
# Neural Network
# =========================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        width = 80
        depth = 8

        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 4)]  # [eps_hat, u_hat, n_hat, q_hat]

        self.net = nn.Sequential(*layers)

    def forward(self, t, x):
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)

model = PINN().to(device)

# =========================
# ACTO: Hard IC + Periodic BC
# =========================
def ACTO_transform(t, x, raw, IC_func):
    beta = 1.0
    u_hat = raw
    u_IC = IC_func(x)

    # Initial condition enforcement
    u_tilde = u_IC * torch.exp(-beta*t) + u_hat * (1 - torch.exp(-beta*t))

    # Periodic BC enforcement
    xL = torch.full_like(x, L)
    xLm = torch.full_like(x, -L)

    u_L = model(t, xL)
    u_mL = model(t, xLm)

    correction = ((x + L)/(2*L)) * (u_L - u_mL)
    return u_tilde - correction

# =========================
# Initial Conditions
# =========================
def IC(x):
    eps0 = 1.0 + 0.1*torch.exp(-x**2)
    u0 = torch.zeros_like(x)
    n0 = 1.0 + 0.05*torch.exp(-x**2)
    q0 = torch.zeros_like(x)
    return torch.cat([eps0, u0, n0, q0], dim=1)

# =========================
# Helpers
# =========================
def compute_physics(vars):
    eps, u, n, q = vars[:,0:1], vars[:,1:2], vars[:,2:3], vars[:,3:4]

    u0 = torch.sqrt(1 + u**2)
    P = P_e * eps

    T = eps**0.25  # EOS
    return eps, u, u0, n, q, P, T

# =========================
# PDE Residuals
# =========================
def residuals(t, x):

    t.requires_grad_(True)
    x.requires_grad_(True)

    raw = model(t, x)
    vars = ACTO_transform(t, x, raw, IC)

    eps, u, u0, n, q, P, T = compute_physics(vars)

    # ===== derivatives =====
    def grad(f):
        return torch.autograd.grad(f, (t,x),
            grad_outputs=torch.ones_like(f),
            create_graph=True)

    eps_t, eps_x = grad(eps)
    u_t, u_x = grad(u)
    q_t, q_x = grad(q)
    T_t, T_x = grad(T)

    # ===== T^{μν} =====
    T00 = eps*(u0**2) + P*(u**2)
    T01 = (eps + P)*u0*u
    T11 = -eps + (eps+P)*(u0**2)

    T00_t, T00_x = grad(T00)
    T01_t, T01_x = grad(T01)
    T11_t, T11_x = grad(T11)

    # ===== J^μ =====
    J0 = n*u0 + (u/u0)*q
    J1 = n*u + q

    J0_t, J0_x = grad(J0)
    J1_t, J1_x = grad(J1)

    # ===== Conservation =====
    R_energy = T00_t + T01_x
    R_momentum = T01_t + T11_x
    R_charge = J0_t + J1_x

    # ===== IS relaxation =====
    u_dot_T = u0*T_t + u*T_x
    u_dot_u = u0*u_t + u*u_x

    flux_q = (u/u0)*q
    flux_q_x = grad(flux_q)[1]

    S_q = (
        q*u_x/(u0**3)
        + (u0**2 * 4*eps*kappa)/(3*n*u0*tau_q)*u_dot_T
        + (1/u0)*((u*q)/(u0**2) + (4*eps*kappa*T)/(3*n*tau_q))*u_dot_u
        - q/(u0*tau_q)
    )

    R_q = q_t + flux_q_x - S_q

    return torch.cat([R_energy, R_momentum, R_charge, R_q], dim=1)

# =========================
# SA-PINN weights
# =========================
class SAPINN(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.z = nn.Parameter(torch.zeros(N,1))

    def forward(self):
        return torch.log(1 + torch.exp(self.z))  # softplus

# =========================
# Training
# =========================
N_f = 8000
sapinn = SAPINN(N_f).to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(sapinn.parameters()), lr=1e-3
)

for epoch in range(8000):

    t = torch.rand(N_f,1).to(device)*t_max
    x = (2*torch.rand(N_f,1)-1).to(device)*L

    R = residuals(t,x)
    lambdas = sapinn()

    loss = torch.mean((lambdas * R)**2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(epoch, loss.item())

# =========================
# L-BFGS refinement
# =========================
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=500)

def closure():
    optimizer_lbfgs.zero_grad()
    t = torch.rand(N_f,1).to(device)*t_max
    x = (2*torch.rand(N_f,1)-1).to(device)*L
    R = residuals(t,x)
    loss = torch.mean(R**2)
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)

print("Training complete.")
