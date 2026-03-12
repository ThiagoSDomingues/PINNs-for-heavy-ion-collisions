import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

# ============================================================
# constants
# ============================================================

Nc = 3
Nf = 3
T = 0.2

# ============================================================
# thermodynamics
# ============================================================

def alpha_from_n(n):
    return 27*n/(Nc*Nf*T**3)

def sigma(alpha):
    return (5*Nc*Nf*T/(12*np.pi))*(1/27 + alpha**2/(243*np.pi**2))

def tau_J(alpha,n):
    return 12*sigma(alpha)*T/n


# ============================================================
# neural network
# ============================================================

class PINN(nn.Module):

    def __init__(self,layers):

        super().__init__()

        self.activation = nn.Tanh()

        self.layers = nn.ModuleList()

        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i],layers[i+1]))

    def forward(self,x):

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        return self.layers[-1](x)


# ============================================================
# derivatives
# ============================================================

def gradients(y,x):

    return torch.autograd.grad(
        y,x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]


# ============================================================
# PDE residual
# ============================================================

def residual(model,t,x):

    tx = torch.cat([t,x],dim=1)
    tx.requires_grad_(True)

    pred = model(tx)

    n = pred[:,0:1]
    J = pred[:,1:2]

    dn = gradients(n,tx)
    dJ = gradients(J,tx)

    dn_dt = dn[:,0:1]
    dn_dx = dn[:,1:2]

    dJ_dt = dJ[:,0:1]
    dJ_dx = dJ[:,1:2]

    alpha = alpha_from_n(n)

    sig = sigma(alpha)
    tau = tau_J(alpha,n)

    tau_sig = tau/(sig*T)

    dt_tau_sig = gradients(tau_sig,tx)[:,0:1]

    R1 = dn_dt + dJ_dx

    R2 = (
        tau*dJ_dt
        + J
        + 0.5*sig*T*J*dt_tau_sig
        + sig*T*gradients(alpha,tx)[:,1:2]
    )

    return R1,R2


# ============================================================
# loss
# ============================================================

def loss_fn(model,t_r,x_r,t0,x0,n0):

    R1,R2 = residual(model,t_r,x_r)

#    loss_pde = torch.mean(R1**2 + R2**2)

    tx0 = torch.cat([t0,x0],dim=1)
    pred0 = model(tx0)

    n_pred = pred0[:,0:1]
    J_pred = pred0[:,1:2]

    loss_ic = torch.mean((n_pred-n0)**2) + torch.mean(J_pred**2)

#    return loss_pde + loss_ic
    return loss_ic


# ============================================================
# domain
# ============================================================

t_min = 0
t_max = 5

x_min = -10
x_max = 10

# collocation points

N_r = 10000

t_r = torch.rand(N_r,1)*(t_max-t_min)+t_min
x_r = torch.rand(N_r,1)*(x_max-x_min)+x_min

t_r = t_r.to(device)
x_r = x_r.to(device)


# ============================================================
# initial condition
# ============================================================

N0 = 1000

x0 = torch.linspace(x_min,x_max,N0).view(-1,1)
t0 = torch.zeros_like(x0)

# gaussian perturbation

n_background = 0.0001 # 10^-4
amp = 0.00002
width = 2

n0 = n_background + amp*torch.exp(-x0**2/width**2)

x0 = x0.to(device)
t0 = t0.to(device)
n0 = n0.to(device)


# ============================================================
# model
# ============================================================

model = PINN([2,50,50,50,50,50,50,50,2]).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)


# ============================================================
# training
# ============================================================

epochs = 20000

for it in range(epochs):

    loss = loss_fn(model,t_r,x_r,t0,x0,n0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 500 == 0:
        print("iter:",it,"loss:",loss.item())


# ============================================================
# evaluation
# ============================================================

Nx = 200
Nt = 100

x = torch.linspace(x_min,x_max,Nx)
t = torch.linspace(t_min,t_max,Nt)

X,T = torch.meshgrid(x,t,indexing='ij')

tx = torch.stack([T.flatten(),X.flatten()],dim=1).to(device)

with torch.no_grad():

    pred = model(tx)

n_pred = pred[:,0].cpu().reshape(Nx,Nt)

plt.figure(figsize=(6,5))
plt.contourf(t,x,n_pred,50)
plt.xlabel("t")
plt.ylabel("x")
plt.title("Density evolution n(t,x)")
plt.colorbar()
plt.show()
