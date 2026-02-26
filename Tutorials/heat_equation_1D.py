# Author: OptimusThi
# Code: heat propagation in a metal rod (heat equation in 1D)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Neural Network: we define a model as a class
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), # The input is 2 (Position $x$ and Time $t$). It outputs 32 hidden features;
            nn.Linear(32, 32), nn.Tanh(), # Tanh is the "Activation Function." It’s a smooth curve;
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1) # The final output is 1 (The temperature $u$ at that specific $x$ and $t$).
        )

    def forward(self, x, t):
        # Concatenate x and t as input features
        u = self.net(torch.cat([x, t], dim=1))
        return u

def physics_loss(model, x, t, alpha=0.01):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    
    # Calculate derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    
    residual = u_t - alpha * u_xx
    return torch.mean(residual**2)

# 3. Training Setup
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

# Data Generation (Collocation points)
x_col = torch.rand(2000, 1)
t_col = torch.rand(2000, 1)

# Boundary/Initial Conditions (IC/BC)
x_ic = torch.rand(500, 1)
t_ic = torch.zeros(500, 1)
u_ic = torch.sin(np.pi * x_ic) # IC: u(x,0) = sin(pi*x)

x_bc = torch.cat([torch.zeros(250, 1), torch.ones(250, 1)], dim=0)
t_bc = torch.rand(500, 1)
u_bc = torch.zeros(500, 1) # BC: u(0,t) = u(1,t) = 0

# Training Loop
