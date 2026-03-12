#!shebang

"""
Physics-Informed Neural Network (PINN) for Israel-Stewart diffusion: forward problem (evolve n(t,x) and J(t,x)). 
The PINN must learn (n(t,x), J(t,x)) subject to local rest frame pde's for conserved current and relaxation time equation.
Solves the (1+1)D Israel-Stewart diffusion system in the local rest frame (v=0, fixed uniform temperature).

NN(t,x) --> (n, J)

"""

import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------
# constants
# ------------------

Nc = 3 
Nf = 3 
T = 0.2 # constant temperature

# ------------------
# thermodynamics functions
# ------------------

def alpha_from_n(n):
    return

def sigma(alpha):
    return
    
def tau_J(alpha,n):
    return 12*sigma(alpha)*T/n

# ------------------
# neural network
# ------------------

class PINN(nn.Module):
    
    def __init__(self, layers):
     
# ------------------
# derivatives
# ------------------

def gradients(y,x):
    return torch.autograd.grad()
    
# ------------------
# PDE residual
# ------------------    

def residual(model, t,x):

# ------------------
# training
# ------------------

model = 

optimizer = torch.optim.Adam()

for it in range(20000): # 20,000 ? 
    
    loss = loss_fn(model, t_r, x_r, t0, x0, n0)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if it % 500 == 0:
        print(it, loss.item())
