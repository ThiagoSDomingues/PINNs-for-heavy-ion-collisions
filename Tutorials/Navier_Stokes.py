#!shebang
"""
PINN for 2D incompressible Navier-Stokes (non-relativistic).
Inputs: (x, y, t)
Outputs: (u, v, p)
"""

import torch 
from torch.autograd import grad # automatic derivatives 
from pyDOE import lhs # Latin Hypercube for collocation points?


# usually we define a class for a PINN: input (space and time?) and output (physical quantities)

# -----------------
# Neural Network (MLP)
# -----------------
class PINN_NS(args):

    def __init__(self, layers):
        self.attribute = something

    def function:
        return result

# training dataset: do we call it collocation points?
x = 
y = 
t = 
# Initial conditions 
IC
# Boundary conditions
BC

# define the activation function here? 
# ReLU, tanh, etc...

# -----------------
# PINN wrapper
# -----------------
class NavierStokesPINN:
    def __init__(self, args):
        """
        class description
        """
    def class_function:
        return
    
# PDE residual loss 
def loss(self, args):
    # -----------------
    # Initial condition
    # -----------------
    loss_ic = 
    
    # ------------------
    # Boundary condition
    # ------------------
    
    loss_bc = 
    # -----------------
    # PDE residual loss
    # -----------------
    loss_pde = 
    loss_total = loss_ic + loss_bc + loss_pde
    return loss_total, loss_ic, loss_bc, loss_pde

if __name___ == '__main__':
