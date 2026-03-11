#!shebang
"""
Script description: using PINNs to solve Navier-Stokes (non-relativistic) equations. Probably I will divide this script in several pieces to have a modular code. 
"""

import torch 
from torch.autograd import grad # (automatic derivatives) 
from PyDoe import lhs # Latin Hypercube for collacation points?


# usually we define a class for a PINN: input (space and time?) and output (physical quantities)

# Neural Network Architecture: usually multi-perceptron layers (MPL)
class PINN:

    def __init__(self):
        self.attribute 

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

# define loss function
def loss_function:
    L_IC
    L_BC
    L_PDE
    L_total = L_IC + L_BC + L_PDE
    return 

if __name___ == '__main__':
