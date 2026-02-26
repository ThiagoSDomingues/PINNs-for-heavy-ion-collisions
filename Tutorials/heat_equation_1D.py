# Author: OptimusThi

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

