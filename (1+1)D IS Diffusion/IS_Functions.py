# Author: OptimusThi

import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import numpy as np

# QCD-like equation of state (Conformal (since ε = 3P))
N_c = 3 # colors
N_f = 3 # 3 flavors
