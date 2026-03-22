import torch
import torch.nn as nn
import torch.autograd
from BDNK_IS_Functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
