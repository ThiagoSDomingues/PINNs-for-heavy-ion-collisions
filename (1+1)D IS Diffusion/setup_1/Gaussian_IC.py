"""
Script to generate a Gaussian initial condition for IS PINNs outputs: alpha and scriptJ
"""

import numpy as np, math

# Exact IC parameters from the paper
L     = 50.0    # GeV^-1  (half-domain)
T     = 0.3     # GeV
Nc, Nf, PI = 3, 3, math.pi

x = np.linspace(-L, L, 2000)

# ICs
n_ic  = 0.2  * np.exp(-(7*x/L)**2)  + 1.00
J0_ic = 0.05 * np.exp(-(10*x/L)**2) + 1.05

# Standard deviations
sig_n  = L / (7  * math.sqrt(2))
sig_J0 = L / (10 * math.sqrt(2))

# alpha from n
def alpha(n):
    return 27*n / (Nc*Nf*T**3)

alpha_ic = alpha(n_ic)

print(f"n IC:  bg={1.00}, amp={0.20}, peak={n_ic.max():.3f},  σ={sig_n:.3f} GeV^-1")
print(f"J0 IC: bg={1.05}, amp={0.05}, peak={J0_ic.max():.4f}, σ={sig_J0:.3f} GeV^-1")
print(f"α IC:  min={alpha_ic.min():.3f}, max={alpha_ic.max():.3f}")
print(f"S_N={1.0}, S_alpha={alpha(1.0):.4f}")
print(f"Snapshot times: 0, {20/3:.4f}, {2*20/3:.4f}, 20.0 GeV^-1")

#Output

#n IC:  bg=1.0, amp=0.2, peak=1.200,  σ=5.051 GeV^-1
#J0 IC: bg=1.05, amp=0.05, peak=1.1000, σ=3.536 GeV^-1
#α IC:  min=111.111, max=133.333
#S_N=1.0, S_alpha=111.1111 scaling N and alpha
#Snapshot times: 0, 6.6667, 13.3333, 20.0 GeV^-1
