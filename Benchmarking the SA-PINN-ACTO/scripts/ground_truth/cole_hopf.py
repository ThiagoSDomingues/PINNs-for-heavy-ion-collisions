import numpy as np

def cole_hopf_burgers_1d(x, t, nu, L, K=10):
    """
    Exact solution of viscous Burgers using periodic Cole–Hopf.

    x  : (Nx,) 1D array of spatial points
    t  : (Nt,) 1D array of time points
    nu : viscosity
    returns: u (Nt, Nx)
    """
    # (Nt, Nx) grids
    Nx = len(x)
    Nt = len(t)

    dx = x[1] - x[0]

    # Precompute psi(y) or phi(0,x)
    y = x.copy()

    psi = np.exp(
        -(L / (2 * nu * np.pi)) *
        (np.cos(np.pi * y / L) - 1.0)
    )
    
    u = np.zeros((Nt, Nx))

    for ti, tt in enumerate(t):

        if tt == 0:
            u[ti, :] = -np.sin(np.pi * x / L)
            continue
        
        # Build convolution
        phi = np.zeros(Nx)

        for k in range(-K, K + 1):
            shift = 2 * k * L

            diff = x[:, None] - (y[None, :] + shift)

            G = np.exp(-diff**2 / (4 * nu * tt)) / np.sqrt(4 * np.pi * nu * tt)

            phi += np.sum(G * psi[None, :], axis=1) * dx

            # Compute derivative using central difference
            dphi_dx = np.gradient(phi, dx)

            u[ti, :] = -2 * nu * dphi_dx / phi            
    
    return u 
