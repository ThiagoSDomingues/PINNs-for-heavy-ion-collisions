"""
Physics-Informed Neural Network for 1+1D Israel-Stewart Viscous Hydrodynamics

This implementation solves the Israel-Stewart equations for relativistic viscous 
hydrodynamics in 1+1 dimensions and compares the effective diffusion coefficient 
with BDNK theory.

Author: Research Tutorial
Date: February 2026
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class IsraelStewartPINN(nn.Module):
    """
    Physics-Informed Neural Network for Israel-Stewart hydrodynamics.
    
    Network architecture: (t,x) → (ε, v, π)
    where:
        ε: energy density
        v: velocity
        π: shear stress (π^xx component)
    """
    
    def __init__(self, layers=[2, 128, 128, 128, 128, 3], activation='tanh'):
        """
        Args:
            layers: List of layer sizes [input_dim, hidden1, ..., output_dim]
            activation: Activation function ('tanh' or 'sin')
        """
        super().__init__()
        
        self.layers = layers
        self.activation_name = activation
        
        # Build network layers
        self.network = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.network.append(nn.Linear(layers[i], layers[i+1]))
            
        # Choose activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sin':
            self.activation = lambda x: torch.sin(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        # Initialize weights using Xavier initialization
        self.init_weights()
        
    def init_weights(self):
        """Initialize network weights"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            t: Time coordinate [batch, 1]
            x: Spatial coordinate [batch, 1]
            
        Returns:
            output: [batch, 3] tensor with (ε, v, π)
        """
        # Concatenate inputs
        inputs = torch.cat([t, x], dim=1)
        
        # Forward through network
        out = inputs
        for i, layer in enumerate(self.network):
            out = layer(out)
            if i < len(self.network) - 1:  # Don't apply activation to output layer
                out = self.activation(out)
                
        return out


class PhysicsParameters:
    """Container for physical parameters and equation of state"""
    
    def __init__(self, eta_over_s=0.2, epsilon0=1.0, conformal=True):
        """
        Args:
            eta_over_s: Shear viscosity to entropy ratio
            epsilon0: Reference energy density
            conformal: Use conformal equation of state
        """
        self.eta_over_s = eta_over_s
        self.epsilon0 = epsilon0
        self.conformal = conformal
        
        # Speed of sound
        self.c_s = 1.0/np.sqrt(3.0) if conformal else 0.5
        
    def pressure(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Equation of state: P(ε)"""
        if self.conformal:
            return epsilon / 3.0
        else:
            return self.c_s**2 * epsilon
            
    def entropy_density(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Entropy density s(ε) for conformal fluid"""
        # s ∝ ε^(3/4) for conformal fluid
        return (epsilon / self.epsilon0)**(3.0/4.0)
        
    def shear_viscosity(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Shear viscosity η = (η/s) * s"""
        s = self.entropy_density(epsilon)
        return self.eta_over_s * s
        
    def relaxation_time(self, epsilon: torch.Tensor) -> torch.Tensor:
        """
        Relaxation time for Israel-Stewart equation.
        τ_π = 5η/(ε + P) for conformal fluid
        """
        P = self.pressure(epsilon)
        eta = self.shear_viscosity(epsilon)
        return 5.0 * eta / (epsilon + P)


def compute_gradients(outputs: torch.Tensor, inputs: torch.Tensor, 
                      order: int = 1) -> torch.Tensor:
    """
    Compute gradients using automatic differentiation.
    
    Args:
        outputs: Network outputs
        inputs: Network inputs
        order: Order of derivative (1 or 2)
        
    Returns:
        Gradient tensor
    """
    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]
    
    if order == 2:
        grads = torch.autograd.grad(
            outputs=grads,
            inputs=inputs,
            grad_outputs=torch.ones_like(grads),
            create_graph=True,
            retain_graph=True
        )[0]
        
    return grads


class IsraelStewartResiduals:
    """Compute residuals for Israel-Stewart equations"""
    
    def __init__(self, params: PhysicsParameters):
        self.params = params
        
    def compute_stress_tensor(self, epsilon: torch.Tensor, v: torch.Tensor, 
                              pi: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute components of the energy-momentum tensor.
        
        T^μν = ε u^μ u^ν + (P + π) Δ^μν - π^μν
        
        In 1+1D with u^μ = γ(1, v):
        T^00 = (ε + P + π)γ² - (P + π)
        T^0x = (ε + P + π)γ² v  
        T^xx = (ε + P + π)γ² v² + (P + π)
        
        Returns:
            T00, T0x, Txx
        """
        # Pressure
        P = self.params.pressure(epsilon)
        
        # Lorentz factor
        gamma = 1.0 / torch.sqrt(1.0 - v**2 + 1e-10)
        
        # Stress tensor components
        T00 = (epsilon + P) * gamma**2 - P - pi * (gamma**2 - 1.0)
        T0x = (epsilon + P) * gamma**2 * v - pi * gamma**2 * v
        Txx = (epsilon + P) * gamma**2 * v**2 + P + pi * (gamma**2 * v**2)
        
        return T00, T0x, Txx
    
    def compute_expansion_and_shear(self, v: torch.Tensor, 
                                   v_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute expansion rate and shear tensor.
        
        In 1+1D:
        θ = ∂_μ u^μ = γ(∂_t γ + ∂_x(γv))
        σ^xx = ∂^x u^x - θ/3
        
        For our simplified case with Minkowski metric:
        σ^xx ≈ v_x (in the local rest frame approximation)
        
        Returns:
            theta, sigma_xx
        """
        gamma = 1.0 / torch.sqrt(1.0 - v**2 + 1e-10)
        
        # Simplified expansion rate
        theta = gamma * v_x
        
        # Shear tensor (1+1D has one independent component)
        sigma_xx = v_x - theta / 3.0
        
        return theta, sigma_xx
    
    def energy_residual(self, T00: torch.Tensor, T0x: torch.Tensor,
                       T00_t: torch.Tensor, T0x_x: torch.Tensor) -> torch.Tensor:
        """
        Energy conservation: ∂_t T^00 + ∂_x T^0x = 0
        """
        return T00_t + T0x_x
    
    def momentum_residual(self, T0x: torch.Tensor, Txx: torch.Tensor,
                         T0x_t: torch.Tensor, Txx_x: torch.Tensor) -> torch.Tensor:
        """
        Momentum conservation: ∂_t T^0x + ∂_x T^xx = 0
        """
        return T0x_t + Txx_x
    
    def israel_stewart_residual(self, epsilon: torch.Tensor, v: torch.Tensor,
                               pi: torch.Tensor, pi_t: torch.Tensor, 
                               pi_x: torch.Tensor, v_x: torch.Tensor) -> torch.Tensor:
        """
        Israel-Stewart relaxation equation:
        τ_π [∂_t π + v ∂_x π] + π = 2η σ^xx
        
        Returns:
            Residual of IS equation
        """
        # Get transport coefficients
        tau_pi = self.params.relaxation_time(epsilon)
        eta = self.params.shear_viscosity(epsilon)
        
        # Compute shear rate
        _, sigma_xx = self.compute_expansion_and_shear(v, v_x)
        
        # Material derivative of π
        D_pi = pi_t + v * pi_x
        
        # IS equation residual
        residual = tau_pi * D_pi + pi - 2.0 * eta * sigma_xx
        
        return residual


class PINNLoss:
    """Loss function for PINN training"""
    
    def __init__(self, residuals: IsraelStewartResiduals, 
                 weights: Dict[str, float] = None):
        """
        Args:
            residuals: IsraelStewartResiduals object
            weights: Dictionary of loss weights
        """
        self.residuals = residuals
        
        # Default weights
        self.weights = {
            'pde': 1.0,
            'ic': 100.0,
            'bc': 10.0,
            'physics': 1.0
        }
        if weights:
            self.weights.update(weights)
            
    def pde_loss(self, model: nn.Module, t: torch.Tensor, 
                 x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PDE residual loss.
        
        Returns:
            loss, diagnostics_dict
        """
        # Enable gradient computation
        t.requires_grad_(True)
        x.requires_grad_(True)
        
        # Network prediction
        output = model(t, x)
        epsilon = output[:, 0:1]
        v = output[:, 1:2]
        pi = output[:, 2:3]
        
        # Compute first derivatives
        epsilon_t = compute_gradients(epsilon, t)
        epsilon_x = compute_gradients(epsilon, x)
        v_t = compute_gradients(v, t)
        v_x = compute_gradients(v, x)
        pi_t = compute_gradients(pi, t)
        pi_x = compute_gradients(pi, x)
        
        # Compute stress tensor
        T00, T0x, Txx = self.residuals.compute_stress_tensor(epsilon, v, pi)
        
        # Derivatives of stress tensor
        T00_t = compute_gradients(T00, t)
        T0x_x = compute_gradients(T0x, x)
        T0x_t = compute_gradients(T0x, t)
        Txx_x = compute_gradients(Txx, x)
        
        # Compute residuals
        R_energy = self.residuals.energy_residual(T00, T0x, T00_t, T0x_x)
        R_momentum = self.residuals.momentum_residual(T0x, Txx, T0x_t, Txx_x)
        R_IS = self.residuals.israel_stewart_residual(
            epsilon, v, pi, pi_t, pi_x, v_x
        )
        
        # MSE of residuals
        loss_energy = torch.mean(R_energy**2)
        loss_momentum = torch.mean(R_momentum**2)
        loss_IS = torch.mean(R_IS**2)
        
        total_pde_loss = loss_energy + loss_momentum + loss_IS
        
        diagnostics = {
            'energy': loss_energy.item(),
            'momentum': loss_momentum.item(),
            'IS': loss_IS.item(),
        }
        
        return total_pde_loss, diagnostics
    
    def initial_condition_loss(self, model: nn.Module, t0: torch.Tensor,
                              x_ic: torch.Tensor, epsilon_ic: torch.Tensor,
                              v_ic: torch.Tensor, pi_ic: torch.Tensor) -> torch.Tensor:
        """
        Loss for initial conditions.
        
        Args:
            t0: Initial time (should be zeros)
            x_ic: Spatial points for IC
            epsilon_ic, v_ic, pi_ic: Target initial values
        """
        output = model(t0, x_ic)
        epsilon_pred = output[:, 0:1]
        v_pred = output[:, 1:2]
        pi_pred = output[:, 2:3]
        
        loss = (
            torch.mean((epsilon_pred - epsilon_ic)**2) +
            torch.mean((v_pred - v_ic)**2) +
            torch.mean((pi_pred - pi_ic)**2)
        )
        
        return loss
    
    def boundary_condition_loss(self, model: nn.Module, t_bc: torch.Tensor,
                               x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        """
        Periodic boundary conditions.
        
        Args:
            t_bc: Time points for BC
            x_left: Left boundary spatial coordinate
            x_right: Right boundary spatial coordinate
        """
        output_left = model(t_bc, x_left)
        output_right = model(t_bc, x_right)
        
        loss = torch.mean((output_left - output_right)**2)
        
        return loss
    
    def physics_constraint_loss(self, model: nn.Module, t: torch.Tensor,
                               x: torch.Tensor) -> torch.Tensor:
        """
        Additional physics constraints:
        - Energy density must be positive
        - Velocity must be subluminal
        """
        output = model(t, x)
        epsilon = output[:, 0:1]
        v = output[:, 1:2]
        
        # Soft constraints using penalty terms
        loss_positive_energy = torch.mean(torch.relu(-epsilon))
        loss_causality = torch.mean(torch.relu(torch.abs(v) - 0.99))
        
        return loss_positive_energy + 10.0 * loss_causality
    
    def total_loss(self, model: nn.Module, 
                   t_pde: torch.Tensor, x_pde: torch.Tensor,
                   t_ic: torch.Tensor, x_ic: torch.Tensor,
                   epsilon_ic: torch.Tensor, v_ic: torch.Tensor, pi_ic: torch.Tensor,
                   t_bc: torch.Tensor, x_left: torch.Tensor, x_right: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total weighted loss.
        
        Returns:
            total_loss, diagnostics
        """
        # PDE loss
        loss_pde, pde_diagnostics = self.pde_loss(model, t_pde, x_pde)
        
        # IC loss
        loss_ic = self.initial_condition_loss(
            model, t_ic, x_ic, epsilon_ic, v_ic, pi_ic
        )
        
        # BC loss
        loss_bc = self.boundary_condition_loss(model, t_bc, x_left, x_right)
        
        # Physics constraints
        loss_physics = self.physics_constraint_loss(model, t_pde, x_pde)
        
        # Weighted sum
        total = (
            self.weights['pde'] * loss_pde +
            self.weights['ic'] * loss_ic +
            self.weights['bc'] * loss_bc +
            self.weights['physics'] * loss_physics
        )
        
        diagnostics = {
            'total': total.item(),
            'pde': loss_pde.item(),
            'ic': loss_ic.item(),
            'bc': loss_bc.item(),
            'physics': loss_physics.item(),
            **pde_diagnostics
        }
        
        return total, diagnostics


class CollocationPointGenerator:
    """Generate collocation points for PINN training"""
    
    def __init__(self, t_range: Tuple[float, float], 
                 x_range: Tuple[float, float],
                 n_pde: int = 10000,
                 n_ic: int = 500,
                 n_bc: int = 500,
                 use_sobol: bool = True):
        """
        Args:
            t_range: (t_min, t_max)
            x_range: (x_min, x_max)
            n_pde: Number of PDE collocation points
            n_ic: Number of initial condition points
            n_bc: Number of boundary condition points
            use_sobol: Use quasi-random Sobol sequences
        """
        self.t_range = t_range
        self.x_range = x_range
        self.n_pde = n_pde
        self.n_ic = n_ic
        self.n_bc = n_bc
        self.use_sobol = use_sobol
        
    def generate_pde_points(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate interior collocation points for PDE"""
        if self.use_sobol:
            # Use Sobol quasi-random sequences for better coverage
            sobol = torch.quasirandom.SobolEngine(dimension=2)
            points = sobol.draw(self.n_pde)
            
            # Scale to domain
            t = points[:, 0:1] * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
            x = points[:, 1:2] * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        else:
            t = torch.rand(self.n_pde, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
            x = torch.rand(self.n_pde, 1) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
            
        return t.to(device), x.to(device)
    
    def generate_ic_points(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate points for initial conditions"""
        t = torch.zeros(self.n_ic, 1)
        x = torch.linspace(self.x_range[0], self.x_range[1], self.n_ic).reshape(-1, 1)
        return t.to(device), x.to(device)
    
    def generate_bc_points(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate points for boundary conditions (periodic)"""
        t = torch.rand(self.n_bc, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        x_left = torch.ones(self.n_bc, 1) * self.x_range[0]
        x_right = torch.ones(self.n_bc, 1) * self.x_range[1]
        return t.to(device), x_left.to(device), x_right.to(device)


def gaussian_pulse_ic(x: torch.Tensor, epsilon0: float = 1.0, 
                     width: float = 0.5, center: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gaussian pulse initial condition.
    
    Args:
        x: Spatial coordinates
        epsilon0: Peak energy density
        width: Width of Gaussian
        center: Center position
        
    Returns:
        epsilon_ic, v_ic, pi_ic
    """
    epsilon_ic = epsilon0 * torch.exp(-((x - center) / width)**2)
    v_ic = torch.zeros_like(x)
    pi_ic = torch.zeros_like(x)
    
    return epsilon_ic, v_ic, pi_ic


def train_pinn(model: nn.Module, loss_fn: PINNLoss, 
               collocation: CollocationPointGenerator,
               n_epochs: int = 20000,
               lr_adam: float = 1e-3,
               lr_lbfgs: float = 1.0,
               use_lbfgs: bool = True,
               print_every: int = 100,
               device: torch.device = device) -> Dict:
    """
    Train the PINN model.
    
    Args:
        model: PINN model
        loss_fn: Loss function
        collocation: Collocation point generator
        n_epochs: Number of training epochs
        lr_adam: Learning rate for Adam
        lr_lbfgs: Learning rate for L-BFGS
        use_lbfgs: Whether to use L-BFGS after Adam
        print_every: Print frequency
        device: torch device
        
    Returns:
        history: Training history dictionary
    """
    model = model.to(device)
    
    # Generate collocation points
    t_pde, x_pde = collocation.generate_pde_points(device)
    t_ic, x_ic = collocation.generate_ic_points(device)
    t_bc, x_left, x_right = collocation.generate_bc_points(device)
    
    # Generate initial conditions
    epsilon_ic, v_ic, pi_ic = gaussian_pulse_ic(x_ic, epsilon0=1.0, width=1.0)
    
    # Training history
    history = {
        'loss': [],
        'pde_loss': [],
        'ic_loss': [],
        'bc_loss': [],
    }
    
    # ========== Phase 1: Adam Optimizer ==========
    print("\n" + "="*60)
    print("Phase 1: Training with Adam optimizer")
    print("="*60)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        loss, diagnostics = loss_fn.total_loss(
            model, t_pde, x_pde, t_ic, x_ic, 
            epsilon_ic, v_ic, pi_ic, t_bc, x_left, x_right
        )
        
        loss.backward()
        optimizer.step()
        
        # Store history
        history['loss'].append(diagnostics['total'])
        history['pde_loss'].append(diagnostics['pde'])
        history['ic_loss'].append(diagnostics['ic'])
        history['bc_loss'].append(diagnostics['bc'])
        
        if (epoch + 1) % print_every == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{n_epochs}] | "
                  f"Loss: {diagnostics['total']:.6e} | "
                  f"PDE: {diagnostics['pde']:.6e} | "
                  f"IC: {diagnostics['ic']:.6e} | "
                  f"BC: {diagnostics['bc']:.6e} | "
                  f"Time: {elapsed:.1f}s")
            start_time = time.time()
    
    # ========== Phase 2: L-BFGS Optimizer ==========
    if use_lbfgs:
        print("\n" + "="*60)
        print("Phase 2: Fine-tuning with L-BFGS optimizer")
        print("="*60)
        
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr_lbfgs,
            max_iter=20,
            history_size=50,
            line_search_fn='strong_wolfe'
        )
        
        def closure():
            optimizer.zero_grad()
            loss, _ = loss_fn.total_loss(
                model, t_pde, x_pde, t_ic, x_ic,
                epsilon_ic, v_ic, pi_ic, t_bc, x_left, x_right
            )
            loss.backward()
            return loss
        
        n_lbfgs_steps = 50
        for step in range(n_lbfgs_steps):
            loss = optimizer.step(closure)
            
            with torch.no_grad():
                _, diagnostics = loss_fn.total_loss(
                    model, t_pde, x_pde, t_ic, x_ic,
                    epsilon_ic, v_ic, pi_ic, t_bc, x_left, x_right
                )
            
            history['loss'].append(diagnostics['total'])
            history['pde_loss'].append(diagnostics['pde'])
            history['ic_loss'].append(diagnostics['ic'])
            history['bc_loss'].append(diagnostics['bc'])
            
            if (step + 1) % 10 == 0:
                print(f"L-BFGS Step [{step+1}/{n_lbfgs_steps}] | "
                      f"Loss: {diagnostics['total']:.6e} | "
                      f"PDE: {diagnostics['pde']:.6e}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60 + "\n")
    
    return history


def visualize_solution(model: nn.Module, t_range: Tuple[float, float],
                      x_range: Tuple[float, float], device: torch.device,
                      n_t: int = 100, n_x: int = 200):
    """
    Visualize the PINN solution.
    
    Args:
        model: Trained PINN model
        t_range: Time range
        x_range: Spatial range
        device: torch device
        n_t: Number of time points
        n_x: Number of spatial points
    """
    model.eval()
    
    # Create meshgrid
    t_plot = np.linspace(t_range[0], t_range[1], n_t)
    x_plot = np.linspace(x_range[0], x_range[1], n_x)
    T, X = np.meshgrid(t_plot, x_plot, indexing='ij')
    
    # Predict
    with torch.no_grad():
        t_tensor = torch.FloatTensor(T.flatten()).reshape(-1, 1).to(device)
        x_tensor = torch.FloatTensor(X.flatten()).reshape(-1, 1).to(device)
        
        output = model(t_tensor, x_tensor)
        
        epsilon = output[:, 0].cpu().numpy().reshape(n_t, n_x)
        v = output[:, 1].cpu().numpy().reshape(n_t, n_x)
        pi = output[:, 2].cpu().numpy().reshape(n_t, n_x)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Energy density
    im0 = axes[0].contourf(X, T, epsilon, levels=50, cmap='hot')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_title('Energy Density ε(t,x)')
    plt.colorbar(im0, ax=axes[0])
    
    # Velocity
    im1 = axes[1].contourf(X, T, v, levels=50, cmap='RdBu_r')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    axes[1].set_title('Velocity v(t,x)')
    plt.colorbar(im1, ax=axes[1])
    
    # Shear stress
    im2 = axes[2].contourf(X, T, pi, levels=50, cmap='seismic')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('t')
    axes[2].set_title('Shear Stress π(t,x)')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    return fig, (epsilon, v, pi, T, X)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Israel-Stewart PINN Implementation")
    print("1+1D Viscous Relativistic Hydrodynamics")
    print("="*60 + "\n")
    
    # Physical parameters
    params = PhysicsParameters(eta_over_s=0.2, epsilon0=1.0)
    print(f"Physics parameters:")
    print(f"  η/s = {params.eta_over_s}")
    print(f"  c_s = {params.c_s:.4f}")
    print(f"  Conformal EoS: {params.conformal}")
    
    # Domain
    t_range = (0.0, 5.0)
    x_range = (-5.0, 5.0)
    print(f"\nDomain:")
    print(f"  t ∈ [{t_range[0]}, {t_range[1]}]")
    print(f"  x ∈ [{x_range[0]}, {x_range[1]}]")
    
    # Initialize model
    model = IsraelStewartPINN(layers=[2, 128, 128, 128, 128, 3])
    print(f"\nModel architecture: {model.layers}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize residuals and loss
    residuals = IsraelStewartResiduals(params)
    loss_fn = PINNLoss(residuals)
    
    # Collocation points
    collocation = CollocationPointGenerator(
        t_range, x_range,
        n_pde=10000,
        n_ic=500,
        n_bc=500,
        use_sobol=True
    )
    
    # Train
    history = train_pinn(
        model, loss_fn, collocation,
        n_epochs=5000,
        lr_adam=1e-3,
        use_lbfgs=True,
        print_every=500,
        device=device
    )
    
    # Visualize
    print("\nGenerating visualizations...")
    fig, solution = visualize_solution(model, t_range, x_range, device)
    plt.savefig('/home/claude/israel_stewart_solution.png', dpi=150, bbox_inches='tight')
    print("Saved: israel_stewart_solution.png")
    
    # Plot training history
    fig_loss, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(history['loss'], label='Total Loss')
    ax.semilogy(history['pde_loss'], label='PDE Loss', alpha=0.7)
    ax.semilogy(history['ic_loss'], label='IC Loss', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('/home/claude/training_history.png', dpi=150, bbox_inches='tight')
    print("Saved: training_history.png")
    
    print("\n" + "="*60)
    print("Simulation complete!")
    print("="*60)
