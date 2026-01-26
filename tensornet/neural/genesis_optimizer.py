"""
Genesis Optimizer
==================

Domain-aware optimizer for Tensor Train manifolds. Combines:
1. Riemannian Gradient Descent on Stiefel Manifold
2. Scale-Aware Learning Rates (macro > micro)
3. Adaptive λ scheduling for Discovery Sensitivity

Mathematical Foundation:
- TT cores live on product of Stiefel manifolds St(r_k, n_k)
- Riemannian gradient: proj_tangent(∇f) preserves orthogonality
- Retraction: QR-based or polar decomposition

References:
- Absil et al. (2008): Optimization Algorithms on Matrix Manifolds
- Lubich et al. (2015): Time Integration of Tensor Trains
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


@dataclass
class GenesisOptimizerConfig:
    """Configuration for Genesis Optimizer."""
    
    # Base learning rate
    lr: float = 1e-3
    
    # Scale-aware learning rate multipliers
    # Higher cores (macro) get higher LR, lower cores (micro) get lower
    macro_lr_multiplier: float = 2.0  # For first 25% of cores
    micro_lr_multiplier: float = 0.5  # For last 25% of cores
    
    # Riemannian optimization
    use_riemannian: bool = True
    retraction_type: str = 'qr'  # 'qr', 'polar', or 'cayley'
    
    # Momentum (on tangent space)
    momentum: float = 0.9
    use_nesterov: bool = True
    
    # Weight decay (Riemannian-aware)
    weight_decay: float = 0.0
    
    # Lambda scheduling for discovery sensitivity
    lambda_init: float = 1e-3
    lambda_min: float = 1e-5
    lambda_max: float = 1e-1
    lambda_schedule: str = 'adaptive'  # 'constant', 'decay', 'adaptive'
    
    # Gradient clipping
    max_grad_norm: float = 10.0
    
    # Numerical stability
    eps: float = 1e-8


class StiefelManifold:
    """
    Operations on the Stiefel manifold St(n, p) = {X ∈ R^{n×p} : X^T X = I_p}.
    
    For TT cores reshaped as (r_left * d_k, r_right), we project gradients
    to the tangent space and use retractions to stay on the manifold.
    """
    
    @staticmethod
    def project_tangent(X: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        Project Euclidean gradient G onto tangent space T_X St(n, p).
        
        The tangent space at X is:
        T_X St = {Z : X^T Z + Z^T X = 0}
        
        Projection formula:
        proj(G) = G - X @ sym(X^T @ G)
        
        where sym(A) = (A + A^T) / 2
        """
        XTG = X.T @ G
        sym_XTG = 0.5 * (XTG + XTG.T)
        return G - X @ sym_XTG
    
    @staticmethod
    def retract_qr(X: torch.Tensor, V: torch.Tensor, step: float = 1.0) -> torch.Tensor:
        """
        QR-based retraction: R_X(V) = qf(X + step * V)
        
        This is the most stable retraction for optimization.
        """
        Y = X + step * V
        Q, R = torch.linalg.qr(Y)
        # Ensure consistent orientation (positive diagonal of R)
        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1
        Q = Q * signs.unsqueeze(0)
        return Q
    
    @staticmethod
    def retract_polar(X: torch.Tensor, V: torch.Tensor, step: float = 1.0) -> torch.Tensor:
        """
        Polar retraction: R_X(V) = (X + step * V)(I + step^2 V^T V)^{-1/2}
        
        More accurate than QR but more expensive.
        """
        Y = X + step * V
        # Polar decomposition via SVD
        U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
        return U @ Vh
    
    @staticmethod
    def retract_cayley(X: torch.Tensor, V: torch.Tensor, step: float = 1.0) -> torch.Tensor:
        """
        Cayley retraction for skew-symmetric updates.
        
        W = V @ X^T - X @ V^T (skew-symmetric)
        R_X(V) = (I - step/2 W)^{-1} (I + step/2 W) X
        """
        n = X.shape[0]
        W = V @ X.T - X @ V.T
        I = torch.eye(n, device=X.device, dtype=X.dtype)
        
        # Cayley transform
        A = I - (step / 2) * W
        B = I + (step / 2) * W
        
        # Solve A @ Y = B @ X
        Y = torch.linalg.solve(A, B @ X)
        return Y


class GenesisOptimizer(Optimizer):
    """
    Riemannian optimizer for Tensor Train cores.
    
    Features:
    - Manifold-aware updates (Stiefel projection + retraction)
    - Scale-aware learning rates (macro cores learn faster)
    - Adaptive lambda scheduling for nuclear norm penalty
    - Momentum on tangent space
    
    Usage:
        optimizer = GenesisOptimizer(qtt.parameters(), config)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = compute_loss(qtt)
            loss.backward()
            optimizer.step()
            optimizer.update_lambda(loss.item())  # Adaptive lambda
    """
    
    def __init__(
        self,
        params: Iterator[nn.Parameter],
        config: Optional[GenesisOptimizerConfig] = None
    ):
        if config is None:
            config = GenesisOptimizerConfig()
        
        self.config = config
        self.manifold = StiefelManifold()
        
        # Current lambda for nuclear norm
        self._lambda = config.lambda_init
        self._loss_history: List[float] = []
        
        defaults = dict(
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            use_riemannian=config.use_riemannian,
            retraction_type=config.retraction_type,
        )
        
        super().__init__(params, defaults)
        
        # Assign scale-aware learning rates
        self._assign_scale_aware_lr()
    
    def _assign_scale_aware_lr(self) -> None:
        """Assign learning rates based on core position (macro vs micro)."""
        for group in self.param_groups:
            params = list(group['params'])
            n_params = len(params)
            
            if n_params == 0:
                continue
            
            # First 25% = macro (high LR), last 25% = micro (low LR)
            macro_cutoff = max(1, n_params // 4)
            micro_cutoff = n_params - max(1, n_params // 4)
            
            for i, p in enumerate(params):
                state = self.state[p]
                if i < macro_cutoff:
                    state['lr_multiplier'] = self.config.macro_lr_multiplier
                    state['core_type'] = 'macro'
                elif i >= micro_cutoff:
                    state['lr_multiplier'] = self.config.micro_lr_multiplier
                    state['core_type'] = 'micro'
                else:
                    state['lr_multiplier'] = 1.0
                    state['core_type'] = 'mid'
    
    @property
    def current_lambda(self) -> float:
        """Current nuclear norm penalty weight."""
        return self._lambda
    
    def update_lambda(self, current_loss: float) -> None:
        """
        Update lambda based on loss trajectory (adaptive schedule).
        
        Strategy:
        - If loss decreasing fast: increase lambda (push for simplicity)
        - If loss stagnating: decrease lambda (allow more complexity)
        - If loss increasing: significantly decrease lambda
        """
        if self.config.lambda_schedule == 'constant':
            return
        
        self._loss_history.append(current_loss)
        
        if len(self._loss_history) < 5:
            return
        
        # Compute loss trend over last 5 steps
        recent = self._loss_history[-5:]
        trend = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)
        
        if self.config.lambda_schedule == 'decay':
            # Simple exponential decay
            self._lambda *= 0.99
        
        elif self.config.lambda_schedule == 'adaptive':
            if trend < -0.1:
                # Loss decreasing fast: push for simplicity
                self._lambda = min(self._lambda * 1.1, self.config.lambda_max)
            elif trend > 0.05:
                # Loss increasing: allow more complexity
                self._lambda = max(self._lambda * 0.8, self.config.lambda_min)
            elif abs(trend) < 0.01:
                # Stagnating: slight decrease
                self._lambda = max(self._lambda * 0.95, self.config.lambda_min)
        
        # Keep history bounded
        if len(self._loss_history) > 100:
            self._loss_history = self._loss_history[-50:]
    
    def _get_retraction(self, retraction_type: str):
        """Get retraction function based on type."""
        if retraction_type == 'qr':
            return self.manifold.retract_qr
        elif retraction_type == 'polar':
            return self.manifold.retract_polar
        elif retraction_type == 'cayley':
            return self.manifold.retract_cayley
        else:
            raise ValueError(f"Unknown retraction type: {retraction_type}")
    
    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """
        Perform a single optimization step.
        
        For Riemannian optimization:
        1. Project gradient to tangent space
        2. Apply momentum (in tangent space)
        3. Retract to manifold
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            use_riemannian = group['use_riemannian']
            retraction_type = group['retraction_type']
            
            retract = self._get_retraction(retraction_type)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Get scale-aware learning rate
                lr_mult = state.get('lr_multiplier', 1.0)
                effective_lr = lr * lr_mult
                
                # Initialize state
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['step'] = 0
                
                state['step'] += 1
                buf = state['momentum_buffer']
                
                # Gradient clipping
                grad_norm = grad.norm()
                if grad_norm > self.config.max_grad_norm:
                    grad = grad * (self.config.max_grad_norm / (grad_norm + self.config.eps))
                
                # Weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                if use_riemannian and p.dim() == 3:
                    # TT core: use soft Riemannian constraint
                    # Instead of hard Stiefel projection, we use gradient projection
                    # plus periodic orthogonalization
                    r_left, d_k, r_right = p.shape
                    
                    # Unfold to (r_left * d_k, r_right)
                    X = p.reshape(r_left * d_k, r_right)
                    G = grad.reshape(r_left * d_k, r_right)
                    
                    # Only apply Riemannian for tall matrices
                    if X.shape[0] >= X.shape[1] and X.shape[1] > 1:
                        # Soft Riemannian: project gradient, Euclidean update
                        # This removes the component that would break orthogonality
                        G_tangent = self.manifold.project_tangent(X, G)
                        
                        # Use projected gradient but Euclidean step
                        # (no retraction - more stable for TT training)
                        buf_reshaped = buf.reshape(r_left * d_k, r_right)
                        if momentum != 0:
                            buf_reshaped.mul_(momentum).add_(G_tangent)
                            if self.config.use_nesterov:
                                G_tangent = G_tangent.add(buf_reshaped, alpha=momentum)
                            else:
                                G_tangent = buf_reshaped.clone()
                        
                        # Euclidean update with projected gradient
                        X_new = X - effective_lr * G_tangent
                        
                        # Periodic soft orthogonalization (every 10 steps)
                        # instead of hard retraction each step
                        if state['step'] % 10 == 0:
                            # Soft orthogonalization via QR + Polar blend
                            # 0.8 QR + 0.2 Polar for faster convergence
                            Q, R = torch.linalg.qr(X_new)
                            signs = torch.sign(torch.diag(R))
                            Q_qr = Q * signs.unsqueeze(0)
                            
                            U, S, Vh = torch.linalg.svd(X_new, full_matrices=False)
                            Q_polar = U @ Vh
                            
                            # Blend: 80% QR (stable), 20% Polar (scale-preserving)
                            X_ortho = 0.8 * Q_qr + 0.2 * Q_polar * S.mean()
                            # Blend: 80% current, 20% orthogonal
                            X_new = 0.8 * X_new + 0.2 * X_ortho
                        
                        p.copy_(X_new.reshape(r_left, d_k, r_right))
                    else:
                        # Standard Euclidean update for small matrices
                        if momentum != 0:
                            buf.mul_(momentum).add_(grad)
                            if self.config.use_nesterov:
                                grad = grad.add(buf, alpha=momentum)
                            else:
                                grad = buf.clone()
                        p.add_(grad, alpha=-effective_lr)
                else:
                    # Standard Euclidean update
                    if momentum != 0:
                        buf.mul_(momentum).add_(grad)
                        if self.config.use_nesterov:
                            grad = grad.add(buf, alpha=momentum)
                        else:
                            grad = buf.clone()
                    p.add_(grad, alpha=-effective_lr)
        
        return loss
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get optimizer diagnostics for logging."""
        diagnostics = {
            'current_lambda': self._lambda,
            'loss_trend': 0.0,
            'core_stats': [],
        }
        
        if len(self._loss_history) >= 5:
            recent = self._loss_history[-5:]
            diagnostics['loss_trend'] = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)
        
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                state = self.state.get(p, {})
                if p.grad is not None:
                    diagnostics['core_stats'].append({
                        'core_idx': i,
                        'core_type': state.get('core_type', 'unknown'),
                        'lr_multiplier': state.get('lr_multiplier', 1.0),
                        'grad_norm': p.grad.norm().item(),
                        'param_norm': p.norm().item(),
                    })
        
        return diagnostics


class DifferentiablePersistence(nn.Module):
    """
    Differentiable Persistent Homology via Persistence Landscapes.
    
    Traditional PH is non-differentiable because birth/death of features
    are discrete events. We use persistence landscapes to create a smooth,
    differentiable representation.
    
    A persistence landscape is a collection of piecewise linear functions
    that encode the persistence diagram in a functional form.
    
    Reference:
    - Bubenik (2015): Statistical Topological Data Analysis using Persistence Landscapes
    """
    
    def __init__(
        self,
        resolution: int = 100,
        num_landscapes: int = 5,
        max_dim: int = 1,
        sigma: float = 0.1,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.resolution = resolution
        self.num_landscapes = num_landscapes
        self.max_dim = max_dim
        self.sigma = sigma
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Grid for landscape evaluation
        self.register_buffer(
            'grid',
            torch.linspace(0, 1, resolution, device=self.device)
        )
    
    def compute_distance_matrix(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distance matrix (differentiable).
        
        Args:
            points: (N, D) tensor of points
            
        Returns:
            (N, N) distance matrix
        """
        # Squared Euclidean distance
        sq_dists = torch.cdist(points, points, p=2)
        return sq_dists
    
    def soft_rips_filtration(
        self,
        distances: torch.Tensor,
        num_thresholds: int = 50
    ) -> torch.Tensor:
        """
        Soft Rips filtration using sigmoid approximation.
        
        Instead of discrete 0/1 for edge inclusion, we use
        sigmoid(threshold - distance) for smooth transitions.
        
        Args:
            distances: (N, N) distance matrix
            num_thresholds: Number of filtration levels
            
        Returns:
            (num_thresholds, N, N) tensor of soft adjacency matrices
        """
        # Filtration thresholds
        max_dist = distances.max()
        thresholds = torch.linspace(0, max_dist, num_thresholds, device=self.device)
        
        # Soft inclusion: sigmoid((threshold - distance) / sigma)
        # Shape: (num_thresholds, N, N)
        soft_adj = torch.sigmoid(
            (thresholds.view(-1, 1, 1) - distances.unsqueeze(0)) / self.sigma
        )
        
        return soft_adj
    
    def soft_betti_numbers(
        self,
        soft_adj: torch.Tensor,
        dim: int = 0
    ) -> torch.Tensor:
        """
        Approximate Betti numbers from soft adjacency matrices.
        
        For dim=0: β_0 ≈ trace(exp(-L)) where L is graph Laplacian
        For dim=1: Uses spectral approximation
        
        Args:
            soft_adj: (num_thresholds, N, N) soft adjacency
            dim: Homology dimension
            
        Returns:
            (num_thresholds,) approximate Betti numbers
        """
        num_thresholds, n, _ = soft_adj.shape
        betti = torch.zeros(num_thresholds, device=self.device)
        
        for t in range(num_thresholds):
            A = soft_adj[t]
            
            # Degree matrix
            D = torch.diag(A.sum(dim=1))
            
            # Graph Laplacian
            L = D - A
            
            # Compute eigenvalues
            eigenvalues = torch.linalg.eigvalsh(L)
            
            if dim == 0:
                # β_0 = number of zero eigenvalues (connected components)
                # Soft approximation: sum of exp(-λ/σ)
                betti[t] = torch.exp(-eigenvalues / self.sigma).sum()
            elif dim == 1:
                # β_1 ≈ edges - vertices + components (Euler characteristic relation)
                num_edges = A.sum() / 2
                num_vertices = n
                beta_0 = torch.exp(-eigenvalues / self.sigma).sum()
                betti[t] = num_edges - num_vertices + beta_0
        
        return betti
    
    def persistence_landscape(
        self,
        birth_death: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute persistence landscape from birth-death pairs.
        
        For each pair (b, d), the tent function is:
        λ_k(t) = k-th largest value of min(t - b, d - t)
        
        Args:
            birth_death: (num_pairs, 2) tensor of (birth, death) pairs
            
        Returns:
            (num_landscapes, resolution) landscape functions
        """
        if birth_death.shape[0] == 0:
            return torch.zeros(self.num_landscapes, self.resolution, device=self.device)
        
        # Normalize to [0, 1]
        bd_min = birth_death.min()
        bd_max = birth_death.max()
        bd_range = bd_max - bd_min + 1e-8
        
        birth = (birth_death[:, 0] - bd_min) / bd_range
        death = (birth_death[:, 1] - bd_min) / bd_range
        
        # Evaluate tent functions at grid points
        # tent(t) = min(t - birth, death - t) if birth <= t <= death, else 0
        # Soft approximation using smooth min
        
        grid = self.grid.unsqueeze(0)  # (1, resolution)
        birth = birth.unsqueeze(1)  # (num_pairs, 1)
        death = death.unsqueeze(1)  # (num_pairs, 1)
        
        # Tent function values: (num_pairs, resolution)
        left = grid - birth
        right = death - grid
        
        # Soft min
        tent = torch.minimum(left, right)
        tent = torch.clamp(tent, min=0)
        
        # Sort to get k-th largest at each grid point
        # landscapes[k, t] = k-th largest tent value at grid point t
        tent_sorted, _ = torch.sort(tent, dim=0, descending=True)
        
        # Take top num_landscapes
        landscapes = tent_sorted[:self.num_landscapes]
        
        # Pad if fewer pairs than landscapes
        if landscapes.shape[0] < self.num_landscapes:
            padding = torch.zeros(
                self.num_landscapes - landscapes.shape[0],
                self.resolution,
                device=self.device
            )
            landscapes = torch.cat([landscapes, padding], dim=0)
        
        return landscapes
    
    def forward(
        self,
        points: torch.Tensor,
        return_diagnostics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Compute differentiable persistence representation.
        
        Args:
            points: (N, D) point cloud
            return_diagnostics: If True, return additional info
            
        Returns:
            Persistence landscape tensor (num_landscapes, resolution)
            Optionally: diagnostics dict
        """
        # Distance matrix
        distances = self.compute_distance_matrix(points)
        
        # Soft Rips filtration
        soft_adj = self.soft_rips_filtration(distances)
        
        # Approximate Betti numbers (these form a "birth-death" curve)
        betti_0 = self.soft_betti_numbers(soft_adj, dim=0)
        
        # Instead of extracting discrete birth/death pairs (non-differentiable),
        # directly use the Betti curve as a smooth persistence signature.
        # This maintains full gradient flow.
        
        # Betti-0 encodes connected component evolution
        # We create a multi-scale representation by:
        # 1. Using the raw Betti curve
        # 2. Adding smoothed derivatives at different scales
        
        # Interpolate to target resolution
        betti_interp = torch.nn.functional.interpolate(
            betti_0.unsqueeze(0).unsqueeze(0),
            size=self.resolution,
            mode='linear',
            align_corners=True
        ).squeeze()  # (resolution,)
        
        # Build multi-landscape representation
        landscapes = []
        
        # Landscape 0: normalized Betti curve
        betti_norm = betti_interp / (betti_interp.max() + 1e-8)
        landscapes.append(betti_norm)
        
        # Landscape 1+: smoothed derivatives at different scales
        for k in range(1, self.num_landscapes):
            kernel_size = min(2 * k + 1, self.resolution // 4)
            if kernel_size > 1:
                # Smooth with 1D convolution
                kernel = torch.ones(kernel_size, device=self.device) / kernel_size
                padded = torch.nn.functional.pad(betti_interp.unsqueeze(0).unsqueeze(0), 
                                                  (kernel_size//2, kernel_size//2), mode='replicate')
                smoothed = torch.nn.functional.conv1d(padded, kernel.view(1, 1, -1)).squeeze()
                
                # Derivative of smoothed curve
                deriv = smoothed[1:] - smoothed[:-1]
                deriv = torch.nn.functional.pad(deriv.unsqueeze(0), (0, 1), mode='replicate').squeeze()
                deriv_norm = deriv / (deriv.abs().max() + 1e-8)
                landscapes.append(deriv_norm)
            else:
                landscapes.append(betti_norm)
        
        landscape = torch.stack(landscapes, dim=0)  # (num_landscapes, resolution)
        
        if return_diagnostics:
            return landscape, {'betti_0': betti_0, 'method': 'betti_curve_differentiable'}
        return landscape


class GeometricRotorLearner(nn.Module):
    """
    Learn geometric transformations between market states using Rotors.
    
    In Geometric Algebra, a rotor R transforms a multivector X via:
    X' = R X R†
    
    We parameterize rotors as exponentials of bivectors:
    R = exp(B/2) where B is a bivector (antisymmetric)
    
    This ensures R R† = 1 (normalization) automatically.
    
    Reference:
    - Dorst et al. (2007): Geometric Algebra for Computer Science
    """
    
    def __init__(
        self,
        state_dim: int,
        num_bivectors: int = 6,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            state_dim: Dimension of market state vectors
            num_bivectors: Number of independent rotation planes
            device: Compute device
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.num_bivectors = num_bivectors
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Bivector coefficients (these parameterize the rotation)
        # For n dimensions, we have n(n-1)/2 independent bivectors
        max_bivectors = state_dim * (state_dim - 1) // 2
        actual_bivectors = min(num_bivectors, max_bivectors)
        
        # Learnable bivector coefficients
        self.bivector_coeffs = nn.Parameter(
            torch.zeros(actual_bivectors, device=self.device)
        )
        
        # Pre-compute bivector basis (antisymmetric matrices)
        self.register_buffer('bivector_basis', self._compute_bivector_basis())
    
    def _compute_bivector_basis(self) -> torch.Tensor:
        """
        Compute basis of bivectors as antisymmetric matrices.
        
        Each bivector e_i ∧ e_j corresponds to the antisymmetric matrix
        E_{ij} - E_{ji} where E_{ij} has 1 at position (i,j).
        
        Returns:
            (num_bivectors, state_dim, state_dim) tensor
        """
        n = self.state_dim
        basis = []
        
        for i in range(n):
            for j in range(i + 1, n):
                B = torch.zeros(n, n, device=self.device)
                B[i, j] = 1.0
                B[j, i] = -1.0
                basis.append(B)
                
                if len(basis) >= self.bivector_coeffs.shape[0]:
                    break
            if len(basis) >= self.bivector_coeffs.shape[0]:
                break
        
        return torch.stack(basis)
    
    def get_bivector(self) -> torch.Tensor:
        """
        Compute the current bivector from coefficients.
        
        B = Σ_k θ_k B_k
        
        Returns:
            (state_dim, state_dim) antisymmetric matrix
        """
        # Weighted sum of basis bivectors
        B = torch.einsum('k,kij->ij', self.bivector_coeffs, self.bivector_basis)
        return B
    
    def get_rotor_matrix(self) -> torch.Tensor:
        """
        Compute the rotor as a rotation matrix.
        
        R = exp(B/2) where B is antisymmetric
        
        For antisymmetric B, exp(B) is orthogonal (rotation).
        
        Returns:
            (state_dim, state_dim) orthogonal matrix
        """
        B = self.get_bivector()
        
        # Matrix exponential of antisymmetric matrix = rotation
        R = torch.matrix_exp(B / 2)
        
        return R
    
    def forward(
        self,
        state_a: torch.Tensor,
        state_b: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply rotor to transform state_a, optionally computing loss to state_b.
        
        Args:
            state_a: Source state (batch_size, state_dim)
            state_b: Target state (optional)
            
        Returns:
            Transformed state (batch_size, state_dim)
            If state_b provided: (transformed, loss)
        """
        R = self.get_rotor_matrix()
        
        # Apply rotation: state_a' = state_a @ R^T
        transformed = state_a @ R.T
        
        if state_b is not None:
            # Compute "geometric work" = squared distance after transformation
            loss = torch.nn.functional.mse_loss(transformed, state_b)
            return transformed, loss
        
        return transformed
    
    def compute_geometric_work(
        self,
        state_a: torch.Tensor,
        state_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the minimal geometric work to transform A to B.
        
        This is the squared Frobenius norm of log(R) where R is the
        optimal rotor mapping A to B.
        
        Args:
            state_a: Source state
            state_b: Target state
            
        Returns:
            Scalar geometric work
        """
        # Current rotor
        R = self.get_rotor_matrix()
        
        # Apply rotation
        transformed = state_a @ R.T
        
        # Reconstruction error
        recon_error = (transformed - state_b).pow(2).sum()
        
        # Regularization: prefer small rotations (geodesic on SO(n))
        # ||log(R)||_F = ||B/2||_F
        B = self.get_bivector()
        rotation_magnitude = (B / 2).pow(2).sum()
        
        return recon_error + 0.01 * rotation_magnitude
    
    def get_rotation_angle(self) -> torch.Tensor:
        """
        Get the total rotation angle (Frobenius norm of bivector).
        
        Returns:
            Scalar rotation angle in radians
        """
        B = self.get_bivector()
        return torch.sqrt((B / 2).pow(2).sum())


class TopologyAwareDiscoveryLoss(nn.Module):
    """
    Combined loss function that incorporates:
    1. Reconstruction loss (fit the data)
    2. Nuclear norm regularization (structural simplicity)
    3. Topological persistence loss (maintain invariants)
    4. Geometric rotor loss (smooth transformations)
    
    L_total = L_recon + λ_1 L_nuclear + λ_2 L_topology + λ_3 L_geometric
    """
    
    def __init__(
        self,
        lambda_nuclear: float = 1e-3,
        lambda_topology: float = 1e-4,
        lambda_geometric: float = 1e-4,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.lambda_nuclear = lambda_nuclear
        self.lambda_topology = lambda_topology
        self.lambda_geometric = lambda_geometric
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Import from differentiable_qtt
        from tensornet.neural.differentiable_qtt import NuclearNormRegularizer
        self.nuclear_reg = NuclearNormRegularizer(
            lambda_reg=1.0,  # We scale externally
            device=self.device
        )
        
        # Persistence module
        self.persistence = DifferentiablePersistence(device=self.device)
    
    def forward(
        self,
        recon_loss: torch.Tensor,
        cores: List[torch.Tensor],
        point_cloud: Optional[torch.Tensor] = None,
        target_landscape: Optional[torch.Tensor] = None,
        state_a: Optional[torch.Tensor] = None,
        state_b: Optional[torch.Tensor] = None,
        rotor_learner: Optional[GeometricRotorLearner] = None,
        return_breakdown: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute total discovery loss.
        
        Args:
            recon_loss: Base reconstruction loss
            cores: TT cores for nuclear norm
            point_cloud: Points for topology loss (optional)
            target_landscape: Target persistence landscape (optional)
            state_a, state_b: States for geometric loss (optional)
            rotor_learner: Rotor module for geometric loss (optional)
            return_breakdown: If True, return component losses
            
        Returns:
            Total loss, optionally with breakdown dict
        """
        total_loss = recon_loss
        breakdown = {'recon': recon_loss}
        
        # Nuclear norm regularization
        nuclear_loss = self.nuclear_reg(cores)
        total_loss = total_loss + self.lambda_nuclear * nuclear_loss
        breakdown['nuclear'] = nuclear_loss
        
        # Topology loss (if point cloud provided)
        if point_cloud is not None and target_landscape is not None:
            current_landscape = self.persistence(point_cloud)
            topology_loss = torch.nn.functional.mse_loss(
                current_landscape, target_landscape
            )
            total_loss = total_loss + self.lambda_topology * topology_loss
            breakdown['topology'] = topology_loss
        
        # Geometric rotor loss (if states and rotor provided)
        if state_a is not None and state_b is not None and rotor_learner is not None:
            geometric_loss = rotor_learner.compute_geometric_work(state_a, state_b)
            total_loss = total_loss + self.lambda_geometric * geometric_loss
            breakdown['geometric'] = geometric_loss
        
        if return_breakdown:
            return total_loss, breakdown
        return total_loss


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def train_with_genesis_optimizer(
    qtt_module: nn.Module,
    target: torch.Tensor,
    epochs: int = 100,
    config: Optional[GenesisOptimizerConfig] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train a differentiable QTT module using the Genesis Optimizer.
    
    Args:
        qtt_module: DifferentiableQTTCores or RankAdaptiveQTT module
        target: Target tensor to reconstruct
        epochs: Number of training epochs
        config: Optimizer configuration
        verbose: Print progress
        
    Returns:
        Training history dict
    """
    from tensornet.neural.differentiable_qtt import (
        compute_reconstruction_loss,
        NuclearNormRegularizer
    )
    
    if config is None:
        config = GenesisOptimizerConfig()
    
    optimizer = GenesisOptimizer(qtt_module.parameters(), config)
    nuclear_reg = NuclearNormRegularizer(lambda_reg=1.0, device=target.device)
    
    history = {
        'total_loss': [],
        'recon_loss': [],
        'nuclear_loss': [],
        'lambda': [],
    }
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get cores
        cores = qtt_module.get_cores()
        
        # Losses
        recon_loss = compute_reconstruction_loss(cores, target, loss_type='mse')
        nuclear_loss = nuclear_reg(cores)
        
        total_loss = recon_loss + optimizer.current_lambda * nuclear_loss
        
        # Backward
        total_loss.backward()
        optimizer.step()
        optimizer.update_lambda(total_loss.item())
        
        # Record
        history['total_loss'].append(total_loss.item())
        history['recon_loss'].append(recon_loss.item())
        history['nuclear_loss'].append(nuclear_loss.item())
        history['lambda'].append(optimizer.current_lambda)
        
        if verbose and epoch % (epochs // 10) == 0:
            print(f"  Epoch {epoch:4d}: Loss={total_loss.item():.6f}, "
                  f"Recon={recon_loss.item():.6f}, "
                  f"Nuclear={nuclear_loss.item():.6f}, "
                  f"λ={optimizer.current_lambda:.6f}")
    
    return history
