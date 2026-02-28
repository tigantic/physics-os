"""
Gaussian Process Regression with QTT Acceleration

Implements GP regression with efficient matrix operations.

Key equations:
- Prior: f ~ GP(m, k)
- Posterior mean: μ = K_* K^{-1} y
- Posterior variance: σ² = k(x_*, x_*) - K_* K^{-1} K_*^T
- Marginal likelihood: log p(y|X) = -½ y^T K^{-1} y - ½ log|K| - n/2 log(2π)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable, Union
import torch

from .kernels import Kernel, RBFKernel
from .kernel_matrix import QTTKernelMatrix, kernel_matrix


@dataclass
class GPPrior:
    """
    Gaussian Process prior specification.
    
    Defines mean function and covariance kernel.
    
    Attributes:
        kernel: Covariance kernel function
        mean_func: Mean function, default zero
    """
    kernel: Kernel
    mean_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    
    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate mean function."""
        if self.mean_func is None:
            return torch.zeros(x.shape[0])
        return self.mean_func(x)
    
    def covariance(self, x: torch.Tensor, 
                   y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate covariance kernel."""
        return self.kernel.matrix(x, y)
    
    def sample(self, x: torch.Tensor, 
               n_samples: int = 1,
               seed: Optional[int] = None) -> torch.Tensor:
        """
        Draw samples from the GP prior.
        
        Args:
            x: Input points, shape (n, d)
            n_samples: Number of samples
            seed: Random seed
            
        Returns:
            Samples, shape (n_samples, n)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        n = x.shape[0]
        mean = self.mean(x)
        cov = self.covariance(x)
        
        # Add jitter for numerical stability (larger for reliable Cholesky)
        cov = cov + 1e-4 * torch.eye(n)
        
        # Cholesky decomposition
        L = torch.linalg.cholesky(cov)
        
        # Sample from N(0, I) and transform
        z = torch.randn(n, n_samples)
        samples = mean.unsqueeze(1) + L @ z
        
        return samples.T


@dataclass
class GPPosterior:
    """
    Gaussian Process posterior after conditioning on data.
    
    Attributes:
        prior: Prior specification
        x_train: Training inputs
        y_train: Training outputs
        noise_variance: Observation noise σ²_n
        K_inv_y: Cached K^{-1} y
        L: Cached Cholesky factor of (K + σ²I)
    """
    prior: GPPrior
    x_train: torch.Tensor
    y_train: torch.Tensor
    noise_variance: float = 1e-6
    K_inv_y: Optional[torch.Tensor] = None
    L: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Precompute cached quantities."""
        if self.K_inv_y is None:
            self._precompute()
    
    def _precompute(self):
        """Compute and cache inverse and Cholesky."""
        n = self.x_train.shape[0]
        K = self.prior.covariance(self.x_train)
        K_noisy = K + self.noise_variance * torch.eye(n)
        
        # Add jitter for stability
        K_noisy = K_noisy + 1e-6 * torch.eye(n)
        
        # Cholesky decomposition
        self.L = torch.linalg.cholesky(K_noisy)
        
        # Solve for K^{-1} y (need 2D for solve_triangular)
        y_2d = self.y_train.unsqueeze(-1) if self.y_train.dim() == 1 else self.y_train
        alpha = torch.linalg.solve_triangular(self.L, y_2d, upper=False)
        K_inv_y_2d = torch.linalg.solve_triangular(self.L.T, alpha, upper=True)
        self.K_inv_y = K_inv_y_2d.squeeze(-1) if self.y_train.dim() == 1 else K_inv_y_2d
    
    def mean(self, x_star: torch.Tensor) -> torch.Tensor:
        """
        Posterior mean at test points.
        
        μ_* = m(x_*) + K_* K^{-1} (y - m(X))
        
        Args:
            x_star: Test points, shape (n_*, d)
            
        Returns:
            Posterior mean, shape (n_*,)
        """
        # Prior mean
        m_star = self.prior.mean(x_star)
        m_train = self.prior.mean(self.x_train)
        
        # Cross-covariance K(x_*, X)
        K_star = self.prior.covariance(x_star, self.x_train)
        
        # Posterior mean
        residual = self.y_train - m_train
        return m_star + K_star @ self.K_inv_y
    
    def variance(self, x_star: torch.Tensor) -> torch.Tensor:
        """
        Posterior variance at test points.
        
        σ²_* = k(x_*, x_*) - K_* K^{-1} K_*^T
        
        Args:
            x_star: Test points, shape (n_*, d)
            
        Returns:
            Posterior variance, shape (n_*,)
        """
        # Prior variance
        k_star = self.prior.kernel.diagonal(x_star)
        
        # Cross-covariance
        K_star = self.prior.covariance(x_star, self.x_train)
        
        # Solve L^{-1} K_*^T
        v = torch.linalg.solve_triangular(self.L, K_star.T, upper=False)
        
        # Posterior variance
        return k_star - (v ** 2).sum(dim=0)
    
    def predict(self, x_star: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full prediction with mean and variance.
        
        Args:
            x_star: Test points, shape (n_*, d)
            
        Returns:
            (mean, variance) tuple
        """
        return self.mean(x_star), self.variance(x_star)
    
    def sample(self, x_star: torch.Tensor,
               n_samples: int = 1,
               seed: Optional[int] = None) -> torch.Tensor:
        """
        Draw samples from the posterior.
        
        Args:
            x_star: Test points, shape (n_*, d)
            n_samples: Number of samples
            seed: Random seed
            
        Returns:
            Samples, shape (n_samples, n_*)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        mean = self.mean(x_star)
        var = self.variance(x_star)
        
        # Sample with correct variance
        z = torch.randn(n_samples, len(mean))
        return mean + torch.sqrt(torch.clamp(var, min=1e-10)) * z
    
    def log_marginal_likelihood(self) -> float:
        """
        Compute log marginal likelihood.
        
        log p(y|X) = -½ y^T K^{-1} y - ½ log|K| - n/2 log(2π)
        
        Returns:
            Log marginal likelihood
        """
        n = len(self.y_train)
        
        # Data fit term
        data_fit = -0.5 * (self.y_train @ self.K_inv_y).item()
        
        # Complexity term (log determinant)
        log_det = 2 * torch.log(torch.diag(self.L)).sum().item()
        complexity = -0.5 * log_det
        
        # Normalization
        normalization = -0.5 * n * math.log(2 * math.pi)
        
        return data_fit + complexity + normalization


class GPRegressor:
    """
    Gaussian Process Regression model.
    
    Convenience class wrapping prior and posterior.
    """
    
    def __init__(self, 
                 kernel: Optional[Kernel] = None,
                 noise_variance: float = 1e-6,
                 mean_func: Optional[Callable] = None):
        """
        Initialize GP regressor.
        
        Args:
            kernel: Covariance kernel (default RBF)
            noise_variance: Observation noise
            mean_func: Prior mean function
        """
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.noise_variance = noise_variance
        self.mean_func = mean_func
        self.posterior: Optional[GPPosterior] = None
    
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> 'GPRegressor':
        """
        Fit GP to training data.
        
        Args:
            x: Training inputs, shape (n, d)
            y: Training outputs, shape (n,)
            
        Returns:
            Self
        """
        prior = GPPrior(self.kernel, self.mean_func)
        self.posterior = GPPosterior(
            prior=prior,
            x_train=x,
            y_train=y,
            noise_variance=self.noise_variance
        )
        return self
    
    def predict(self, x: torch.Tensor, 
                return_std: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Make predictions at test points.
        
        Args:
            x: Test points, shape (n_*, d)
            return_std: Whether to return standard deviation
            
        Returns:
            Mean, or (mean, std) if return_std=True
        """
        if self.posterior is None:
            raise ValueError("Must call fit() before predict()")
        
        mean, var = self.posterior.predict(x)
        
        if return_std:
            return mean, torch.sqrt(torch.clamp(var, min=0))
        return mean
    
    def sample(self, x: torch.Tensor, 
               n_samples: int = 1) -> torch.Tensor:
        """Draw posterior samples."""
        if self.posterior is None:
            raise ValueError("Must call fit() before sample()")
        return self.posterior.sample(x, n_samples)
    
    def score(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute R² score on test data.
        
        Args:
            x: Test inputs
            y: Test outputs
            
        Returns:
            R² score
        """
        y_pred = self.predict(x)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - (ss_res / ss_tot).item()


def gp_predict(x_train: torch.Tensor,
               y_train: torch.Tensor,
               x_test: torch.Tensor,
               kernel: Kernel,
               noise_variance: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Functional GP prediction.
    
    Args:
        x_train: Training inputs
        y_train: Training outputs
        x_test: Test inputs
        kernel: Covariance kernel
        noise_variance: Observation noise
        
    Returns:
        (mean, variance) at test points
    """
    prior = GPPrior(kernel)
    posterior = GPPosterior(prior, x_train, y_train, noise_variance)
    return posterior.predict(x_test)


def gp_posterior_sample(posterior: GPPosterior,
                        x_test: torch.Tensor,
                        n_samples: int = 10) -> torch.Tensor:
    """
    Draw posterior samples at test locations.
    
    For accurate function samples (not just pointwise),
    this uses the full posterior covariance.
    
    Args:
        posterior: GP posterior
        x_test: Test locations
        n_samples: Number of samples
        
    Returns:
        Function samples, shape (n_samples, n_test)
    """
    n_test = x_test.shape[0]
    
    # Mean and covariance
    mean = posterior.mean(x_test)
    
    # Full covariance K_** - K_* K^{-1} K_*^T
    K_star_star = posterior.prior.covariance(x_test)
    K_star = posterior.prior.covariance(x_test, posterior.x_train)
    
    v = torch.linalg.solve_triangular(posterior.L, K_star.T, upper=False)
    cov = K_star_star - v.T @ v
    
    # Add jitter
    cov = cov + 1e-6 * torch.eye(n_test)
    
    # Sample
    L = torch.linalg.cholesky(cov)
    z = torch.randn(n_test, n_samples)
    
    samples = mean.unsqueeze(1) + L @ z
    return samples.T


def gp_marginal_likelihood(x: torch.Tensor,
                           y: torch.Tensor,
                           kernel: Kernel,
                           noise_variance: float = 1e-6) -> float:
    """
    Compute log marginal likelihood for hyperparameter optimization.
    
    Args:
        x: Training inputs
        y: Training outputs
        kernel: Covariance kernel
        noise_variance: Observation noise
        
    Returns:
        Log marginal likelihood
    """
    prior = GPPrior(kernel)
    posterior = GPPosterior(prior, x, y, noise_variance)
    return posterior.log_marginal_likelihood()


class SparseGP:
    """
    Sparse Gaussian Process using inducing points.
    
    Reduces complexity from O(n³) to O(nm²) where m << n.
    """
    
    def __init__(self,
                 kernel: Kernel,
                 n_inducing: int = 100,
                 noise_variance: float = 1e-6):
        """
        Initialize sparse GP.
        
        Args:
            kernel: Covariance kernel
            n_inducing: Number of inducing points
            noise_variance: Observation noise
        """
        self.kernel = kernel
        self.n_inducing = n_inducing
        self.noise_variance = noise_variance
        
        self.inducing_points: Optional[torch.Tensor] = None
        self.alpha: Optional[torch.Tensor] = None
        self.L_mm: Optional[torch.Tensor] = None
    
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> 'SparseGP':
        """
        Fit sparse GP using FITC approximation.
        
        Args:
            x: Training inputs
            y: Training outputs
            
        Returns:
            Self
        """
        n = x.shape[0]
        m = min(self.n_inducing, n)
        
        # Select inducing points (random subset)
        indices = torch.randperm(n)[:m]
        self.inducing_points = x[indices]
        
        # Compute kernel matrices
        K_mm = self.kernel.matrix(self.inducing_points)
        K_nm = self.kernel.matrix(x, self.inducing_points)
        k_diag = self.kernel.diagonal(x)
        
        # Add larger jitter for stability
        K_mm = K_mm + 1e-4 * torch.eye(m)
        
        # FITC: approximate K_nn with Q_nn + diag(K_nn - Q_nn)
        self.L_mm = torch.linalg.cholesky(K_mm)
        
        # Solve L_mm^{-1} K_mn
        V = torch.linalg.solve_triangular(self.L_mm, K_nm.T, upper=False)
        
        # Lambda = diag(k_ii - q_ii) + σ²
        q_diag = (V ** 2).sum(dim=0)
        Lambda = k_diag - q_diag + self.noise_variance
        Lambda = torch.clamp(Lambda, min=1e-10)
        
        # Compute sigma = K_mm + K_mn Λ^{-1} K_nm
        Sigma = K_mm + K_nm.T @ (K_nm / Lambda.unsqueeze(1))
        Sigma = Sigma + 1e-4 * torch.eye(m)
        
        L_sigma = torch.linalg.cholesky(Sigma)
        
        # Compute alpha (need 2D for solve_triangular)
        rhs = K_nm.T @ (y / Lambda)
        rhs_2d = rhs.unsqueeze(-1) if rhs.dim() == 1 else rhs
        beta = torch.linalg.solve_triangular(L_sigma, rhs_2d, upper=False)
        alpha_2d = torch.linalg.solve_triangular(L_sigma.T, beta, upper=True)
        self.alpha = alpha_2d.squeeze(-1) if rhs.dim() == 1 else alpha_2d
        
        self.L_sigma = L_sigma
        
        return self
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions at test points.
        
        Args:
            x: Test points
            
        Returns:
            (mean, variance)
        """
        if self.inducing_points is None:
            raise ValueError("Must call fit() first")
        
        K_xm = self.kernel.matrix(x, self.inducing_points)
        k_diag = self.kernel.diagonal(x)
        
        # Mean
        mean = K_xm @ self.alpha
        
        # Variance
        V = torch.linalg.solve_triangular(self.L_mm, K_xm.T, upper=False)
        q_diag = (V ** 2).sum(dim=0)
        
        W = torch.linalg.solve_triangular(self.L_sigma, K_xm.T, upper=False)
        var_reduction = (W ** 2).sum(dim=0)
        
        var = k_diag - q_diag + var_reduction
        var = torch.clamp(var, min=0)
        
        return mean, var
