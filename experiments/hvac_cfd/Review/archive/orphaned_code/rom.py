"""
TigantiCFD Reduced Order Modeling
=================================

Proper Orthogonal Decomposition (POD) for real-time digital twin applications.

Capabilities:
- T4.01: POD basis extraction from CFD snapshots
- T4.02: ROM construction via Galerkin projection
- T4.03: Real-time prediction for digital twin
- T4.04: Error estimation and model validation

Reference:
    Holmes, P. et al. (2012). "Turbulence, Coherent Structures, 
    Dynamical Systems and Symmetry." Cambridge University Press.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class PODBasis:
    """POD basis functions and associated data."""
    modes: np.ndarray           # Shape: (n_modes, n_dof)
    singular_values: np.ndarray # Shape: (n_modes,)
    mean_field: np.ndarray      # Shape: (n_dof,)
    energy_content: np.ndarray  # Cumulative energy ratio
    n_modes: int
    n_snapshots: int
    reconstruction_error: float


@dataclass
class ROMState:
    """State of the reduced-order model."""
    coefficients: np.ndarray    # Modal coefficients
    full_field: np.ndarray      # Reconstructed full field
    residual: float             # Projection residual
    

class PODAnalysis:
    """
    Proper Orthogonal Decomposition for field data.
    
    Extracts the most energetic modes from a collection of CFD snapshots
    to create a low-dimensional basis for reduced-order modeling.
    
    The POD modes Φᵢ are computed as eigenvectors of the correlation matrix:
        C = (1/N) Σ uₙ uₙᵀ
    
    or equivalently via SVD:
        U = [u₁, u₂, ..., uₙ]
        U = Φ Σ Vᵀ
    """
    
    def __init__(self, energy_threshold: float = 0.99):
        """
        Initialize POD analysis.
        
        Args:
            energy_threshold: Keep modes capturing this fraction of total energy
        """
        self.energy_threshold = energy_threshold
        self.basis: Optional[PODBasis] = None
        
    def compute_basis(
        self,
        snapshots: np.ndarray,
        n_modes: Optional[int] = None
    ) -> PODBasis:
        """
        Compute POD basis from snapshot matrix.
        
        Args:
            snapshots: Array of shape (n_snapshots, n_dof)
            n_modes: Number of modes to keep (if None, use energy threshold)
            
        Returns:
            PODBasis with computed modes
        """
        n_snapshots, n_dof = snapshots.shape
        
        # 1. Compute mean and subtract
        mean_field = np.mean(snapshots, axis=0)
        fluctuations = snapshots - mean_field
        
        # 2. SVD (more numerically stable than eigenvalue decomposition)
        # For efficiency, compute "thin" SVD
        if n_snapshots < n_dof:
            # Snapshot method: more efficient for typical CFD
            U, S, Vt = np.linalg.svd(fluctuations.T, full_matrices=False)
            modes = U.T  # Shape: (n_modes, n_dof)
        else:
            U, S, Vt = np.linalg.svd(fluctuations, full_matrices=False)
            modes = Vt
        
        # 3. Compute energy content
        energy = S**2
        total_energy = np.sum(energy)
        cumulative_energy = np.cumsum(energy) / total_energy
        
        # 4. Determine number of modes
        if n_modes is None:
            n_modes = np.searchsorted(cumulative_energy, self.energy_threshold) + 1
            n_modes = min(n_modes, len(S))
        
        # 5. Truncate
        modes = modes[:n_modes]
        singular_values = S[:n_modes]
        energy_content = cumulative_energy[:n_modes]
        
        # 6. Compute reconstruction error
        if n_modes < len(S):
            reconstruction_error = np.sqrt(np.sum(S[n_modes:]**2) / total_energy)
        else:
            reconstruction_error = 0.0
        
        self.basis = PODBasis(
            modes=modes,
            singular_values=singular_values,
            mean_field=mean_field,
            energy_content=energy_content,
            n_modes=n_modes,
            n_snapshots=n_snapshots,
            reconstruction_error=reconstruction_error
        )
        
        return self.basis
    
    def project(self, field: np.ndarray) -> np.ndarray:
        """
        Project a full field onto the POD basis.
        
        Args:
            field: Full field of shape (n_dof,)
            
        Returns:
            Modal coefficients of shape (n_modes,)
        """
        if self.basis is None:
            raise ValueError("Must compute basis first")
        
        fluctuation = field - self.basis.mean_field
        coefficients = self.basis.modes @ fluctuation
        
        return coefficients
    
    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Reconstruct full field from modal coefficients.
        
        Args:
            coefficients: Modal coefficients of shape (n_modes,)
            
        Returns:
            Reconstructed field of shape (n_dof,)
        """
        if self.basis is None:
            raise ValueError("Must compute basis first")
        
        fluctuation = coefficients @ self.basis.modes
        field = fluctuation + self.basis.mean_field
        
        return field
    
    def reconstruction_error(self, field: np.ndarray) -> float:
        """Compute relative reconstruction error for a field."""
        coeffs = self.project(field)
        reconstructed = self.reconstruct(coeffs)
        
        error = np.linalg.norm(field - reconstructed)
        reference = np.linalg.norm(field - self.basis.mean_field)
        
        return error / (reference + 1e-10)


class GalerkinROM:
    """
    Galerkin Reduced-Order Model for CFD.
    
    Projects the Navier-Stokes equations onto the POD basis to obtain
    a system of ODEs for the modal coefficients:
    
        da/dt = A·a + a·B·a + f
    
    where:
        A = linear operator (viscous/pressure terms)
        B = quadratic operator (convection)
        f = forcing term
    """
    
    def __init__(
        self,
        pod_basis: PODBasis,
        viscosity: float = 1.5e-5
    ):
        self.basis = pod_basis
        self.nu = viscosity
        self.n_modes = pod_basis.n_modes
        
        # ROM operators (computed during training)
        self.A: Optional[np.ndarray] = None  # Linear: (n_modes, n_modes)
        self.B: Optional[np.ndarray] = None  # Quadratic: (n_modes, n_modes, n_modes)
        self.f: Optional[np.ndarray] = None  # Forcing: (n_modes,)
        
    def train(
        self,
        snapshots: np.ndarray,
        dt: float,
        regularization: float = 1e-6
    ) -> None:
        """
        Learn ROM operators from snapshot data.
        
        Uses data-driven approach: minimize ||da/dt - A·a - B·a·a - f||²
        
        Args:
            snapshots: Snapshot matrix (n_snapshots, n_dof)
            dt: Time step between snapshots
            regularization: Tikhonov regularization parameter
        """
        n_snap = len(snapshots)
        
        # Project snapshots onto POD basis
        coeffs = np.array([
            self.basis.modes @ (s - self.basis.mean_field)
            for s in snapshots
        ])
        
        # Compute time derivatives (central difference)
        da_dt = np.zeros_like(coeffs)
        da_dt[1:-1] = (coeffs[2:] - coeffs[:-2]) / (2 * dt)
        da_dt[0] = (coeffs[1] - coeffs[0]) / dt
        da_dt[-1] = (coeffs[-1] - coeffs[-2]) / dt
        
        # Build regression matrix for linear + constant terms
        # [a, 1] -> da/dt
        n_modes = self.n_modes
        
        # Simple linear regression for A and f
        # da/dt ≈ A·a + f
        X = np.hstack([coeffs, np.ones((n_snap, 1))])
        
        # Ridge regression
        XtX = X.T @ X + regularization * np.eye(n_modes + 1)
        XtY = X.T @ da_dt
        
        params = np.linalg.solve(XtX, XtY)
        
        self.A = params[:n_modes].T
        self.f = params[-1]
        
        # Quadratic term (simplified - set to zero for linear ROM)
        self.B = np.zeros((n_modes, n_modes, n_modes))
        
    def predict(
        self,
        a0: np.ndarray,
        dt: float,
        n_steps: int
    ) -> np.ndarray:
        """
        Predict modal coefficients forward in time.
        
        Args:
            a0: Initial modal coefficients
            dt: Time step
            n_steps: Number of steps to predict
            
        Returns:
            Trajectory of shape (n_steps+1, n_modes)
        """
        if self.A is None:
            raise ValueError("Must train ROM first")
        
        trajectory = np.zeros((n_steps + 1, self.n_modes))
        trajectory[0] = a0
        
        a = a0.copy()
        for i in range(n_steps):
            # RK4 integration
            k1 = self._rhs(a)
            k2 = self._rhs(a + 0.5 * dt * k1)
            k3 = self._rhs(a + 0.5 * dt * k2)
            k4 = self._rhs(a + dt * k3)
            
            a = a + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory[i + 1] = a
        
        return trajectory
    
    def _rhs(self, a: np.ndarray) -> np.ndarray:
        """Compute right-hand side of ROM ODE."""
        # Linear term
        rhs = self.A @ a + self.f
        
        # Quadratic term
        for i in range(self.n_modes):
            rhs[i] += a @ self.B[i] @ a
        
        return rhs
    
    def reconstruct_trajectory(
        self,
        trajectory: np.ndarray
    ) -> np.ndarray:
        """Reconstruct full field trajectory from modal coefficients."""
        n_steps = len(trajectory)
        n_dof = len(self.basis.mean_field)
        
        fields = np.zeros((n_steps, n_dof))
        for i, a in enumerate(trajectory):
            fields[i] = a @ self.basis.modes + self.basis.mean_field
        
        return fields


class DigitalTwinROM:
    """
    Real-time digital twin using ROM.
    
    Combines POD-Galerkin ROM with sensor data assimilation
    for real-time monitoring and prediction.
    """
    
    def __init__(
        self,
        rom: GalerkinROM,
        sensor_locations: np.ndarray,  # Shape: (n_sensors, 3)
        observation_operator: np.ndarray  # Shape: (n_sensors, n_dof)
    ):
        self.rom = rom
        self.H = observation_operator  # Maps full state to sensor readings
        
        # Project observation operator to modal space
        # H_r = H @ Φᵀ
        self.H_modal = self.H @ self.rom.basis.modes.T
        
        # Kalman filter state
        self.a_estimate: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None  # Error covariance
        
        # Noise parameters
        self.Q = 1e-4 * np.eye(rom.n_modes)  # Process noise
        self.R = 1e-2 * np.eye(len(sensor_locations))  # Observation noise
        
    def initialize(self, initial_field: np.ndarray) -> None:
        """Initialize digital twin state from full field."""
        fluctuation = initial_field - self.rom.basis.mean_field
        self.a_estimate = self.rom.basis.modes @ fluctuation
        self.P = 0.1 * np.eye(self.rom.n_modes)
        
    def update(
        self,
        sensor_readings: np.ndarray,
        dt: float
    ) -> ROMState:
        """
        Update digital twin with new sensor data.
        
        Uses Extended Kalman Filter for state estimation.
        
        Args:
            sensor_readings: Current sensor values
            dt: Time since last update
            
        Returns:
            Updated ROM state
        """
        if self.a_estimate is None:
            raise ValueError("Must initialize first")
        
        # Prediction step
        a_pred = self._predict_state(self.a_estimate, dt)
        
        # Compute Jacobian of dynamics
        F = np.eye(self.rom.n_modes) + dt * self.rom.A
        
        # Predicted covariance
        P_pred = F @ self.P @ F.T + self.Q
        
        # Update step (Kalman filter)
        # Innovation
        y_pred = self.H_modal @ a_pred + self.H @ self.rom.basis.mean_field
        innovation = sensor_readings - y_pred
        
        # Kalman gain
        S = self.H_modal @ P_pred @ self.H_modal.T + self.R
        K = P_pred @ self.H_modal.T @ np.linalg.inv(S)
        
        # Update state
        self.a_estimate = a_pred + K @ innovation
        self.P = (np.eye(self.rom.n_modes) - K @ self.H_modal) @ P_pred
        
        # Reconstruct full field
        full_field = (self.a_estimate @ self.rom.basis.modes + 
                     self.rom.basis.mean_field)
        
        # Compute residual
        residual = float(np.linalg.norm(innovation))
        
        return ROMState(
            coefficients=self.a_estimate.copy(),
            full_field=full_field,
            residual=residual
        )
    
    def _predict_state(self, a: np.ndarray, dt: float) -> np.ndarray:
        """Forward prediction using ROM."""
        # Simple Euler for real-time
        return a + dt * (self.rom.A @ a + self.rom.f)
    
    def predict_future(
        self,
        n_steps: int,
        dt: float
    ) -> np.ndarray:
        """
        Predict future states from current estimate.
        
        Returns:
            Full field trajectory of shape (n_steps+1, n_dof)
        """
        if self.a_estimate is None:
            raise ValueError("Must initialize first")
        
        trajectory = self.rom.predict(self.a_estimate, dt, n_steps)
        return self.rom.reconstruct_trajectory(trajectory)


def analyze_pod_basis(basis: PODBasis) -> str:
    """Generate analysis report for POD basis."""
    lines = [
        "=" * 60,
        "POD BASIS ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Number of snapshots: {basis.n_snapshots}",
        f"Number of modes retained: {basis.n_modes}",
        f"Reconstruction error: {basis.reconstruction_error:.4%}",
        "",
        "Mode Energy Content:",
        "-" * 40,
    ]
    
    for i in range(min(10, basis.n_modes)):
        individual = (basis.singular_values[i]**2 / 
                     np.sum(basis.singular_values**2) * 100)
        cumulative = basis.energy_content[i] * 100
        lines.append(f"  Mode {i+1:3d}: {individual:6.2f}% (cumulative: {cumulative:6.2f}%)")
    
    if basis.n_modes > 10:
        lines.append(f"  ... ({basis.n_modes - 10} more modes)")
    
    lines.extend([
        "",
        f"Energy captured by {basis.n_modes} modes: {basis.energy_content[-1]*100:.2f}%",
        "=" * 60,
    ])
    
    return "\n".join(lines)
