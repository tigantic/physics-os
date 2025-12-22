"""
FADS (Flush Air Data System) Sensor Model
==========================================

Phase 22: Sensor modeling for Aero-TRN navigation.

Flush Air Data Systems measure surface pressures on hypersonic vehicles
to infer atmospheric flight parameters (Mach, angle of attack, sideslip).

Key Features:
- Pressure port modeling with location-dependent measurements
- Noise models for sensor degradation
- Jacobian computation for navigation filtering
- Integration with CFD solutions

References:
    - Whitmore & Moes, "FADS System Design Guidelines" NASA TM-2000-209012
    - Karlgaard et al., "MSL Entry Reconstruction" NASA TM-2014-218314
    - Ingoldby et al., "X-15 FADS Analysis" NASA TN D-3331

Constitution Compliance: Article II.1, Article V
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, NamedTuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import math


# =============================================================================
# Constants
# =============================================================================

# Standard atmosphere
R_GAS = 287.05  # J/(kg·K) for air
GAMMA_AIR = 1.4


# =============================================================================
# Data Structures
# =============================================================================

class NoiseModel(Enum):
    """Sensor noise model types."""
    NONE = 'none'
    GAUSSIAN = 'gaussian'
    BIAS_DRIFT = 'bias_drift'
    QUANTIZATION = 'quantization'
    COMBINED = 'combined'


@dataclass
class PressurePort:
    """
    Single pressure measurement port.
    
    Attributes:
        id: Unique identifier
        position: (x, y, z) position on vehicle surface (m)
        normal: (nx, ny, nz) surface normal at port
        port_type: 'static' or 'pitot' (total pressure)
        calibration_factor: Multiplicative calibration
        bias: Additive bias (Pa)
    """
    id: str
    position: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    port_type: str = 'static'
    calibration_factor: float = 1.0
    bias: float = 0.0


@dataclass
class SensorNoiseConfig:
    """
    Configuration for sensor noise modeling.
    
    Attributes:
        model: Type of noise model
        std_dev_Pa: Gaussian noise standard deviation (Pa)
        bias_Pa: Systematic bias (Pa)
        drift_rate_Pa_s: Bias drift rate (Pa/s)
        quantization_Pa: ADC quantization level (Pa)
        temperature_coeff: Temperature sensitivity coefficient
    """
    model: NoiseModel = NoiseModel.GAUSSIAN
    std_dev_Pa: float = 100.0
    bias_Pa: float = 0.0
    drift_rate_Pa_s: float = 0.0
    quantization_Pa: float = 10.0
    temperature_coeff: float = 0.0


@dataclass
class FADSMeasurement:
    """
    Output from FADS sensor measurement.
    
    Attributes:
        port_ids: List of port identifiers
        pressures_Pa: Measured pressures (Pa)
        timestamps: Measurement timestamps (s)
        validity: Boolean flags for valid measurements
        temperature_K: Sensor temperatures (K)
    """
    port_ids: List[str]
    pressures_Pa: Tensor
    timestamps: Tensor
    validity: Tensor
    temperature_K: Optional[Tensor] = None


@dataclass
class FlightState:
    """
    Estimated flight state from FADS.
    
    Attributes:
        mach: Mach number
        alpha_deg: Angle of attack (degrees)
        beta_deg: Sideslip angle (degrees)
        dynamic_pressure_Pa: Dynamic pressure q∞ (Pa)
        static_pressure_Pa: Freestream static pressure (Pa)
        total_pressure_Pa: Total pressure (Pa)
        covariance: Uncertainty covariance matrix
    """
    mach: float
    alpha_deg: float
    beta_deg: float
    dynamic_pressure_Pa: float
    static_pressure_Pa: float
    total_pressure_Pa: float
    covariance: Optional[Tensor] = None


# =============================================================================
# Pressure Models
# =============================================================================

def modified_newtonian_cp(
    theta: Tensor,
    gamma: float = 1.4
) -> Tensor:
    """
    Modified Newtonian pressure coefficient.
    
    Cp = Cp_max * cos²(θ)
    
    where θ is the angle between surface normal and flow direction,
    and Cp_max ≈ 2 for hypersonic flow.
    
    At stagnation (θ = 0, flow hits normal head-on): Cp = Cp_max
    At tangent (θ = 90°, flow parallel to surface): Cp = 0
    
    Args:
        theta: Angle between flow direction and surface normal (radians)
        gamma: Ratio of specific heats
        
    Returns:
        Pressure coefficient Cp
    """
    # Hypersonic limit Cp_max
    Cp_max = 2.0
    
    # Modified Newtonian: Cp = Cp_max * cos²(θ)
    # θ = 0 means flow hits normal head-on -> maximum pressure
    Cp = Cp_max * torch.cos(theta)**2
    
    # Clamp to physical limits
    Cp = torch.clamp(Cp, min=0.0, max=Cp_max)
    
    return Cp


def tangent_cone_cp(
    mach: float,
    cone_half_angle: float,
    gamma: float = 1.4
) -> float:
    """
    Tangent-cone pressure coefficient for slender bodies.
    
    Uses Taylor-Maccoll solution approximation.
    
    Args:
        mach: Freestream Mach number
        cone_half_angle: Local cone half-angle (radians)
        gamma: Ratio of specific heats
        
    Returns:
        Pressure coefficient Cp
    """
    if mach < 1.0:
        return 0.0
    
    # Hypersonic small-angle approximation
    # Cp ≈ 2 * θ² for M*θ > 0.5
    theta = cone_half_angle
    
    if mach * theta > 0.5:
        return 2 * theta**2
    else:
        # Linear theory for smaller angles
        return 2 * theta / math.sqrt(mach**2 - 1)


def compute_surface_pressure(
    mach: float,
    alpha_rad: float,
    beta_rad: float,
    port_position: Tuple[float, float, float],
    port_normal: Tuple[float, float, float],
    p_inf: float,
    gamma: float = 1.4
) -> float:
    """
    Compute surface pressure at a port location.
    
    Uses modified Newtonian theory.
    
    Args:
        mach: Freestream Mach number
        alpha_rad: Angle of attack (radians)
        beta_rad: Sideslip angle (radians)
        port_position: Port (x, y, z) position
        port_normal: Port surface normal (nx, ny, nz) - OUTWARD pointing
        p_inf: Freestream pressure (Pa)
        gamma: Ratio of specific heats
        
    Returns:
        Surface pressure (Pa)
    """
    # Freestream velocity direction (flow comes from -x direction toward +x)
    # At zero alpha/beta, flow comes from negative x-axis
    # With alpha, nose pitches up (flow appears to come from below)
    # V_freestream = [-cos(α)cos(β), -sin(β), -sin(α)cos(β)]
    v_x = -math.cos(alpha_rad) * math.cos(beta_rad)
    v_y = -math.sin(beta_rad)
    v_z = -math.sin(alpha_rad) * math.cos(beta_rad)
    
    # Surface outward normal
    nx, ny, nz = port_normal
    
    # Cos of angle between flow direction and outward normal
    # Windward: flow opposes normal -> V · n < 0 -> cos_incidence > 0
    cos_incidence = -(v_x * nx + v_y * ny + v_z * nz)
    cos_incidence = max(-1.0, min(1.0, cos_incidence))
    
    if cos_incidence > 0:  # Windward side (flow hitting surface)
        # Incidence angle for Newtonian theory
        # At head-on impact: cos_incidence = 1, theta = 0
        theta = math.acos(cos_incidence)
        Cp = modified_newtonian_cp(torch.tensor(theta), gamma).item()
    else:  # Leeward side
        Cp = -0.1  # Small suction on leeward
    
    # Dynamic pressure
    q_inf = 0.5 * gamma * p_inf * mach**2
    
    # Surface pressure
    p_surface = p_inf + Cp * q_inf
    
    return max(p_surface, 0.01 * p_inf)  # Minimum pressure


# =============================================================================
# FADS Sensor Class
# =============================================================================

class FADSSensor:
    """
    Flush Air Data System sensor model.
    
    Models an array of pressure ports on a hypersonic vehicle for
    measuring flight conditions during atmospheric entry.
    
    Attributes:
        ports: List of pressure port configurations
        noise_config: Sensor noise model configuration
        calibration: Calibration data
    """
    
    def __init__(
        self,
        ports: List[PressurePort],
        noise_config: Optional[SensorNoiseConfig] = None,
        gamma: float = 1.4
    ):
        """
        Initialize FADS sensor.
        
        Args:
            ports: List of pressure port configurations
            noise_config: Noise model configuration
            gamma: Ratio of specific heats
        """
        self.ports = ports
        self.noise_config = noise_config or SensorNoiseConfig()
        self.gamma = gamma
        
        self._port_by_id = {p.id: p for p in ports}
        self._bias_drift = torch.zeros(len(ports), dtype=torch.float64)
        self._time = 0.0
    
    @classmethod
    def typical_nose_array(cls) -> 'FADSSensor':
        """Create typical nose-mounted FADS array."""
        ports = [
            # Center port (stagnation)
            PressurePort('center', (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            # Upper arc
            PressurePort('upper_1', (0.02, 0.0, 0.05), (0.9, 0.0, 0.44)),
            PressurePort('upper_2', (0.05, 0.0, 0.10), (0.7, 0.0, 0.71)),
            # Lower arc
            PressurePort('lower_1', (0.02, 0.0, -0.05), (0.9, 0.0, -0.44)),
            PressurePort('lower_2', (0.05, 0.0, -0.10), (0.7, 0.0, -0.71)),
            # Side ports
            PressurePort('left', (0.02, -0.05, 0.0), (0.9, -0.44, 0.0)),
            PressurePort('right', (0.02, 0.05, 0.0), (0.9, 0.44, 0.0)),
        ]
        return cls(ports)
    
    def measure(
        self,
        mach: float,
        alpha_deg: float,
        beta_deg: float,
        p_inf: float,
        timestamp: float = 0.0
    ) -> FADSMeasurement:
        """
        Measure pressures for given flight conditions.
        
        Args:
            mach: Freestream Mach number
            alpha_deg: Angle of attack (degrees)
            beta_deg: Sideslip angle (degrees)
            p_inf: Freestream static pressure (Pa)
            timestamp: Measurement timestamp (s)
            
        Returns:
            FADSMeasurement with pressure readings
        """
        alpha_rad = math.radians(alpha_deg)
        beta_rad = math.radians(beta_deg)
        
        n_ports = len(self.ports)
        pressures = torch.zeros(n_ports, dtype=torch.float64)
        validity = torch.ones(n_ports, dtype=torch.bool)
        
        for i, port in enumerate(self.ports):
            # True pressure
            p_true = compute_surface_pressure(
                mach, alpha_rad, beta_rad,
                port.position, port.normal,
                p_inf, self.gamma
            )
            
            # Apply calibration
            p_calibrated = p_true * port.calibration_factor + port.bias
            
            # Apply noise
            p_noisy = self._apply_noise(p_calibrated, i, timestamp)
            
            pressures[i] = p_noisy
        
        self._time = timestamp
        
        return FADSMeasurement(
            port_ids=[p.id for p in self.ports],
            pressures_Pa=pressures,
            timestamps=torch.tensor([timestamp] * n_ports, dtype=torch.float64),
            validity=validity
        )
    
    def measure_from_field(
        self,
        pressure_field: Tensor,
        field_positions: Tensor,
        timestamp: float = 0.0
    ) -> FADSMeasurement:
        """
        Measure pressures by interpolating from CFD pressure field.
        
        Args:
            pressure_field: Pressure values at grid points (Pa)
            field_positions: (N, 3) grid point positions
            timestamp: Measurement timestamp (s)
            
        Returns:
            FADSMeasurement with pressure readings
        """
        n_ports = len(self.ports)
        pressures = torch.zeros(n_ports, dtype=torch.float64)
        validity = torch.ones(n_ports, dtype=torch.bool)
        
        for i, port in enumerate(self.ports):
            pos = torch.tensor(port.position, dtype=torch.float64)
            
            # Nearest neighbor interpolation
            distances = torch.norm(field_positions - pos.unsqueeze(0), dim=1)
            nearest_idx = torch.argmin(distances)
            
            p_true = pressure_field.flatten()[nearest_idx].item()
            
            # Apply calibration and noise
            p_calibrated = p_true * port.calibration_factor + port.bias
            p_noisy = self._apply_noise(p_calibrated, i, timestamp)
            
            pressures[i] = p_noisy
        
        self._time = timestamp
        
        return FADSMeasurement(
            port_ids=[p.id for p in self.ports],
            pressures_Pa=pressures,
            timestamps=torch.tensor([timestamp] * n_ports, dtype=torch.float64),
            validity=validity
        )
    
    def _apply_noise(
        self,
        pressure: float,
        port_idx: int,
        timestamp: float
    ) -> float:
        """Apply noise model to pressure reading."""
        p = pressure
        
        if self.noise_config.model == NoiseModel.NONE:
            return p
        
        # Gaussian noise
        if self.noise_config.model in [NoiseModel.GAUSSIAN, NoiseModel.COMBINED]:
            p += torch.randn(1).item() * self.noise_config.std_dev_Pa
        
        # Bias and drift
        if self.noise_config.model in [NoiseModel.BIAS_DRIFT, NoiseModel.COMBINED]:
            dt = timestamp - self._time
            self._bias_drift[port_idx] += self.noise_config.drift_rate_Pa_s * dt
            p += self.noise_config.bias_Pa + self._bias_drift[port_idx].item()
        
        # Quantization
        if self.noise_config.model in [NoiseModel.QUANTIZATION, NoiseModel.COMBINED]:
            q = self.noise_config.quantization_Pa
            p = round(p / q) * q
        
        return p
    
    def jacobian(
        self,
        mach: float,
        alpha_deg: float,
        beta_deg: float,
        p_inf: float,
        delta: float = 0.01
    ) -> Tensor:
        """
        Compute Jacobian of pressure measurements w.r.t. flight state.
        
        ∂p/∂[M, α, β, p_inf] via finite differences.
        
        Args:
            mach: Mach number
            alpha_deg: Angle of attack (degrees)
            beta_deg: Sideslip angle (degrees)
            p_inf: Freestream pressure (Pa)
            delta: Finite difference step
            
        Returns:
            Jacobian matrix (n_ports, 4)
        """
        n_ports = len(self.ports)
        J = torch.zeros(n_ports, 4, dtype=torch.float64)
        
        # Baseline measurement
        meas_0 = self.measure(mach, alpha_deg, beta_deg, p_inf)
        p_0 = meas_0.pressures_Pa
        
        # Perturb each state variable
        states = [
            (mach + delta, alpha_deg, beta_deg, p_inf),
            (mach, alpha_deg + delta, beta_deg, p_inf),
            (mach, alpha_deg, beta_deg + delta, p_inf),
            (mach, alpha_deg, beta_deg, p_inf * (1 + delta)),
        ]
        
        deltas = [delta, delta, delta, p_inf * delta]
        
        for j, (M, a, b, p) in enumerate(states):
            meas = self.measure(M, a, b, p)
            J[:, j] = (meas.pressures_Pa - p_0) / deltas[j]
        
        return J
    
    def estimate_state(
        self,
        measurement: FADSMeasurement,
        p_inf_estimate: float,
        mach_initial: float = 5.0,
        alpha_initial: float = 0.0,
        beta_initial: float = 0.0,
        max_iterations: int = 10
    ) -> FlightState:
        """
        Estimate flight state from pressure measurements.
        
        Uses iterative least-squares with Jacobian.
        
        Args:
            measurement: FADS pressure measurements
            p_inf_estimate: Initial estimate of freestream pressure
            mach_initial: Initial Mach estimate
            alpha_initial: Initial alpha estimate (degrees)
            beta_initial: Initial beta estimate (degrees)
            max_iterations: Maximum iterations
            
        Returns:
            Estimated flight state
        """
        # State vector: [M, α, β]
        x = torch.tensor([mach_initial, alpha_initial, beta_initial], dtype=torch.float64)
        p_inf = p_inf_estimate
        
        p_measured = measurement.pressures_Pa
        
        for _ in range(max_iterations):
            # Predicted pressures
            meas_pred = self.measure(x[0].item(), x[1].item(), x[2].item(), p_inf)
            p_pred = meas_pred.pressures_Pa
            
            # Residual
            residual = p_measured - p_pred
            
            # Jacobian (just first 3 columns for M, α, β)
            J = self.jacobian(x[0].item(), x[1].item(), x[2].item(), p_inf)[:, :3]
            
            # Gauss-Newton update
            try:
                dx = torch.linalg.lstsq(J, residual.unsqueeze(1)).solution.squeeze()
            except:
                break
            
            x = x + 0.5 * dx  # Damped update
            
            # Convergence check
            if torch.norm(dx) < 1e-6:
                break
        
        # Compute derived quantities
        mach = x[0].item()
        q_inf = 0.5 * self.gamma * p_inf * mach**2
        p_total = p_inf * (1 + 0.5 * (self.gamma - 1) * mach**2)**(self.gamma / (self.gamma - 1))
        
        return FlightState(
            mach=mach,
            alpha_deg=x[1].item(),
            beta_deg=x[2].item(),
            dynamic_pressure_Pa=q_inf,
            static_pressure_Pa=p_inf,
            total_pressure_Pa=p_total
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Types
    'NoiseModel',
    'PressurePort',
    'SensorNoiseConfig',
    'FADSMeasurement',
    'FlightState',
    # Functions
    'modified_newtonian_cp',
    'tangent_cone_cp',
    'compute_surface_pressure',
    # Classes
    'FADSSensor',
]
