"""
Flight data comparison and validation utilities.

This module provides tools for comparing CFD simulation results
with flight test data for validation purposes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum, auto
import math
import numpy as np

from .data_loader import FlightRecord, FlightCondition, AerodynamicData


class ValidationMetric(Enum):
    """Validation metrics for comparison."""
    MEAN_ERROR = auto()
    RMS_ERROR = auto()
    MAX_ERROR = auto()
    R_SQUARED = auto()
    CORRELATION = auto()
    BIAS = auto()
    STANDARD_DEVIATION = auto()
    PERCENT_ERROR = auto()
    VALIDATION_FACTOR = auto()


class ComparisonStatus(Enum):
    """Status of comparison result."""
    EXCELLENT = auto()  # < 2% error
    GOOD = auto()       # 2-5% error
    ACCEPTABLE = auto() # 5-10% error
    MARGINAL = auto()   # 10-20% error
    POOR = auto()       # > 20% error
    FAILED = auto()     # Comparison failed


@dataclass
class FieldComparison:
    """Comparison result for a single field."""
    field_name: str
    
    # Data arrays
    flight_data: np.ndarray
    simulation_data: np.ndarray
    
    # Metrics
    mean_error: float = 0.0
    rms_error: float = 0.0
    max_error: float = 0.0
    bias: float = 0.0
    std_dev: float = 0.0
    correlation: float = 0.0
    r_squared: float = 0.0
    
    # Normalized metrics
    mean_percent_error: float = 0.0
    max_percent_error: float = 0.0
    
    # Status
    status: ComparisonStatus = ComparisonStatus.GOOD
    
    # Uncertainty info
    uncertainty_ratio: float = 0.0  # Error / uncertainty
    within_uncertainty: bool = True
    
    def __post_init__(self):
        """Calculate metrics from data."""
        if len(self.flight_data) > 0 and len(self.simulation_data) > 0:
            self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate comparison metrics."""
        flight = np.asarray(self.flight_data)
        sim = np.asarray(self.simulation_data)
        
        # Ensure same length
        n = min(len(flight), len(sim))
        flight = flight[:n]
        sim = sim[:n]
        
        if n == 0:
            return
        
        # Error calculations
        errors = sim - flight
        
        self.mean_error = float(np.mean(errors))
        self.rms_error = float(np.sqrt(np.mean(errors**2)))
        self.max_error = float(np.max(np.abs(errors)))
        self.bias = self.mean_error
        self.std_dev = float(np.std(errors))
        
        # Correlation and R-squared
        if n > 1:
            flight_mean = np.mean(flight)
            sim_mean = np.mean(sim)
            
            numerator = np.sum((flight - flight_mean) * (sim - sim_mean))
            denom1 = np.sqrt(np.sum((flight - flight_mean)**2))
            denom2 = np.sqrt(np.sum((sim - sim_mean)**2))
            
            if denom1 > 0 and denom2 > 0:
                self.correlation = float(numerator / (denom1 * denom2))
            
            # R-squared
            ss_res = np.sum((flight - sim)**2)
            ss_tot = np.sum((flight - flight_mean)**2)
            
            if ss_tot > 0:
                self.r_squared = float(1 - ss_res / ss_tot)
        
        # Percent errors
        flight_ref = np.abs(flight)
        valid_mask = flight_ref > 1e-10
        
        if np.any(valid_mask):
            percent_errors = 100 * np.abs(errors[valid_mask]) / flight_ref[valid_mask]
            self.mean_percent_error = float(np.mean(percent_errors))
            self.max_percent_error = float(np.max(percent_errors))
        
        # Determine status
        self._determine_status()
    
    def _determine_status(self):
        """Determine comparison status based on metrics."""
        mpe = self.mean_percent_error
        
        if mpe < 2:
            self.status = ComparisonStatus.EXCELLENT
        elif mpe < 5:
            self.status = ComparisonStatus.GOOD
        elif mpe < 10:
            self.status = ComparisonStatus.ACCEPTABLE
        elif mpe < 20:
            self.status = ComparisonStatus.MARGINAL
        else:
            self.status = ComparisonStatus.POOR
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'field_name': self.field_name,
            'mean_error': self.mean_error,
            'rms_error': self.rms_error,
            'max_error': self.max_error,
            'bias': self.bias,
            'std_dev': self.std_dev,
            'correlation': self.correlation,
            'r_squared': self.r_squared,
            'mean_percent_error': self.mean_percent_error,
            'max_percent_error': self.max_percent_error,
            'status': self.status.name,
        }


@dataclass
class TemporalComparison:
    """Comparison of time-series data."""
    field_name: str
    timestamps: np.ndarray
    
    flight_values: np.ndarray
    simulation_values: np.ndarray
    error_values: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Time-varying metrics
    instantaneous_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    cumulative_rms: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Overall comparison
    field_comparison: Optional[FieldComparison] = None
    
    def __post_init__(self):
        """Calculate temporal metrics."""
        if len(self.flight_values) > 0:
            self._calculate_temporal_metrics()
    
    def _calculate_temporal_metrics(self):
        """Calculate time-varying metrics."""
        n = min(len(self.flight_values), len(self.simulation_values))
        
        if n == 0:
            return
        
        flight = self.flight_values[:n]
        sim = self.simulation_values[:n]
        
        # Instantaneous errors
        self.error_values = sim - flight
        self.instantaneous_errors = np.abs(self.error_values)
        
        # Cumulative RMS
        self.cumulative_rms = np.zeros(n)
        for i in range(n):
            self.cumulative_rms[i] = np.sqrt(np.mean(self.error_values[:i+1]**2))
        
        # Overall comparison
        self.field_comparison = FieldComparison(
            field_name=self.field_name,
            flight_data=flight,
            simulation_data=sim,
        )
    
    def get_error_at_time(self, t: float) -> float:
        """Get interpolated error at specific time."""
        if len(self.timestamps) == 0:
            return 0.0
        
        return float(np.interp(t, self.timestamps, self.instantaneous_errors))


@dataclass
class SpatialComparison:
    """Comparison of spatial data (e.g., pressure distributions)."""
    field_name: str
    
    # Coordinates
    x: np.ndarray
    y: np.ndarray
    z: Optional[np.ndarray] = None
    
    # Data
    flight_field: np.ndarray = field(default_factory=lambda: np.array([]))
    simulation_field: np.ndarray = field(default_factory=lambda: np.array([]))
    error_field: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Metrics
    l2_norm_error: float = 0.0
    linf_norm_error: float = 0.0
    
    def calculate_metrics(self):
        """Calculate spatial comparison metrics."""
        if len(self.flight_field) == 0:
            return
        
        self.error_field = self.simulation_field - self.flight_field
        
        # Norm errors
        self.l2_norm_error = float(np.sqrt(np.mean(self.error_field**2)))
        self.linf_norm_error = float(np.max(np.abs(self.error_field)))


@dataclass
class ComparisonResult:
    """Complete comparison result."""
    flight_record_id: str
    simulation_id: str
    
    # Field comparisons
    field_comparisons: Dict[str, FieldComparison] = field(default_factory=dict)
    temporal_comparisons: Dict[str, TemporalComparison] = field(default_factory=dict)
    spatial_comparisons: Dict[str, SpatialComparison] = field(default_factory=dict)
    
    # Overall status
    overall_status: ComparisonStatus = ComparisonStatus.GOOD
    validation_passed: bool = True
    
    # Summary statistics
    num_points_compared: int = 0
    comparison_coverage: float = 0.0  # Fraction of flight data compared
    
    # Timestamps
    comparison_time: str = ""
    
    def add_field_comparison(self, comparison: FieldComparison):
        """Add field comparison."""
        self.field_comparisons[comparison.field_name] = comparison
        self._update_overall_status()
    
    def _update_overall_status(self):
        """Update overall status from field comparisons."""
        if not self.field_comparisons:
            return
        
        statuses = [c.status for c in self.field_comparisons.values()]
        
        # Use worst status
        status_order = [
            ComparisonStatus.EXCELLENT,
            ComparisonStatus.GOOD,
            ComparisonStatus.ACCEPTABLE,
            ComparisonStatus.MARGINAL,
            ComparisonStatus.POOR,
            ComparisonStatus.FAILED,
        ]
        
        worst_idx = max(status_order.index(s) for s in statuses)
        self.overall_status = status_order[worst_idx]
        
        # Validation passes if all comparisons are at least MARGINAL
        self.validation_passed = worst_idx <= status_order.index(ComparisonStatus.MARGINAL)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comparison summary."""
        return {
            'flight_record_id': self.flight_record_id,
            'simulation_id': self.simulation_id,
            'overall_status': self.overall_status.name,
            'validation_passed': self.validation_passed,
            'num_fields_compared': len(self.field_comparisons),
            'num_points_compared': self.num_points_compared,
            'field_summaries': {
                name: {
                    'mean_percent_error': c.mean_percent_error,
                    'status': c.status.name,
                }
                for name, c in self.field_comparisons.items()
            },
        }


class FlightDataValidator:
    """
    Validator for comparing simulation results with flight data.
    """
    
    def __init__(
        self,
        tolerances: Optional[Dict[str, float]] = None,
        required_fields: Optional[List[str]] = None,
    ):
        """
        Initialize validator.
        
        Args:
            tolerances: Per-field error tolerances (percent)
            required_fields: Fields that must be present for validation
        """
        self.tolerances = tolerances or {
            'cl': 5.0,
            'cd': 10.0,
            'cm': 10.0,
            'cp': 5.0,
        }
        self.required_fields = required_fields or ['cl', 'cd']
    
    def validate(
        self,
        flight_record: FlightRecord,
        simulation_data: Dict[str, np.ndarray],
        simulation_id: str = "simulation",
    ) -> ComparisonResult:
        """
        Validate simulation against flight data.
        
        Args:
            flight_record: Flight test data
            simulation_data: Simulation results keyed by field name
            simulation_id: Identifier for simulation
        
        Returns:
            ComparisonResult with all comparisons
        """
        result = ComparisonResult(
            flight_record_id=flight_record.record_id,
            simulation_id=simulation_id,
        )
        
        # Extract flight aero data as arrays
        flight_data = self._extract_flight_arrays(flight_record)
        
        # Compare each field
        for field_name, sim_values in simulation_data.items():
            if field_name not in flight_data:
                continue
            
            flight_values = flight_data[field_name]
            
            comparison = FieldComparison(
                field_name=field_name,
                flight_data=flight_values,
                simulation_data=sim_values,
            )
            
            # Check against tolerance
            if field_name in self.tolerances:
                tolerance = self.tolerances[field_name]
                if comparison.mean_percent_error > tolerance:
                    comparison.status = max(
                        comparison.status,
                        ComparisonStatus.MARGINAL
                    )
            
            result.add_field_comparison(comparison)
            result.num_points_compared += len(flight_values)
        
        # Check required fields
        for field_name in self.required_fields:
            if field_name not in result.field_comparisons:
                result.overall_status = ComparisonStatus.FAILED
                result.validation_passed = False
        
        return result
    
    def _extract_flight_arrays(
        self,
        record: FlightRecord,
    ) -> Dict[str, np.ndarray]:
        """Extract flight data as numpy arrays."""
        data = {}
        
        if record.aero_data:
            data['cl'] = np.array([a.cl for a in record.aero_data])
            data['cd'] = np.array([a.cd for a in record.aero_data])
            data['cy'] = np.array([a.cy for a in record.aero_data])
            data['cm'] = np.array([a.cm for a in record.aero_data])
            data['timestamp'] = np.array([a.timestamp for a in record.aero_data])
        
        if record.conditions:
            data['mach'] = np.array([c.mach_number for c in record.conditions])
            data['aoa'] = np.array([c.angle_of_attack_deg for c in record.conditions])
            data['altitude'] = np.array([c.altitude_m for c in record.conditions])
            data['condition_timestamp'] = np.array([c.timestamp for c in record.conditions])
        
        return data
    
    def validate_temporal(
        self,
        flight_record: FlightRecord,
        simulation_timeseries: Dict[str, Tuple[np.ndarray, np.ndarray]],
        simulation_id: str = "simulation",
    ) -> ComparisonResult:
        """
        Validate time-series simulation data.
        
        Args:
            flight_record: Flight test data
            simulation_timeseries: Dict mapping field name to (timestamps, values)
            simulation_id: Identifier for simulation
        
        Returns:
            ComparisonResult with temporal comparisons
        """
        result = ComparisonResult(
            flight_record_id=flight_record.record_id,
            simulation_id=simulation_id,
        )
        
        # Extract flight data
        flight_data = self._extract_flight_arrays(flight_record)
        
        for field_name, (sim_times, sim_values) in simulation_timeseries.items():
            if field_name not in flight_data:
                continue
            
            flight_values = flight_data[field_name]
            
            # Determine common time base
            if field_name in ['cl', 'cd', 'cy', 'cm']:
                flight_times = flight_data.get('timestamp', np.arange(len(flight_values)))
            else:
                flight_times = flight_data.get('condition_timestamp', np.arange(len(flight_values)))
            
            # Interpolate to common times
            common_times = np.union1d(flight_times, sim_times)
            common_times = common_times[
                (common_times >= max(flight_times.min(), sim_times.min())) &
                (common_times <= min(flight_times.max(), sim_times.max()))
            ]
            
            if len(common_times) == 0:
                continue
            
            interp_flight = np.interp(common_times, flight_times, flight_values)
            interp_sim = np.interp(common_times, sim_times, sim_values)
            
            temporal = TemporalComparison(
                field_name=field_name,
                timestamps=common_times,
                flight_values=interp_flight,
                simulation_values=interp_sim,
            )
            
            result.temporal_comparisons[field_name] = temporal
            
            if temporal.field_comparison:
                result.add_field_comparison(temporal.field_comparison)
        
        return result


def compare_flight_data(
    flight_record: FlightRecord,
    simulation_data: Dict[str, np.ndarray],
    tolerances: Optional[Dict[str, float]] = None,
) -> ComparisonResult:
    """
    Compare simulation data with flight record.
    
    Args:
        flight_record: Flight test data
        simulation_data: Simulation results keyed by field name
        tolerances: Per-field error tolerances
    
    Returns:
        ComparisonResult with all comparisons
    """
    validator = FlightDataValidator(tolerances=tolerances)
    return validator.validate(flight_record, simulation_data)
