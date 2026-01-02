"""
Uncertainty quantification for flight data validation.

This module provides tools for propagating and analyzing
uncertainties in both flight data and CFD simulations.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class UncertaintySource(Enum):
    """Sources of uncertainty."""

    # Measurement uncertainties
    SENSOR_ACCURACY = auto()
    SENSOR_RESOLUTION = auto()
    CALIBRATION = auto()
    ENVIRONMENTAL = auto()

    # Model uncertainties
    TURBULENCE_MODEL = auto()
    GRID_RESOLUTION = auto()
    NUMERICAL_SCHEME = auto()
    BOUNDARY_CONDITIONS = auto()

    # Experimental uncertainties
    REPEATABILITY = auto()
    SYSTEMATIC = auto()
    RANDOM = auto()

    # Combined
    TOTAL = auto()


class UncertaintyType(Enum):
    """Types of uncertainty."""

    ALEATORY = auto()  # Inherent randomness
    EPISTEMIC = auto()  # Lack of knowledge
    MIXED = auto()  # Combination


@dataclass
class UncertaintyComponent:
    """Single uncertainty component."""

    name: str
    source: UncertaintySource
    uncertainty_type: UncertaintyType

    # Value
    value: float = 0.0
    value_percent: float = 0.0

    # Distribution parameters
    distribution: str = "normal"  # normal, uniform, triangular
    confidence_level: float = 0.95

    # Degrees of freedom (for t-distribution)
    degrees_of_freedom: int = 0

    # Correlation with other components
    correlations: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "source": self.source.name,
            "type": self.uncertainty_type.name,
            "value": self.value,
            "value_percent": self.value_percent,
            "distribution": self.distribution,
            "confidence_level": self.confidence_level,
        }


@dataclass
class MeasurementUncertainty:
    """Uncertainty analysis for measurements."""

    parameter_name: str
    measured_value: float
    unit: str = ""

    # Uncertainty components
    components: list[UncertaintyComponent] = field(default_factory=list)

    # Combined uncertainty
    combined_standard_uncertainty: float = 0.0
    expanded_uncertainty: float = 0.0
    coverage_factor: float = 2.0

    # Percent uncertainty
    percent_uncertainty: float = 0.0

    def add_component(self, component: UncertaintyComponent):
        """Add uncertainty component."""
        self.components.append(component)
        self._recalculate()

    def _recalculate(self):
        """Recalculate combined uncertainty."""
        if not self.components:
            return

        # Root sum of squares for uncorrelated components
        variance = 0.0

        for i, comp_i in enumerate(self.components):
            variance += comp_i.value**2

            # Add correlation terms
            for j, comp_j in enumerate(self.components):
                if j > i and comp_j.name in comp_i.correlations:
                    rho = comp_i.correlations[comp_j.name]
                    variance += 2 * rho * comp_i.value * comp_j.value

        self.combined_standard_uncertainty = math.sqrt(variance)
        self.expanded_uncertainty = (
            self.coverage_factor * self.combined_standard_uncertainty
        )

        if abs(self.measured_value) > 1e-10:
            self.percent_uncertainty = (
                100 * self.expanded_uncertainty / abs(self.measured_value)
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameter": self.parameter_name,
            "value": self.measured_value,
            "unit": self.unit,
            "combined_standard_uncertainty": self.combined_standard_uncertainty,
            "expanded_uncertainty": self.expanded_uncertainty,
            "coverage_factor": self.coverage_factor,
            "percent_uncertainty": self.percent_uncertainty,
            "components": [c.to_dict() for c in self.components],
        }

    def get_interval(self) -> tuple[float, float]:
        """Get uncertainty interval."""
        return (
            self.measured_value - self.expanded_uncertainty,
            self.measured_value + self.expanded_uncertainty,
        )


@dataclass
class ModelUncertainty:
    """Uncertainty from CFD model."""

    parameter_name: str
    nominal_value: float

    # Model uncertainty components
    grid_uncertainty: float = 0.0
    turbulence_model_uncertainty: float = 0.0
    numerical_uncertainty: float = 0.0
    boundary_condition_uncertainty: float = 0.0

    # Combined
    total_model_uncertainty: float = 0.0

    def __post_init__(self):
        """Calculate total uncertainty."""
        self._recalculate()

    def _recalculate(self):
        """Recalculate total model uncertainty."""
        self.total_model_uncertainty = math.sqrt(
            self.grid_uncertainty**2
            + self.turbulence_model_uncertainty**2
            + self.numerical_uncertainty**2
            + self.boundary_condition_uncertainty**2
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameter": self.parameter_name,
            "nominal_value": self.nominal_value,
            "grid_uncertainty": self.grid_uncertainty,
            "turbulence_model_uncertainty": self.turbulence_model_uncertainty,
            "numerical_uncertainty": self.numerical_uncertainty,
            "boundary_condition_uncertainty": self.boundary_condition_uncertainty,
            "total_model_uncertainty": self.total_model_uncertainty,
        }


@dataclass
class ValidationUncertainty:
    """Combined uncertainty for validation."""

    parameter_name: str

    measurement_uncertainty: MeasurementUncertainty | None = None
    model_uncertainty: ModelUncertainty | None = None

    # Combined validation uncertainty
    combined_uncertainty: float = 0.0

    # Validation comparison
    comparison_error: float = 0.0
    validation_metric: float = 0.0  # Error / uncertainty

    def __post_init__(self):
        """Calculate combined uncertainty."""
        self._recalculate()

    def _recalculate(self):
        """Recalculate combined validation uncertainty."""
        meas_unc = 0.0
        model_unc = 0.0

        if self.measurement_uncertainty:
            meas_unc = self.measurement_uncertainty.expanded_uncertainty

        if self.model_uncertainty:
            model_unc = self.model_uncertainty.total_model_uncertainty

        self.combined_uncertainty = math.sqrt(meas_unc**2 + model_unc**2)

        if self.combined_uncertainty > 1e-10:
            self.validation_metric = (
                abs(self.comparison_error) / self.combined_uncertainty
            )

    def is_validated(self, threshold: float = 2.0) -> bool:
        """
        Check if simulation is validated.

        Validation passes if:
        |E| < U_val * threshold

        where E is comparison error and U_val is validation uncertainty.
        """
        return self.validation_metric < threshold


@dataclass
class UncertaintyBudget:
    """Complete uncertainty budget."""

    name: str
    description: str = ""

    # Measurement uncertainties
    measurement_uncertainties: dict[str, MeasurementUncertainty] = field(
        default_factory=dict
    )

    # Model uncertainties
    model_uncertainties: dict[str, ModelUncertainty] = field(default_factory=dict)

    # Validation uncertainties
    validation_uncertainties: dict[str, ValidationUncertainty] = field(
        default_factory=dict
    )

    def add_measurement_uncertainty(self, unc: MeasurementUncertainty):
        """Add measurement uncertainty."""
        self.measurement_uncertainties[unc.parameter_name] = unc
        self._update_validation(unc.parameter_name)

    def add_model_uncertainty(self, unc: ModelUncertainty):
        """Add model uncertainty."""
        self.model_uncertainties[unc.parameter_name] = unc
        self._update_validation(unc.parameter_name)

    def _update_validation(self, parameter_name: str):
        """Update validation uncertainty for parameter."""
        meas = self.measurement_uncertainties.get(parameter_name)
        model = self.model_uncertainties.get(parameter_name)

        if meas or model:
            val_unc = ValidationUncertainty(
                parameter_name=parameter_name,
                measurement_uncertainty=meas,
                model_uncertainty=model,
            )
            self.validation_uncertainties[parameter_name] = val_unc

    def get_summary(self) -> dict[str, Any]:
        """Get budget summary."""
        return {
            "name": self.name,
            "description": self.description,
            "num_measurement_uncertainties": len(self.measurement_uncertainties),
            "num_model_uncertainties": len(self.model_uncertainties),
            "parameters": list(self.validation_uncertainties.keys()),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "measurement_uncertainties": {
                k: v.to_dict() for k, v in self.measurement_uncertainties.items()
            },
            "model_uncertainties": {
                k: v.to_dict() for k, v in self.model_uncertainties.items()
            },
        }


class UncertaintyPropagation:
    """
    Propagation of uncertainties through calculations.
    """

    def __init__(self, method: str = "linear"):
        """
        Initialize propagation.

        Args:
            method: Propagation method ("linear", "monte_carlo")
        """
        self.method = method

    def propagate_linear(
        self,
        func: Callable[..., float],
        inputs: dict[str, float],
        uncertainties: dict[str, float],
        delta: float = 1e-6,
    ) -> tuple[float, float]:
        """
        Propagate uncertainties using linear approximation.

        Uses finite difference to estimate sensitivities.

        Args:
            func: Function to evaluate
            inputs: Nominal input values
            uncertainties: Input uncertainties
            delta: Finite difference step

        Returns:
            (output_value, output_uncertainty)
        """
        # Nominal output
        nominal = func(**inputs)

        # Calculate sensitivities
        variance = 0.0

        for param, unc in uncertainties.items():
            if param not in inputs:
                continue

            # Perturb parameter
            perturbed_inputs = inputs.copy()
            step = delta * abs(inputs[param]) if inputs[param] != 0 else delta
            perturbed_inputs[param] = inputs[param] + step

            # Sensitivity
            perturbed_output = func(**perturbed_inputs)
            sensitivity = (perturbed_output - nominal) / step

            # Add to variance
            variance += (sensitivity * unc) ** 2

        uncertainty = math.sqrt(variance)

        return nominal, uncertainty

    def propagate_monte_carlo(
        self,
        func: Callable[..., float],
        inputs: dict[str, float],
        uncertainties: dict[str, float],
        n_samples: int = 10000,
        seed: int | None = None,
    ) -> tuple[float, float, np.ndarray]:
        """
        Propagate uncertainties using Monte Carlo sampling.

        Args:
            func: Function to evaluate
            inputs: Nominal input values
            uncertainties: Input uncertainties (standard deviations)
            n_samples: Number of Monte Carlo samples
            seed: Random seed

        Returns:
            (mean_output, std_output, samples)
        """
        if seed is not None:
            np.random.seed(seed)

        samples = np.zeros(n_samples)

        for i in range(n_samples):
            # Sample inputs
            sampled_inputs = {}
            for param, nominal in inputs.items():
                if param in uncertainties:
                    unc = uncertainties[param]
                    sampled_inputs[param] = np.random.normal(nominal, unc)
                else:
                    sampled_inputs[param] = nominal

            # Evaluate function
            samples[i] = func(**sampled_inputs)

        mean_output = float(np.mean(samples))
        std_output = float(np.std(samples))

        return mean_output, std_output, samples

    def propagate(
        self,
        func: Callable[..., float],
        inputs: dict[str, float],
        uncertainties: dict[str, float],
    ) -> tuple[float, float]:
        """
        Propagate uncertainties using configured method.

        Args:
            func: Function to evaluate
            inputs: Nominal input values
            uncertainties: Input uncertainties

        Returns:
            (output_value, output_uncertainty)
        """
        if self.method == "monte_carlo":
            mean, std, _ = self.propagate_monte_carlo(func, inputs, uncertainties)
            return mean, std
        else:
            return self.propagate_linear(func, inputs, uncertainties)


class GridConvergenceIndex:
    """
    Grid Convergence Index (GCI) for CFD uncertainty estimation.

    Based on Roache's method for estimating discretization uncertainty.
    """

    def __init__(
        self,
        refinement_ratio: float = 2.0,
        safety_factor: float = 1.25,
        target_order: float = 2.0,
    ):
        """
        Initialize GCI calculator.

        Args:
            refinement_ratio: Grid refinement ratio
            safety_factor: Safety factor for GCI
            target_order: Target order of accuracy
        """
        self.refinement_ratio = refinement_ratio
        self.safety_factor = safety_factor
        self.target_order = target_order

    def calculate_gci(
        self,
        coarse_value: float,
        medium_value: float,
        fine_value: float,
    ) -> dict[str, float]:
        """
        Calculate GCI from three grid solutions.

        Args:
            coarse_value: Value on coarsest grid
            medium_value: Value on medium grid
            fine_value: Value on finest grid

        Returns:
            Dictionary with GCI metrics
        """
        # Calculate observed order of accuracy
        epsilon_32 = medium_value - fine_value
        epsilon_21 = coarse_value - medium_value

        if abs(epsilon_32) < 1e-15 or abs(epsilon_21) < 1e-15:
            # Grid independent solution
            return {
                "fine_value": fine_value,
                "richardson_extrapolation": fine_value,
                "observed_order": self.target_order,
                "gci_fine": 0.0,
                "gci_medium": 0.0,
                "asymptotic_ratio": 1.0,
            }

        s = 1 if epsilon_32 * epsilon_21 > 0 else -1
        r = self.refinement_ratio

        # Fixed point iteration for observed order
        p = self.target_order
        for _ in range(20):
            p_new = abs(
                math.log(abs(epsilon_21 / epsilon_32))
                + math.log((r**p - s) / (r**p - s))
            ) / math.log(r)
            if abs(p_new - p) < 1e-6:
                break
            p = 0.5 * (p + p_new)

        # Richardson extrapolation
        re = fine_value + epsilon_32 / (r**p - 1)

        # GCI
        gci_fine = self.safety_factor * abs(epsilon_32) / (r**p - 1)
        gci_medium = self.safety_factor * abs(epsilon_21) / (r**p - 1)

        # Asymptotic ratio (should be ~1 for grid convergence)
        gci_21 = gci_medium * r**p
        asymptotic_ratio = gci_21 / gci_fine if gci_fine > 0 else 1.0

        return {
            "fine_value": fine_value,
            "richardson_extrapolation": re,
            "observed_order": p,
            "gci_fine": gci_fine,
            "gci_medium": gci_medium,
            "asymptotic_ratio": asymptotic_ratio,
        }


def calculate_measurement_uncertainty(
    measured_value: float,
    sensor_accuracy: float,
    calibration_uncertainty: float = 0.0,
    repeatability: float = 0.0,
    coverage_factor: float = 2.0,
) -> MeasurementUncertainty:
    """
    Calculate measurement uncertainty.

    Args:
        measured_value: Measured value
        sensor_accuracy: Sensor accuracy (same units as value)
        calibration_uncertainty: Calibration uncertainty
        repeatability: Repeatability uncertainty
        coverage_factor: Coverage factor for expanded uncertainty

    Returns:
        MeasurementUncertainty object
    """
    unc = MeasurementUncertainty(
        parameter_name="measurement",
        measured_value=measured_value,
        coverage_factor=coverage_factor,
    )

    # Add sensor accuracy
    unc.add_component(
        UncertaintyComponent(
            name="sensor_accuracy",
            source=UncertaintySource.SENSOR_ACCURACY,
            uncertainty_type=UncertaintyType.EPISTEMIC,
            value=sensor_accuracy,
        )
    )

    # Add calibration
    if calibration_uncertainty > 0:
        unc.add_component(
            UncertaintyComponent(
                name="calibration",
                source=UncertaintySource.CALIBRATION,
                uncertainty_type=UncertaintyType.EPISTEMIC,
                value=calibration_uncertainty,
            )
        )

    # Add repeatability
    if repeatability > 0:
        unc.add_component(
            UncertaintyComponent(
                name="repeatability",
                source=UncertaintySource.REPEATABILITY,
                uncertainty_type=UncertaintyType.ALEATORY,
                value=repeatability,
            )
        )

    return unc


def calculate_gci(
    coarse_value: float,
    medium_value: float,
    fine_value: float,
    refinement_ratio: float = 2.0,
) -> dict[str, float]:
    """
    Calculate Grid Convergence Index.

    Args:
        coarse_value: Value on coarsest grid
        medium_value: Value on medium grid
        fine_value: Value on finest grid
        refinement_ratio: Grid refinement ratio

    Returns:
        Dictionary with GCI metrics
    """
    gci = GridConvergenceIndex(refinement_ratio=refinement_ratio)
    return gci.calculate_gci(coarse_value, medium_value, fine_value)
