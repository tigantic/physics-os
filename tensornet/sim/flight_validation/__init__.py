"""
Flight data validation module for V&V campaign.

This module provides infrastructure for validating CFD simulations
against real flight data, enabling comprehensive model verification.
"""

from .comparison import (
                         ComparisonResult,
                         ComparisonStatus,
                         FieldComparison,
                         FlightDataValidator,
                         SpatialComparison,
                         TemporalComparison,
                         ValidationMetric,
                         compare_flight_data,
)
from .data_loader import (
                         AerodynamicData,
                         FlightCondition,
                         FlightDataFormat,
                         FlightDataLoader,
                         FlightDataSource,
                         FlightRecord,
                         load_flight_data,
                         parse_telemetry,
)
from .reports import (
                         ReportFormat,
                         ValidationCampaign,
                         ValidationCase,
                         ValidationLevel,
                         ValidationReport,
                         create_validation_case,
                         generate_validation_report,
)
from .uncertainty import (
                         GridConvergenceIndex,
                         MeasurementUncertainty,
                         ModelUncertainty,
                         UncertaintyBudget,
                         UncertaintyComponent,
                         UncertaintyPropagation,
                         UncertaintySource,
                         UncertaintyType,
                         ValidationUncertainty,
                         calculate_gci,
                         calculate_measurement_uncertainty,
)

__all__ = [
    # Data Loader
    "FlightDataSource",
    "FlightDataFormat",
    "FlightRecord",
    "FlightCondition",
    "AerodynamicData",
    "FlightDataLoader",
    "load_flight_data",
    "parse_telemetry",
    # Comparison
    "ValidationMetric",
    "ComparisonStatus",
    "ComparisonResult",
    "FieldComparison",
    "TemporalComparison",
    "SpatialComparison",
    "FlightDataValidator",
    "compare_flight_data",
    # Uncertainty
    "UncertaintySource",
    "UncertaintyType",
    "UncertaintyComponent",
    "MeasurementUncertainty",
    "ModelUncertainty",
    "ValidationUncertainty",
    "UncertaintyBudget",
    "UncertaintyPropagation",
    "GridConvergenceIndex",
    "calculate_measurement_uncertainty",
    "calculate_gci",
    # Reports
    "ReportFormat",
    "ValidationLevel",
    "ValidationCase",
    "ValidationReport",
    "ValidationCampaign",
    "generate_validation_report",
    "create_validation_case",
]
