"""
Flight data validation module for V&V campaign.

This module provides infrastructure for validating CFD simulations
against real flight data, enabling comprehensive model verification.
"""

from .data_loader import (
    FlightDataSource,
    FlightDataFormat,
    FlightRecord,
    FlightCondition,
    AerodynamicData,
    FlightDataLoader,
    load_flight_data,
    parse_telemetry,
)
from .comparison import (
    ValidationMetric,
    ComparisonStatus,
    ComparisonResult,
    FieldComparison,
    TemporalComparison,
    SpatialComparison,
    FlightDataValidator,
    compare_flight_data,
)
from .uncertainty import (
    UncertaintySource,
    UncertaintyType,
    UncertaintyComponent,
    MeasurementUncertainty,
    ModelUncertainty,
    ValidationUncertainty,
    UncertaintyBudget,
    UncertaintyPropagation,
    GridConvergenceIndex,
    calculate_measurement_uncertainty,
    calculate_gci,
)
from .reports import (
    ReportFormat,
    ValidationLevel,
    ValidationCase,
    ValidationReport,
    ValidationCampaign,
    generate_validation_report,
    create_validation_case,
)

__all__ = [
    # Data Loader
    'FlightDataSource',
    'FlightDataFormat',
    'FlightRecord',
    'FlightCondition',
    'AerodynamicData',
    'FlightDataLoader',
    'load_flight_data',
    'parse_telemetry',
    # Comparison
    'ValidationMetric',
    'ComparisonStatus',
    'ComparisonResult',
    'FieldComparison',
    'TemporalComparison',
    'SpatialComparison',
    'FlightDataValidator',
    'compare_flight_data',
    # Uncertainty
    'UncertaintySource',
    'UncertaintyType',
    'UncertaintyComponent',
    'MeasurementUncertainty',
    'ModelUncertainty',
    'ValidationUncertainty',
    'UncertaintyBudget',
    'UncertaintyPropagation',
    'GridConvergenceIndex',
    'calculate_measurement_uncertainty',
    'calculate_gci',
    # Reports
    'ReportFormat',
    'ValidationLevel',
    'ValidationCase',
    'ValidationReport',
    'ValidationCampaign',
    'generate_validation_report',
    'create_validation_case',
]
