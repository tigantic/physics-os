"""
HyperFOAM Staging Area
======================

The "Ingest → Validate → Solve" pipeline.

Components:
- HVACDocumentParser: Extracts data from PDFs/Excel, returns UI State
- SimulationSubmitter: Converts validated UI data to SI Physics Payload
- sanitize: Security utilities for input validation
- logger: Structured logging infrastructure
"""

from .ingestor import HVACDocumentParser
from .submitter import SimulationSubmitter
from .sanitize import (
    sanitize_filename,
    sanitize_project_name,
    sanitize_room_name,
    sanitize_path,
    sanitize_numeric,
    is_safe_extension,
)
from .logger import get_logger, log_operation, log_user_action, log_error

__all__ = [
    "HVACDocumentParser",
    "SimulationSubmitter",
    "sanitize_filename",
    "sanitize_project_name",
    "sanitize_room_name",
    "sanitize_path",
    "sanitize_numeric",
    "is_safe_extension",
    "get_logger",
    "log_operation",
    "log_user_action",
    "log_error",
]
