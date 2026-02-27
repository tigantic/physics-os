"""
HyperFOAM Universal Intake System
=================================

Enterprise-grade document ingestion and extraction for CFD simulation setup.

Supported file types:
- PDF: Blueprint extraction with OCR and dimension parsing
- PNG/JPG: Blueprint images with scale detection
- IFC: BIM models with full geometry extraction
- DOC/DOCX: Specification documents with NLP parsing
- JSON: Direct job_spec import

Features:
- Multi-unit support (feet/inches, meters, centimeters)
- Intelligent field extraction
- Universal form interface
- Validation and compliance checking
- One-click job_spec.json generation

Copyright (c) 2026 HyperFOAM Team
License: Proprietary
"""

__version__ = "1.0.0"
__author__ = "HyperFOAM Team"

from .schema import IntakeField, FieldCategory, IntakeSchema
from .units import UnitSystem, UnitConverter, Measurement
from .job_spec import JobSpecGenerator

__all__ = [
    "IntakeField",
    "FieldCategory", 
    "IntakeSchema",
    "UnitSystem",
    "UnitConverter",
    "Measurement",
    "JobSpecGenerator",
]
