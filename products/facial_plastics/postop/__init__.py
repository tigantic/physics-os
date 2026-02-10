"""Post-operative outcome feedback loop.

Submodules:
  outcome_ingest  – Import actual surgical outcomes (photos, scans)
  alignment       – Register pre-op predictions to post-op reality
  calibration     – Update model parameters from outcome data
  validation      – Track prediction accuracy statistics
"""

from .outcome_ingest import OutcomeIngester, OutcomeRecord
from .alignment import OutcomeAligner, AlignmentResult
from .calibration import ModelCalibrator, CalibrationResult
from .validation import PredictionValidator, ValidationReport

__all__ = [
    "OutcomeIngester",
    "OutcomeRecord",
    "OutcomeAligner",
    "AlignmentResult",
    "ModelCalibrator",
    "CalibrationResult",
    "PredictionValidator",
    "ValidationReport",
]
