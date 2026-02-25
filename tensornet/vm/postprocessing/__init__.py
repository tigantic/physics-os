"""QTT Physics VM — Post-processing extractors.

Converts raw simulation output (probe time series, DFT fields) into
engineering quantities: S-parameters, far-field patterns, gain, etc.
"""

from .s_parameters import SParameterExtractor, SParameterResult
from .far_field import FarFieldExtractor, FarFieldResult

__all__ = [
    "SParameterExtractor",
    "SParameterResult",
    "FarFieldExtractor",
    "FarFieldResult",
]
