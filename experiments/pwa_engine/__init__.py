"""PWA Compute Engine — Full Thesis-Grade Partial Wave Analysis.

Implements the complete intensity construction from Eq. 5.48 of the
Badui dissertation with Gram-matrix-accelerated extended likelihood.

Authority: Adams (2026), HyperTensor-VM Platform V3.0.0
"""

from experiments.pwa_engine.core import (
    Wave,
    WaveSet,
    BasisAmplitudes,
    IntensityModel,
    GramMatrix,
    ExtendedLikelihood,
    SyntheticDataGenerator,
    LBFGSFitter,
    PolarizedIntensityModel,
    ChannelConfig,
    CoupledChannelSystem,
    BreitWigner,
    build_wave_set,
    convention_reduction_test,
    wave_set_scan,
    compress_gram_qtt,
    benchmark_normalization,
    compute_angular_moments,
    moment_comparison,
    beam_asymmetry_sensitivity_test,
    bootstrap_uncertainty,
    coupled_channel_test,
    mass_dependent_fit,
    wigner_small_d,
    wigner_D_element,
)

__all__ = [
    "Wave",
    "WaveSet",
    "BasisAmplitudes",
    "IntensityModel",
    "GramMatrix",
    "ExtendedLikelihood",
    "SyntheticDataGenerator",
    "LBFGSFitter",
    "PolarizedIntensityModel",
    "ChannelConfig",
    "CoupledChannelSystem",
    "BreitWigner",
    "build_wave_set",
    "convention_reduction_test",
    "wave_set_scan",
    "compress_gram_qtt",
    "benchmark_normalization",
    "compute_angular_moments",
    "moment_comparison",
    "beam_asymmetry_sensitivity_test",
    "bootstrap_uncertainty",
    "coupled_channel_test",
    "mass_dependent_fit",
    "wigner_small_d",
    "wigner_D_element",
]
