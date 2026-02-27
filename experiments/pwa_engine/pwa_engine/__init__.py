"""PWA Engine — Partial Wave Analysis Compute Engine V3.0.0.

Standalone distribution of the PWA Compute Engine from HyperTensor-VM.
Implements Eq. 5.48 from Badui (2020) with Gram-matrix-accelerated
extended likelihood evaluation.

Quick start::

    from pwa_engine import build_wave_set, SyntheticDataGenerator, LBFGSFitter
    from pwa_engine import BasisAmplitudes, IntensityModel, GramMatrix
    from pwa_engine import ExtendedLikelihood

    ws = build_wave_set(j_max=2.5)
    # ... see docs/papers/paper/PWA_REPLICATION_NOTE.md for full usage

Install::

    cd pwa_engine/
    pip install -e .

Reference:
    Badui, Bannon, et al. (2020), PhD Dissertation, Indiana University.
    Adams (2026), HyperTensor-VM Platform V3.0.0.
"""

from __future__ import annotations

__version__ = "3.0.0"

# Re-export the canonical implementation from experiments/pwa_engine/core.py.
# When installed standalone (pip install -e pwa_engine/), the parent repo
# must be on PYTHONPATH or the package must be installed alongside tensornet.
# For standalone distribution, copy core.py into this package.
import importlib
import sys
from pathlib import Path

# Attempt to import from the canonical location first (in-repo use).
# Fall back to local copy if experiments/ is not available (standalone install).
_core = None
try:
    from experiments.pwa_engine.core import (  # type: ignore[import-untyped]
        BasisAmplitudes,
        BreitWigner,
        ChannelConfig,
        CoupledChannelSystem,
        ExtendedLikelihood,
        GramMatrix,
        IntensityModel,
        LBFGSFitter,
        PolarizedIntensityModel,
        SyntheticDataGenerator,
        Wave,
        WaveSet,
        beam_asymmetry_sensitivity_test,
        benchmark_normalization,
        bootstrap_uncertainty,
        build_wave_set,
        compress_gram_qtt,
        compute_angular_moments,
        convention_reduction_test,
        coupled_channel_test,
        mass_dependent_fit,
        moment_comparison,
        wave_set_scan,
        wigner_D_element,
        wigner_small_d,
    )
    _core = True
except ImportError:
    # Standalone install: core.py should be copied into this package
    try:
        from pwa_engine.core import (  # type: ignore[import-untyped]
            BasisAmplitudes,
            BreitWigner,
            ChannelConfig,
            CoupledChannelSystem,
            ExtendedLikelihood,
            GramMatrix,
            IntensityModel,
            LBFGSFitter,
            PolarizedIntensityModel,
            SyntheticDataGenerator,
            Wave,
            WaveSet,
            beam_asymmetry_sensitivity_test,
            benchmark_normalization,
            bootstrap_uncertainty,
            build_wave_set,
            compress_gram_qtt,
            compute_angular_moments,
            convention_reduction_test,
            coupled_channel_test,
            mass_dependent_fit,
            moment_comparison,
            wave_set_scan,
            wigner_D_element,
            wigner_small_d,
        )
        _core = True
    except ImportError:
        _core = False

if not _core:
    raise ImportError(
        "PWA Engine core module not found. Either install the full "
        "HyperTensor-VM repo or copy experiments/pwa_engine/core.py "
        "into pwa_engine/core.py for standalone use."
    )

__all__ = [
    "BasisAmplitudes",
    "BreitWigner",
    "ChannelConfig",
    "CoupledChannelSystem",
    "ExtendedLikelihood",
    "GramMatrix",
    "IntensityModel",
    "LBFGSFitter",
    "PolarizedIntensityModel",
    "SyntheticDataGenerator",
    "Wave",
    "WaveSet",
    "beam_asymmetry_sensitivity_test",
    "benchmark_normalization",
    "bootstrap_uncertainty",
    "build_wave_set",
    "compress_gram_qtt",
    "compute_angular_moments",
    "convention_reduction_test",
    "coupled_channel_test",
    "mass_dependent_fit",
    "moment_comparison",
    "wave_set_scan",
    "wigner_D_element",
    "wigner_small_d",
]
