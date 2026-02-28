"""QTT Physics VM — Parametric antenna sweep orchestrator.

Automates execution of antenna simulations across a parameter space.
Each design point is compiled, executed on GPU, and post-processed
to extract S₁₁, impedance, gain, directivity, efficiency, and pattern
metrics — producing a ``DesignPoint`` that feeds the Pareto optimizer.

Architecture:
    SweepOrchestrator
      ├── takes: AntennaDesign + parameter list
      ├── for each parameter set:
      │     compile → execute → S₁₁ → far-field → metrics
      └── returns: SweepResult (list of DesignPoints)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DesignPoint:
    """Complete evaluation of one antenna design candidate.

    Contains the parameter vector, all extracted RF metrics, and
    metadata for scoring and attestation.
    """

    # ── Identity ────────────────────────────────────────────────────
    candidate_id: str = ""
    """Unique identifier (auto-generated from param hash)."""

    family: str = ""
    """Antenna family name (e.g. ``"rectangular_patch"``)."""

    params: dict[str, float] = field(default_factory=dict)
    """Design parameter vector."""

    param_hash: str = ""
    """SHA-256 hash of the serialised parameter dict."""

    # ── Simulation metadata ─────────────────────────────────────────
    n_bits: int = 0
    """Grid resolution (bits per dimension)."""

    n_steps: int = 0
    """Number of time steps."""

    grid_size: str = ""
    """Human-readable grid size (e.g. ``"1024³"``)."""

    wall_time_s: float = 0.0
    """Total simulation wall-clock time (seconds)."""

    gpu_mem_mb: float = 0.0
    """Peak GPU memory allocation (MB)."""

    chi_max: int = 0
    """Maximum QTT rank observed."""

    success: bool = False
    """Whether the simulation succeeded without errors."""

    error: str = ""
    """Error message if simulation failed."""

    # ── S-parameter metrics ─────────────────────────────────────────
    s11_min_db: float = 0.0
    """Best (most negative) S₁₁ in dB."""

    f_resonance: float = 0.0
    """Resonant frequency (where |S₁₁| is minimised)."""

    z_in_at_resonance: complex = 0j
    """Input impedance at resonance."""

    vswr_min: float = float("inf")
    """Best VSWR across the band."""

    bandwidth_f_low: float = 0.0
    """Lower -10 dB bandwidth edge."""

    bandwidth_f_high: float = 0.0
    """Upper -10 dB bandwidth edge."""

    fractional_bandwidth: float = 0.0
    """Fractional bandwidth at -10 dB."""

    n_freq_bins: int = 0
    """Number of frequency bins in S₁₁ extraction."""

    # ── Far-field metrics ───────────────────────────────────────────
    peak_gain_dbi: float = -100.0
    """Peak realised gain (dBi)."""

    peak_directivity_dbi: float = -100.0
    """Peak directivity (dBi)."""

    radiation_efficiency: float = 0.0
    """Radiation efficiency η = P_rad / P_accepted."""

    peak_theta_deg: float = 0.0
    """Elevation angle of peak gain (degrees)."""

    peak_phi_deg: float = 0.0
    """Azimuth angle of peak gain (degrees)."""

    # ── DFT field norms ─────────────────────────────────────────────
    dft_norms: dict[str, float] = field(default_factory=dict)
    """DFT field norms (for debugging / quality check)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON output."""
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "params": self.params,
            "param_hash": self.param_hash,
            "n_bits": self.n_bits,
            "n_steps": self.n_steps,
            "grid_size": self.grid_size,
            "wall_time_s": self.wall_time_s,
            "gpu_mem_mb": self.gpu_mem_mb,
            "chi_max": self.chi_max,
            "success": self.success,
            "error": self.error,
            "s11_min_db": self.s11_min_db,
            "f_resonance": self.f_resonance,
            "z_in_at_resonance": str(self.z_in_at_resonance),
            "vswr_min": self.vswr_min,
            "bandwidth_f_low": self.bandwidth_f_low,
            "bandwidth_f_high": self.bandwidth_f_high,
            "fractional_bandwidth": self.fractional_bandwidth,
            "n_freq_bins": self.n_freq_bins,
            "peak_gain_dbi": self.peak_gain_dbi,
            "peak_directivity_dbi": self.peak_directivity_dbi,
            "radiation_efficiency": self.radiation_efficiency,
            "peak_theta_deg": self.peak_theta_deg,
            "peak_phi_deg": self.peak_phi_deg,
            "dft_norms": self.dft_norms,
        }


@dataclass
class SweepResult:
    """Result of a parametric sweep: collection of design points.

    Includes summary statistics and metadata for attestation.
    """

    family: str
    """Antenna family name."""

    points: list[DesignPoint] = field(default_factory=list)
    """All evaluated design points."""

    total_wall_time_s: float = 0.0
    """Total sweep wall time."""

    n_bits: int = 0
    """Resolution used for the sweep."""

    n_steps: int = 0
    """Steps per design point."""

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_successful(self) -> int:
        return sum(1 for p in self.points if p.success)

    @property
    def n_failed(self) -> int:
        return sum(1 for p in self.points if not p.success)

    def successful_points(self) -> list[DesignPoint]:
        """Return only successful design points."""
        return [p for p in self.points if p.success]

    def best_by_gain(self) -> DesignPoint | None:
        """Return the design point with highest peak gain."""
        valid = self.successful_points()
        if not valid:
            return None
        return max(valid, key=lambda p: p.peak_gain_dbi)

    def best_by_bandwidth(self) -> DesignPoint | None:
        """Return the design point with widest fractional bandwidth."""
        valid = self.successful_points()
        if not valid:
            return None
        return max(valid, key=lambda p: p.fractional_bandwidth)

    def best_by_s11(self) -> DesignPoint | None:
        """Return the design point with best (lowest) S₁₁."""
        valid = self.successful_points()
        if not valid:
            return None
        return min(valid, key=lambda p: p.s11_min_db)

    def summary(self) -> dict[str, Any]:
        """Summary statistics for the sweep."""
        valid = self.successful_points()
        return {
            "family": self.family,
            "n_points": self.n_points,
            "n_successful": self.n_successful,
            "n_failed": self.n_failed,
            "total_wall_time_s": self.total_wall_time_s,
            "n_bits": self.n_bits,
            "n_steps": self.n_steps,
            "best_gain_dbi": (
                max(p.peak_gain_dbi for p in valid) if valid else None
            ),
            "best_s11_db": (
                min(p.s11_min_db for p in valid) if valid else None
            ),
            "best_fractional_bw": (
                max(p.fractional_bandwidth for p in valid) if valid else None
            ),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise full sweep result to JSON."""
        data = {
            "summary": self.summary(),
            "points": [p.to_dict() for p in self.points],
        }
        return json.dumps(data, indent=indent, default=str)


class SweepOrchestrator:
    """Automated parametric antenna sweep engine.

    Compiles, executes, and post-processes antenna simulations across
    a parameter space at QTT scale.

    Parameters
    ----------
    n_bits : int
        Grid resolution (bits per dimension).  512³ = 9, 1024³ = 10, etc.
    n_steps : int
        Number of time steps per simulation.
    max_rank : int
        QTT rank ceiling.
    extract_far_field : bool
        Whether to compute far-field patterns (adds cost).
    n_surface_samples : int
        Surface samples for far-field extraction.
    n_theta : int
        Elevation angle samples for far-field.
    n_phi : int
        Azimuth angle samples for far-field.
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        n_bits: int = 10,
        n_steps: int = 300,
        max_rank: int = 48,
        extract_far_field: bool = True,
        n_surface_samples: int = 8,
        n_theta: int = 37,
        n_phi: int = 18,
        verbose: bool = True,
    ) -> None:
        self._n_bits = n_bits
        self._n_steps = n_steps
        self._max_rank = max_rank
        self._extract_ff = extract_far_field
        self._n_surf = n_surface_samples
        self._n_theta = n_theta
        self._n_phi = n_phi
        self._verbose = verbose

    def sweep(
        self,
        design: Any,
        param_list: list[dict[str, float]],
    ) -> SweepResult:
        """Execute a parametric sweep.

        Parameters
        ----------
        design : _BaseAntennaDesign
            Parametric antenna design instance.
        param_list : list[dict[str, float]]
            List of parameter dictionaries to evaluate.

        Returns
        -------
        SweepResult
            Collection of evaluated design points.
        """
        import torch

        from ..gpu_runtime import GPURuntime, GPURankGovernor
        from ..postprocessing.s_parameters import SParameterExtractor
        from ..postprocessing.far_field import FarFieldExtractor

        sweep_start = time.perf_counter()
        result = SweepResult(
            family=design.family_name,
            n_bits=self._n_bits,
            n_steps=self._n_steps,
        )

        n_total = len(param_list)
        if self._verbose:
            N = 2 ** self._n_bits
            print(f"\n{'='*72}")
            print(f"  PARAMETRIC SWEEP — {design.family_name}")
            print(f"  Grid: {N}³  |  Steps: {self._n_steps}  |  "
                  f"Candidates: {n_total}")
            print(f"{'='*72}")

        for idx, params in enumerate(param_list):
            point = self._evaluate_one(
                design, params, idx, n_total,
                GPURankGovernor, GPURuntime,
                SParameterExtractor, FarFieldExtractor,
                torch,
            )
            result.points.append(point)

            # Free GPU memory between runs
            torch.cuda.empty_cache()
            gc.collect()

        result.total_wall_time_s = time.perf_counter() - sweep_start

        if self._verbose:
            summary = result.summary()
            print(f"\n  Sweep complete: {summary['n_successful']}/{n_total} "
                  f"succeeded in {result.total_wall_time_s:.1f}s")
            if summary["best_gain_dbi"] is not None:
                print(f"  Best gain:      {summary['best_gain_dbi']:.1f} dBi")
                print(f"  Best S₁₁:      {summary['best_s11_db']:.1f} dB")
                print(f"  Best BW:        "
                      f"{summary['best_fractional_bw']*100:.1f}%")

        return result

    def _evaluate_one(
        self,
        design: Any,
        params: dict[str, float],
        idx: int,
        n_total: int,
        GPURankGovernor: type,
        GPURuntime: type,
        SParameterExtractor: type,
        FarFieldExtractor: type,
        torch: Any,
    ) -> DesignPoint:
        """Evaluate a single design candidate."""
        # Generate candidate ID from parameter hash
        param_json = json.dumps(params, sort_keys=True)
        param_hash = hashlib.sha256(param_json.encode()).hexdigest()[:16]
        candidate_id = f"{design.family_name[:4].upper()}-{param_hash}"

        N = 2 ** self._n_bits
        point = DesignPoint(
            candidate_id=candidate_id,
            family=design.family_name,
            params=dict(params),
            param_hash=param_hash,
            n_bits=self._n_bits,
            n_steps=self._n_steps,
            grid_size=f"{N}³",
        )

        if self._verbose:
            compact = ", ".join(f"{k}={v:.3f}" for k, v in params.items())
            print(f"\n  [{idx+1}/{n_total}] {candidate_id}: {compact}")

        try:
            # ── Compile ────────────────────────────────────────────
            compiler = design.to_compiler(
                params,
                n_bits=self._n_bits,
                n_steps=self._n_steps,
                dft_all_components=self._extract_ff,
            )
            program = compiler.compile()

            # ── Execute ────────────────────────────────────────────
            governor = GPURankGovernor(
                max_rank=self._max_rank, rel_tol=1e-10
            )
            runtime = GPURuntime(governor=governor)

            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            exec_result = runtime.execute(program)
            point.wall_time_s = time.perf_counter() - t0
            point.gpu_mem_mb = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            )

            if not exec_result.success:
                point.success = False
                point.error = exec_result.error
                if self._verbose:
                    print(f"    FAIL: {exec_result.error}")
                return point

            # ── Extract chi_max ────────────────────────────────────
            chi = 0
            for ft in exec_result.fields.values():
                for c in ft.cores:
                    chi = max(chi, c.shape[0])
            point.chi_max = chi

            # ── S-parameter extraction ────────────────────────────
            s_extractor = SParameterExtractor(
                dt=program.dt,
                z0=program.params.get("port_impedance", 1.0),
                gap_size=program.params.get("port_gap_size", 0.02),
                h_loop_half_side=program.params.get(
                    "port_h_loop_half_side", 0.02
                ),
                polarization=int(
                    program.params.get("source_polarization", 2)
                ),
            )
            s_result = s_extractor.extract(exec_result.probes)

            if len(s_result.frequencies) > 0:
                s_summary = s_result.summary()
                point.s11_min_db = s_summary["s11_min_dB"]
                point.f_resonance = s_summary["f_resonance"]
                point.z_in_at_resonance = s_summary["z_in_at_resonance"]
                point.vswr_min = float(np.min(s_result.vswr))
                point.bandwidth_f_low = s_summary["bandwidth_f_low"]
                point.bandwidth_f_high = s_summary["bandwidth_f_high"]
                point.fractional_bandwidth = s_summary[
                    "fractional_bandwidth"
                ]
                point.n_freq_bins = s_summary["n_freq_bins"]

            # ── DFT norms ─────────────────────────────────────────
            for comp in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
                re_k = f"dft_re_{comp}"
                im_k = f"dft_im_{comp}"
                if re_k in exec_result.fields:
                    point.dft_norms[f"|Re({comp})|"] = float(
                        exec_result.fields[re_k].norm()
                    )
                if im_k in exec_result.fields:
                    point.dft_norms[f"|Im({comp})|"] = float(
                        exec_result.fields[im_k].norm()
                    )

            # ── Far-field extraction ──────────────────────────────
            if self._extract_ff:
                # Check DFT fields have meaningful energy before
                # attempting far-field (short runs produce zero DFT)
                dft_energy = sum(
                    abs(v) for v in point.dft_norms.values()
                )
                if dft_energy < 1e-20:
                    if self._verbose:
                        print(
                            "    ⚠ DFT fields near-zero — "
                            "skipping far-field (need more steps)"
                        )
                else:
                    ff_extractor = FarFieldExtractor(
                        frequency=program.params.get("freq_center", 1.0),
                        domain_size=1.0,
                        n_surface_samples=self._n_surf,
                        n_theta=self._n_theta,
                        n_phi=self._n_phi,
                        surface_margin=0.15,
                    )
                    ff_result = ff_extractor.extract(exec_result.fields)
                    ff_summary = ff_result.summary()

                    # Guard against unconverged far-field
                    gain = ff_summary["peak_gain_dBi"]
                    if gain > -50.0:
                        point.peak_gain_dbi = gain
                        point.peak_directivity_dbi = ff_summary[
                            "peak_directivity_dBi"
                        ]
                        point.radiation_efficiency = ff_summary[
                            "radiation_efficiency"
                        ]
                        point.peak_theta_deg = ff_summary["peak_theta_deg"]
                        point.peak_phi_deg = ff_summary["peak_phi_deg"]
                    elif self._verbose:
                        print(
                            f"    ⚠ Far-field unconverged "
                            f"(gain={gain:.0f} dBi) — "
                            "need more time steps"
                        )

            point.success = True

            if self._verbose:
                print(
                    f"    OK: {point.wall_time_s:.1f}s, "
                    f"χ={point.chi_max}, "
                    f"S₁₁={point.s11_min_db:.1f} dB, "
                    f"gain={point.peak_gain_dbi:.1f} dBi, "
                    f"BW={point.fractional_bandwidth*100:.1f}%"
                )

            # Clean up GPU tensors
            del exec_result, runtime, governor

        except Exception as exc:
            point.success = False
            point.error = str(exc)
            if self._verbose:
                print(f"    ERROR: {exc}")
            logger.exception(
                "Sweep candidate %s failed: %s", candidate_id, exc
            )

        return point
