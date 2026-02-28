"""QTT Physics VM — S-Parameter extraction from probe time series.

Extracts input impedance Z_in(f) and reflection coefficient S₁₁(f)
from time-domain voltage and current probes recorded by the antenna
compiler's lumped port.

Physics
-------
A lumped port in FDTD-style simulation records:

  V(t) = −E_p(feed) × gap_size       (voltage across the gap)
  I(t) = ∮ B · dl around the feed      (current via Ampère loop)

FFT→ V(f), I(f) → Z_in(f) = V(f)/I(f)
S₁₁(f) = (Z_in − Z₀) / (Z_in + Z₀)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SParameterResult:
    """Frequency-domain S-parameter extraction result.

    All arrays are indexed by frequency bin.
    """

    frequencies: NDArray
    """Frequency array (simulation units)."""

    s11_complex: NDArray
    """Complex S₁₁(f) = (Z_in − Z₀) / (Z_in + Z₀)."""

    z_in_complex: NDArray
    """Complex input impedance Z_in(f) = V(f) / I(f)."""

    v_spectrum: NDArray
    """Voltage spectrum V(f) (complex)."""

    i_spectrum: NDArray
    """Current spectrum I(f) (complex)."""

    dt: float
    """Time step used in simulation."""

    z0: float
    """Reference impedance."""

    @property
    def s11_db(self) -> NDArray:
        """Return loss |S₁₁| in dB."""
        mag = np.abs(self.s11_complex)
        mag = np.where(mag > 1e-30, mag, 1e-30)
        return 20.0 * np.log10(mag)

    @property
    def vswr(self) -> NDArray:
        """Voltage Standing Wave Ratio from |S₁₁|."""
        mag = np.abs(self.s11_complex)
        mag = np.clip(mag, 0.0, 0.999)
        return (1.0 + mag) / (1.0 - mag)

    @property
    def z_in_real(self) -> NDArray:
        """Real part of input impedance (resistance)."""
        return np.real(self.z_in_complex)

    @property
    def z_in_imag(self) -> NDArray:
        """Imaginary part of input impedance (reactance)."""
        return np.imag(self.z_in_complex)

    def bandwidth(
        self,
        threshold_db: float = -10.0,
    ) -> tuple[float, float, float]:
        """Impedance bandwidth where S₁₁ < threshold.

        Returns
        -------
        f_low : float
            Lower band edge.
        f_high : float
            Upper band edge.
        fractional_bw : float
            Fractional bandwidth = (f_high − f_low) / f_center.
        """
        mask = self.s11_db < threshold_db
        if not np.any(mask):
            return 0.0, 0.0, 0.0
        indices = np.where(mask)[0]
        f_low = float(self.frequencies[indices[0]])
        f_high = float(self.frequencies[indices[-1]])
        f_center = 0.5 * (f_low + f_high)
        if f_center < 1e-30:
            return f_low, f_high, 0.0
        return f_low, f_high, (f_high - f_low) / f_center

    def summary(self) -> dict[str, Any]:
        """Summary metrics for reporting."""
        if len(self.frequencies) == 0:
            return {
                "s11_min_dB": float("nan"),
                "f_resonance": float("nan"),
                "z_in_at_resonance": complex("nan"),
                "bandwidth_f_low": 0.0,
                "bandwidth_f_high": 0.0,
                "fractional_bandwidth": 0.0,
                "z0": self.z0,
                "n_freq_bins": 0,
            }
        s11_min_db = float(np.min(self.s11_db))
        f_resonance_idx = int(np.argmin(np.abs(self.s11_complex)))
        f_resonance = float(self.frequencies[f_resonance_idx])
        z_at_res = complex(self.z_in_complex[f_resonance_idx])
        f_low, f_high, frac_bw = self.bandwidth(-10.0)
        return {
            "s11_min_dB": s11_min_db,
            "f_resonance": f_resonance,
            "z_in_at_resonance": z_at_res,
            "bandwidth_f_low": f_low,
            "bandwidth_f_high": f_high,
            "fractional_bandwidth": frac_bw,
            "z0": self.z0,
            "n_freq_bins": len(self.frequencies),
        }


class SParameterExtractor:
    """Extract S-parameters from lumped-port probe time series.

    Parameters
    ----------
    dt : float
        Simulation time step.
    z0 : float
        Reference impedance (simulation units).
    gap_size : float
        Voltage gap size for V = −E × gap.
    h_loop_half_side : float
        Half-side of the rectangular Ampère-loop for I.
    polarization : int
        Source polarisation axis (0=x, 1=y, 2=z).
    """

    def __init__(
        self,
        dt: float,
        z0: float = 1.0,
        gap_size: float = 0.02,
        h_loop_half_side: float = 0.02,
        polarization: int = 2,
    ) -> None:
        self._dt = dt
        self._z0 = z0
        self._gap = gap_size
        self._delta = h_loop_half_side
        self._pol = polarization

    def extract(
        self,
        probes: dict[str, list[float]],
        freq_min: float | None = None,
        freq_max: float | None = None,
    ) -> SParameterResult:
        """Extract S₁₁(f) from probe time series.

        Parameters
        ----------
        probes : dict[str, list[float]]
            Probe time series from ``GPUExecutionResult.probes``.
            Expected keys: ``V_port`` and the four B-loop probes.
        freq_min, freq_max : float, optional
            Restrict output to this frequency range.  If *None*,
            uses the full Nyquist range.

        Returns
        -------
        SParameterResult
        """
        # ── Extract raw time series ─────────────────────────────────
        v_raw = np.asarray(probes["V_port"], dtype=np.float64)
        n_steps = len(v_raw)

        # ── Reconstruct current from B-loop probes ──────────────────
        _PERP = {0: (1, 2), 1: (2, 0), 2: (0, 1)}
        q_axis, r_axis = _PERP[self._pol]
        q_label = ["x", "y", "z"][q_axis]
        r_label = ["x", "y", "z"][r_axis]

        bq_r_neg = np.asarray(
            probes[f"B{q_label}_r_neg"], dtype=np.float64
        )
        bq_r_pos = np.asarray(
            probes[f"B{q_label}_r_pos"], dtype=np.float64
        )
        br_q_pos = np.asarray(
            probes[f"B{r_label}_q_pos"], dtype=np.float64
        )
        br_q_neg = np.asarray(
            probes[f"B{r_label}_q_neg"], dtype=np.float64
        )

        # Rectangular loop integral: I = 2δ × (B_q|_r- − B_q|_r+ + B_r|_q+ − B_r|_q-)
        two_delta = 2.0 * self._delta
        i_raw = two_delta * (
            bq_r_neg - bq_r_pos + br_q_pos - br_q_neg
        )

        # ── Voltage: V = −E_pol × gap_size ─────────────────────────
        v_time = -v_raw * self._gap

        # ── FFT ─────────────────────────────────────────────────────
        # Apply Hann window to reduce spectral leakage
        window = np.hanning(n_steps)
        v_fft = np.fft.rfft(v_time * window)
        i_fft = np.fft.rfft(i_raw * window)
        freqs = np.fft.rfftfreq(n_steps, d=self._dt)

        # ── Frequency range selection ───────────────────────────────
        if freq_min is not None or freq_max is not None:
            f_lo = freq_min if freq_min is not None else 0.0
            f_hi = freq_max if freq_max is not None else freqs[-1]
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            freqs = freqs[mask]
            v_fft = v_fft[mask]
            i_fft = i_fft[mask]

        # ── Impedance & S₁₁ ────────────────────────────────────────
        # Avoid division by zero where current is negligible
        i_safe = np.where(np.abs(i_fft) > 1e-30, i_fft, 1e-30)
        z_in = v_fft / i_safe
        s11 = (z_in - self._z0) / (z_in + self._z0)

        return SParameterResult(
            frequencies=freqs,
            s11_complex=s11,
            z_in_complex=z_in,
            v_spectrum=v_fft,
            i_spectrum=i_fft,
            dt=self._dt,
            z0=self._z0,
        )
