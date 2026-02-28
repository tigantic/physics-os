"""QTT Frequency Sweep for 1D EM S-Parameter Analysis.

Phase 5 of the QTT Frequency-Domain Maxwell program.

Implements
----------
1. **Uniform frequency sweep** — S-parameters at evenly spaced
   frequencies in a band [f_min, f_max].
2. **Adaptive frequency sweep** — refines frequency resolution
   near resonances or rapid S-parameter variation using bisection.
3. **Multi-port sweep** — full S-matrix at each frequency point.
4. **Interpolation & post-processing** — rational interpolation,
   bandwidth extraction, resonance detection.
5. **Result container** — ``FrequencySweepResult`` with convenience
   methods for plotting data and quality metrics.

Architecture
------------
At each frequency f (or wavenumber k₀ = 2πf/c), the sweep:
  1. Rebuilds the Helmholtz MPO  H(k₀) = L_s(k₀) + k₀²·diag(ε)
     (the stretched Laplacian depends on k₀ through σ/ω in PML).
  2. Builds the port source for the current k₀.
  3. Solves  H(k₀)·E = -J  via DMRG.
  4. Extracts S-parameters by mode decomposition.
  5. Stores results.

The geometry (ε profile, conductors) is frequency-independent.
Only the Helmholtz operator changes with frequency.

Dependencies
------------
- ``ontic.em.s_parameters``: ``Port``, ``port_source_tt``,
  ``extract_mode_coefficients_lsq``, ``compute_s11``,
  ``SParameterResult``, ``compute_s_matrix_1d``,
  ``solve_and_extract_s11``, ``s_to_db``
- ``ontic.em.boundaries``: ``Geometry1D``, ``helmholtz_mpo_with_bc``,
  ``PMLConfig``
- ``ontic.em.qtt_helmholtz``: ``tt_amen_solve``, ``reconstruct_1d``,
  ``array_to_tt``
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ontic.em.qtt_helmholtz import (
    array_to_tt,
    reconstruct_1d,
    tt_amen_solve,
)
from ontic.em.boundaries import (
    Geometry1D,
    PMLConfig,
    helmholtz_mpo_with_bc,
)
from ontic.em.s_parameters import (
    Port,
    SParameterResult,
    port_source_tt,
    compute_s11,
    compute_s21,
    extract_mode_coefficients_lsq,
    compute_impedance,
    s_to_db,
    fresnel_slab_reflection,
    fresnel_slab_transmission,
)


# =====================================================================
# Section 1: Result Containers
# =====================================================================

@dataclass
class FrequencyPoint:
    """S-parameter data at a single frequency.

    Attributes
    ----------
    k0 : float
        Free-space wavenumber (2πf/c in normalised units).
    frequency_hz : float
        Frequency in Hz.
    S : NDArray
        Complex S-matrix (n_ports × n_ports).
    Z_in : NDArray
        Input impedance at each port (Ω).
    solver_residual : float
        Final DMRG residual for this frequency point.
    solve_time_s : float
        Wall-clock solve time in seconds.
    E_solutions : Optional[list[list[NDArray]]]
        QTT field solutions (stored only if ``store_fields=True``).
    """

    k0: float
    frequency_hz: float
    S: NDArray
    Z_in: NDArray
    solver_residual: float
    solve_time_s: float
    E_solutions: Optional[list[list[NDArray]]] = None


@dataclass
class FrequencySweepResult:
    """Complete frequency sweep results.

    Attributes
    ----------
    points : list[FrequencyPoint]
        Per-frequency S-parameter data (sorted by frequency).
    ports : list[Port]
        Port definitions used.
    geometry_description : str
        Human-readable geometry description.
    sweep_type : str
        "uniform" or "adaptive".
    total_time_s : float
        Total wall-clock time for the sweep.
    """

    points: list[FrequencyPoint] = field(default_factory=list)
    ports: list[Port] = field(default_factory=list)
    geometry_description: str = ""
    sweep_type: str = "uniform"
    total_time_s: float = 0.0

    @property
    def n_points(self) -> int:
        """Number of frequency points."""
        return len(self.points)

    @property
    def k0_array(self) -> NDArray:
        """Wavenumber array (sorted)."""
        return np.array([p.k0 for p in self.points])

    @property
    def frequency_hz_array(self) -> NDArray:
        """Frequency array in Hz (sorted)."""
        return np.array([p.frequency_hz for p in self.points])

    @property
    def n_ports(self) -> int:
        """Number of ports."""
        return len(self.ports)

    def s_parameter(self, i: int, j: int) -> NDArray:
        """Extract S_ij across all frequency points.

        Parameters
        ----------
        i : int
            Output port index (0-based).
        j : int
            Input port index (0-based).

        Returns
        -------
        NDArray
            Complex S_ij array of length n_points.
        """
        return np.array([p.S[i, j] for p in self.points])

    def s_parameter_db(self, i: int, j: int) -> NDArray:
        """Extract |S_ij| in dB across all frequency points.

        Parameters
        ----------
        i : int
            Output port index.
        j : int
            Input port index.

        Returns
        -------
        NDArray
            |S_ij| in dB, array of length n_points.
        """
        s = self.s_parameter(i, j)
        return np.array([s_to_db(v) for v in s])

    def impedance(self, port_idx: int = 0) -> NDArray:
        """Input impedance at a port across frequency.

        Parameters
        ----------
        port_idx : int
            Port index.

        Returns
        -------
        NDArray
            Complex Z_in array of length n_points.
        """
        return np.array([p.Z_in[port_idx] for p in self.points])

    def residuals(self) -> NDArray:
        """Solver residuals across frequency.

        Returns
        -------
        NDArray
            Residual array of length n_points.
        """
        return np.array([p.solver_residual for p in self.points])

    def find_resonances(
        self,
        port_idx: int = 0,
        threshold_db: float = -10.0,
    ) -> list[dict]:
        """Detect resonant frequencies where |S₁₁| dips below threshold.

        Finds local minima of |S₁₁| that fall below ``threshold_db``.

        Parameters
        ----------
        port_idx : int
            Port index for S_ii.
        threshold_db : float
            Detection threshold in dB (default: -10 dB).

        Returns
        -------
        list[dict]
            List of resonance dicts with keys:
            'k0', 'frequency_hz', 's11_db', 'index'.
        """
        s11_db = self.s_parameter_db(port_idx, port_idx)
        resonances: list[dict] = []

        for idx in range(1, len(s11_db) - 1):
            if (s11_db[idx] < s11_db[idx - 1] and
                    s11_db[idx] < s11_db[idx + 1] and
                    s11_db[idx] < threshold_db):
                resonances.append({
                    "k0": self.points[idx].k0,
                    "frequency_hz": self.points[idx].frequency_hz,
                    "s11_db": float(s11_db[idx]),
                    "index": idx,
                })

        return resonances

    def bandwidth_3db(
        self,
        port_idx: int = 0,
        center_k0: Optional[float] = None,
    ) -> Optional[dict]:
        """Compute -3 dB bandwidth around a resonance.

        Finds the frequency range where |S₁₁| < S₁₁_min + 3 dB.

        Parameters
        ----------
        port_idx : int
            Port index for S_ii.
        center_k0 : float, optional
            Center wavenumber of the resonance to analyze.
            If None, uses the global minimum of |S₁₁|.

        Returns
        -------
        Optional[dict]
            Dict with 'k0_lower', 'k0_upper', 'bandwidth_k0',
            'center_k0', 'fractional_bw', 'Q_factor'.
            None if bandwidth cannot be determined.
        """
        s11_db = self.s_parameter_db(port_idx, port_idx)
        k0_arr = self.k0_array

        if center_k0 is not None:
            center_idx = int(np.argmin(np.abs(k0_arr - center_k0)))
        else:
            center_idx = int(np.argmin(s11_db))

        min_db = s11_db[center_idx]
        threshold = min_db + 3.0

        # Search left for -3 dB crossing
        k0_lower = k0_arr[0]
        for idx in range(center_idx, 0, -1):
            if s11_db[idx] >= threshold:
                # Linear interpolation for crossing point
                frac = (threshold - s11_db[idx]) / (
                    s11_db[idx - 1] - s11_db[idx] + 1e-30
                )
                k0_lower = k0_arr[idx] + frac * (k0_arr[idx - 1] - k0_arr[idx])
                break

        # Search right for -3 dB crossing
        k0_upper = k0_arr[-1]
        for idx in range(center_idx, len(s11_db) - 1):
            if s11_db[idx] >= threshold:
                frac = (threshold - s11_db[idx - 1]) / (
                    s11_db[idx] - s11_db[idx - 1] + 1e-30
                )
                k0_upper = k0_arr[idx - 1] + frac * (
                    k0_arr[idx] - k0_arr[idx - 1]
                )
                break

        bw = k0_upper - k0_lower
        center = k0_arr[center_idx]

        if bw <= 0.0 or center <= 0.0:
            return None

        frac_bw = bw / center
        Q = center / bw if bw > 0 else float("inf")

        return {
            "k0_lower": float(k0_lower),
            "k0_upper": float(k0_upper),
            "bandwidth_k0": float(bw),
            "center_k0": float(center),
            "fractional_bw": float(frac_bw),
            "Q_factor": float(Q),
            "min_s11_db": float(min_db),
        }

    def validate_passivity(self, tol: float = 0.01) -> NDArray:
        """Check passivity at every frequency point.

        Returns
        -------
        NDArray
            Boolean array, True where S-matrix is passive.
        """
        from ontic.em.s_parameters import validate_s_matrix_passivity
        return np.array([
            validate_s_matrix_passivity(p.S, tol=tol)
            for p in self.points
        ])

    def max_residual(self) -> float:
        """Maximum solver residual across frequency points."""
        if not self.points:
            return float("inf")
        return float(max(p.solver_residual for p in self.points))


# =====================================================================
# Section 2: Core Sweep Engine
# =====================================================================

def _solve_at_frequency(
    geometry: Geometry1D,
    k0: float,
    ports: list[Port],
    max_rank: int = 128,
    solver_tol: float = 1e-4,
    n_sweeps: int = 40,
    damping: float = 0.01,
    n_probes: int = 8,
    store_fields: bool = False,
    verbose: bool = False,
) -> FrequencyPoint:
    """Solve the Helmholtz equation and extract S-parameters at one k₀.

    Handles single-port (S₁₁ only) and multi-port (full S-matrix)
    cases.

    Parameters
    ----------
    geometry : Geometry1D
        Geometry specification (frequency-independent part).
    k0 : float
        Free-space wavenumber.
    ports : list[Port]
        Port definitions.
    max_rank : int
        Maximum QTT rank for the DMRG solver.
    solver_tol : float
        DMRG convergence tolerance.
    n_sweeps : int
        Number of DMRG sweeps.
    damping : float
        Helmholtz damping regularisation.
    n_probes : int
        Number of probe points for mode extraction.
    store_fields : bool
        If True, store QTT field solutions in the result.
    verbose : bool
        Print solver progress.

    Returns
    -------
    FrequencyPoint
        S-parameter data at this frequency.
    """
    t0 = time.perf_counter()
    n_bits = geometry.n_bits
    N = 2 ** n_bits
    n_ports = len(ports)
    S = np.zeros((n_ports, n_ports), dtype=np.complex128)
    all_residuals: list[float] = []
    all_solutions: list[list[NDArray]] = [] if store_fields else []

    # Build operator H(k₀) — frequency-dependent due to PML stretching
    H = helmholtz_mpo_with_bc(geometry, k=k0, max_rank=max_rank)

    for j, port_j in enumerate(ports):
        # Build source for port j at this k₀
        rhs = port_source_tt(n_bits, k0, port_j, max_rank=max_rank)

        # Solve H·E = -J via DMRG
        result = tt_amen_solve(
            H, rhs,
            max_rank=max_rank,
            n_sweeps=n_sweeps,
            tol=solver_tol,
            verbose=verbose,
        )
        all_residuals.append(result.final_residual)

        if store_fields:
            all_solutions.append(result.x)

        # Dense field for mode extraction
        E_dense = reconstruct_1d(result.x)

        # --- Mode coefficients at excitation port j ---
        eps_j = complex(port_j.eps_r) * (1.0 + 1j * damping)
        k_j = k0 * np.sqrt(eps_j)
        lam_j = 2.0 * math.pi / max(abs(k_j.real), 1e-30)
        span_j = min(lam_j / 2.0, 0.1)

        x_ref_j = port_j.ref_position
        if port_j.direction > 0:
            x_start_j = x_ref_j
            x_end_j = min(x_ref_j + span_j, 0.99)
        else:
            x_start_j = max(x_ref_j - span_j, 0.01)
            x_end_j = x_ref_j
        x_start_j = max(x_start_j, 0.01)

        A_j_fwd, A_j_bwd, _ = extract_mode_coefficients_lsq(
            E_dense, k_j, x_start_j, x_end_j, n_probes=n_probes,
        )

        # Incident / reflected at port j
        if port_j.direction > 0:
            incident_j = A_j_fwd
            reflected_j = A_j_bwd
        else:
            incident_j = A_j_bwd
            reflected_j = A_j_fwd

        if abs(incident_j) > 1e-30:
            S[j, j] = reflected_j / incident_j

        # Transmission to other ports
        for i, port_i in enumerate(ports):
            if i == j:
                continue
            eps_i = complex(port_i.eps_r) * (1.0 + 1j * damping)
            k_i = k0 * np.sqrt(eps_i)
            lam_i = 2.0 * math.pi / max(abs(k_i.real), 1e-30)
            span_i = min(lam_i / 2.0, 0.1)

            x_ref_i = port_i.ref_position
            if port_i.direction > 0:
                x_start_i = x_ref_i
                x_end_i = min(x_ref_i + span_i, 0.99)
            else:
                x_start_i = max(x_ref_i - span_i, 0.01)
                x_end_i = x_ref_i
            x_start_i = max(x_start_i, 0.01)

            A_i_fwd, A_i_bwd, _ = extract_mode_coefficients_lsq(
                E_dense, k_i, x_start_i, x_end_i, n_probes=n_probes,
            )

            if port_i.direction > 0:
                transmitted_i = A_i_fwd
            else:
                transmitted_i = A_i_bwd

            if abs(incident_j) > 1e-30:
                S[i, j] = transmitted_i / incident_j

    # Input impedances
    Z_in = np.array([
        compute_impedance(S[j, j], port_j.z0())
        for j, port_j in enumerate(ports)
    ], dtype=np.complex128)

    # Frequency in Hz (c = 1 in normalised units, so f = k₀/(2π))
    # For real Hz: f = k₀ · c₀ / (2π) where c₀ = 299792458 m/s
    freq_hz = k0 * 299792458.0 / (2.0 * math.pi)

    worst_residual = max(all_residuals) if all_residuals else float("inf")
    dt = time.perf_counter() - t0

    return FrequencyPoint(
        k0=k0,
        frequency_hz=freq_hz,
        S=S,
        Z_in=Z_in,
        solver_residual=worst_residual,
        solve_time_s=dt,
        E_solutions=all_solutions if store_fields else None,
    )


# =====================================================================
# Section 3: Uniform Frequency Sweep
# =====================================================================

def frequency_sweep_uniform(
    geometry: Geometry1D,
    ports: list[Port],
    k0_min: float,
    k0_max: float,
    n_freq: int,
    max_rank: int = 128,
    solver_tol: float = 1e-4,
    n_sweeps: int = 40,
    damping: float = 0.01,
    n_probes: int = 8,
    store_fields: bool = False,
    verbose: bool = True,
    callback: Optional[Callable[[int, int, FrequencyPoint], None]] = None,
) -> FrequencySweepResult:
    """Uniform frequency sweep across [k₀_min, k₀_max].

    Solves the Helmholtz equation and extracts S-parameters at
    ``n_freq`` evenly spaced wavenumber values.

    Parameters
    ----------
    geometry : Geometry1D
        Geometry specification (shared across frequencies).
    ports : list[Port]
        Port definitions (1 or 2 ports).
    k0_min : float
        Minimum free-space wavenumber.
    k0_max : float
        Maximum free-space wavenumber.
    n_freq : int
        Number of frequency points.
    max_rank : int
        Maximum QTT rank for DMRG solver.
    solver_tol : float
        DMRG convergence tolerance.
    n_sweeps : int
        Number of DMRG sweeps.
    damping : float
        Helmholtz operator damping.
    n_probes : int
        Mode extraction probe count.
    store_fields : bool
        Store full QTT field solutions at each frequency.
    verbose : bool
        Print progress.
    callback : callable, optional
        Called as ``callback(i, n_freq, point)`` after each solve.

    Returns
    -------
    FrequencySweepResult
        Complete sweep results.
    """
    if n_freq < 1:
        raise ValueError(f"n_freq must be >= 1, got {n_freq}")
    if k0_min <= 0 or k0_max <= 0:
        raise ValueError(f"k0 must be positive, got [{k0_min}, {k0_max}]")
    if k0_min >= k0_max:
        raise ValueError(f"k0_min={k0_min} must be < k0_max={k0_max}")

    k0_values = np.linspace(k0_min, k0_max, n_freq)

    t_start = time.perf_counter()
    points: list[FrequencyPoint] = []

    if verbose:
        print(f"Uniform frequency sweep: {n_freq} points in "
              f"k₀ ∈ [{k0_min:.4f}, {k0_max:.4f}]")
        print(f"  Geometry: {geometry.n_bits}-bit QTT, "
              f"{len(ports)} port(s), max_rank={max_rank}")

    for idx, k0 in enumerate(k0_values):
        if verbose:
            print(f"  [{idx + 1}/{n_freq}] k₀ = {k0:.6f} ... ", end="",
                  flush=True)

        point = _solve_at_frequency(
            geometry=geometry,
            k0=k0,
            ports=ports,
            max_rank=max_rank,
            solver_tol=solver_tol,
            n_sweeps=n_sweeps,
            damping=damping,
            n_probes=n_probes,
            store_fields=store_fields,
            verbose=False,
        )
        points.append(point)

        if verbose:
            s11_db = s_to_db(point.S[0, 0])
            print(f"|S₁₁| = {s11_db:.1f} dB, "
                  f"res = {point.solver_residual:.2e}, "
                  f"t = {point.solve_time_s:.1f}s")

        if callback is not None:
            callback(idx, n_freq, point)

    total_time = time.perf_counter() - t_start

    if verbose:
        print(f"Sweep complete: {total_time:.1f}s total, "
              f"{total_time / n_freq:.1f}s/point avg")

    return FrequencySweepResult(
        points=points,
        ports=ports,
        geometry_description=_geometry_summary(geometry),
        sweep_type="uniform",
        total_time_s=total_time,
    )


# =====================================================================
# Section 4: Adaptive Frequency Sweep
# =====================================================================

def frequency_sweep_adaptive(
    geometry: Geometry1D,
    ports: list[Port],
    k0_min: float,
    k0_max: float,
    n_initial: int = 5,
    max_points: int = 50,
    refinement_threshold_db: float = 3.0,
    max_rank: int = 128,
    solver_tol: float = 1e-4,
    n_sweeps: int = 40,
    damping: float = 0.01,
    n_probes: int = 8,
    store_fields: bool = False,
    verbose: bool = True,
    callback: Optional[Callable[[int, int, FrequencyPoint], None]] = None,
) -> FrequencySweepResult:
    """Adaptive frequency sweep with automatic refinement.

    Starts with ``n_initial`` uniformly spaced points, then
    iteratively adds points at the midpoint of intervals where
    |S₁₁| changes by more than ``refinement_threshold_db`` between
    adjacent points.

    This concentrates frequency samples near resonances and rapid
    impedance transitions, while using coarse spacing in flat
    regions.

    Parameters
    ----------
    geometry : Geometry1D
        Geometry specification.
    ports : list[Port]
        Port definitions.
    k0_min : float
        Minimum free-space wavenumber.
    k0_max : float
        Maximum free-space wavenumber.
    n_initial : int
        Number of initial uniformly spaced points.
    max_points : int
        Maximum total number of frequency points.
    refinement_threshold_db : float
        Maximum allowed |S₁₁| change (dB) between adjacent
        points before refinement.
    max_rank : int
        Maximum QTT rank.
    solver_tol : float
        DMRG convergence tolerance.
    n_sweeps : int
        Number of DMRG sweeps.
    damping : float
        Helmholtz damping.
    n_probes : int
        Mode extraction probe count.
    store_fields : bool
        Store field solutions.
    verbose : bool
        Print progress.
    callback : callable, optional
        Called after each solve.

    Returns
    -------
    FrequencySweepResult
        Complete adaptive sweep results.
    """
    if n_initial < 2:
        raise ValueError(f"n_initial must be >= 2, got {n_initial}")
    if max_points < n_initial:
        raise ValueError(
            f"max_points={max_points} must be >= n_initial={n_initial}"
        )
    if k0_min <= 0 or k0_max <= 0:
        raise ValueError(f"k0 must be positive, got [{k0_min}, {k0_max}]")
    if k0_min >= k0_max:
        raise ValueError(f"k0_min={k0_min} must be < k0_max={k0_max}")

    t_start = time.perf_counter()

    if verbose:
        print(f"Adaptive frequency sweep: k₀ ∈ [{k0_min:.4f}, {k0_max:.4f}]")
        print(f"  n_initial={n_initial}, max_points={max_points}, "
              f"threshold={refinement_threshold_db:.1f} dB")

    # --- Phase 1: Initial uniform samples ---
    k0_values = np.linspace(k0_min, k0_max, n_initial).tolist()
    solved: dict[float, FrequencyPoint] = {}
    solve_count = 0

    def _solve_k0(k0: float) -> FrequencyPoint:
        nonlocal solve_count
        solve_count += 1
        if verbose:
            print(f"  [{solve_count}] k₀ = {k0:.6f} ... ", end="",
                  flush=True)
        pt = _solve_at_frequency(
            geometry=geometry,
            k0=k0,
            ports=ports,
            max_rank=max_rank,
            solver_tol=solver_tol,
            n_sweeps=n_sweeps,
            damping=damping,
            n_probes=n_probes,
            store_fields=store_fields,
            verbose=False,
        )
        if verbose:
            s11_db = s_to_db(pt.S[0, 0])
            print(f"|S₁₁| = {s11_db:.1f} dB, "
                  f"res = {pt.solver_residual:.2e}, "
                  f"t = {pt.solve_time_s:.1f}s")
        if callback is not None:
            callback(solve_count - 1, max_points, pt)
        return pt

    # Solve initial points
    for k0 in k0_values:
        solved[k0] = _solve_k0(k0)

    # --- Phase 2: Iterative refinement ---
    while solve_count < max_points:
        # Sort by k0
        sorted_k0 = sorted(solved.keys())
        s11_db_vals = [
            s_to_db(solved[k].S[0, 0]) for k in sorted_k0
        ]

        # Find intervals needing refinement
        intervals_to_refine: list[tuple[float, float, float]] = []
        for idx in range(len(sorted_k0) - 1):
            k_lo = sorted_k0[idx]
            k_hi = sorted_k0[idx + 1]
            delta_db = abs(s11_db_vals[idx + 1] - s11_db_vals[idx])
            if delta_db > refinement_threshold_db:
                intervals_to_refine.append((k_lo, k_hi, delta_db))

        if not intervals_to_refine:
            if verbose:
                print(f"  Converged: all intervals within "
                      f"{refinement_threshold_db:.1f} dB threshold")
            break

        # Sort by largest delta first (refine most important intervals)
        intervals_to_refine.sort(key=lambda t: t[2], reverse=True)

        # Refine up to budget
        n_remaining = max_points - solve_count
        n_to_add = min(len(intervals_to_refine), n_remaining)

        if verbose:
            print(f"  Refining {n_to_add} intervals "
                  f"(of {len(intervals_to_refine)} needing refinement)")

        for k_lo, k_hi, delta in intervals_to_refine[:n_to_add]:
            k_mid = 0.5 * (k_lo + k_hi)
            if k_mid not in solved:
                solved[k_mid] = _solve_k0(k_mid)
            if solve_count >= max_points:
                break

    # --- Assemble result (sorted) ---
    sorted_k0 = sorted(solved.keys())
    points = [solved[k] for k in sorted_k0]

    total_time = time.perf_counter() - t_start

    if verbose:
        print(f"Adaptive sweep complete: {solve_count} points, "
              f"{total_time:.1f}s total")

    return FrequencySweepResult(
        points=points,
        ports=ports,
        geometry_description=_geometry_summary(geometry),
        sweep_type="adaptive",
        total_time_s=total_time,
    )


# =====================================================================
# Section 5: Convenience — Single-Port S₁₁ Sweep
# =====================================================================

def s11_sweep(
    geometry: Geometry1D,
    port: Port,
    k0_min: float,
    k0_max: float,
    n_freq: int = 21,
    adaptive: bool = False,
    max_points: int = 50,
    refinement_threshold_db: float = 3.0,
    max_rank: int = 128,
    solver_tol: float = 1e-4,
    n_sweeps: int = 40,
    damping: float = 0.01,
    n_probes: int = 8,
    verbose: bool = True,
) -> FrequencySweepResult:
    """Single-port S₁₁ frequency sweep (convenience wrapper).

    Parameters
    ----------
    geometry : Geometry1D
        Geometry specification.
    port : Port
        Single port definition.
    k0_min : float
        Minimum wavenumber.
    k0_max : float
        Maximum wavenumber.
    n_freq : int
        Number of frequency points (uniform mode).
    adaptive : bool
        Use adaptive refinement if True.
    max_points : int
        Maximum points (adaptive mode).
    refinement_threshold_db : float
        Adaptive refinement threshold.
    max_rank : int
        DMRG max rank.
    solver_tol : float
        DMRG tolerance.
    n_sweeps : int
        DMRG sweeps.
    damping : float
        Helmholtz damping.
    n_probes : int
        Mode extraction probes.
    verbose : bool
        Print progress.

    Returns
    -------
    FrequencySweepResult
        S₁₁ vs frequency.
    """
    if adaptive:
        return frequency_sweep_adaptive(
            geometry=geometry,
            ports=[port],
            k0_min=k0_min,
            k0_max=k0_max,
            n_initial=n_freq,
            max_points=max_points,
            refinement_threshold_db=refinement_threshold_db,
            max_rank=max_rank,
            solver_tol=solver_tol,
            n_sweeps=n_sweeps,
            damping=damping,
            n_probes=n_probes,
            verbose=verbose,
        )
    else:
        return frequency_sweep_uniform(
            geometry=geometry,
            ports=[port],
            k0_min=k0_min,
            k0_max=k0_max,
            n_freq=n_freq,
            max_rank=max_rank,
            solver_tol=solver_tol,
            n_sweeps=n_sweeps,
            damping=damping,
            n_probes=n_probes,
            verbose=verbose,
        )


# =====================================================================
# Section 6: Rational Interpolation
# =====================================================================

def rational_interpolation(
    k0_samples: NDArray,
    s_samples: NDArray,
    k0_eval: NDArray,
    order: int = -1,
) -> NDArray:
    """Interpolate S-parameter data using AAA rational approximation.

    Uses the AAA (Adaptive Antoulas–Anderson) algorithm to build a
    rational approximant of the form:

        S(k₀) ≈ n(k₀) / d(k₀) = Σ wⱼ·fⱼ/(k₀ - zⱼ) / Σ wⱼ/(k₀ - zⱼ)

    where zⱼ are support points, fⱼ are function values at support
    points, and wⱼ are barycentric weights.

    Rational interpolation captures resonance behaviour (poles and
    zeros) far better than polynomial interpolation for EM
    S-parameters.

    Parameters
    ----------
    k0_samples : NDArray
        Wavenumber sample points (length M).
    s_samples : NDArray
        Complex S-parameter values at sample points (length M).
    k0_eval : NDArray
        Wavenumber points at which to evaluate the interpolant.
    order : int
        Maximum approximation order.  If -1, automatically chosen
        to give relative error < 1e-10.

    Returns
    -------
    NDArray
        Interpolated S-parameter values at ``k0_eval``.
    """
    M = len(k0_samples)
    if M < 2:
        raise ValueError(f"Need at least 2 samples, got {M}")

    z = np.array(k0_samples, dtype=np.complex128)
    f = np.array(s_samples, dtype=np.complex128)

    if order < 0:
        order = M

    # AAA algorithm (Nakatsukasa, Sète, Trefethen 2018)
    n_max = min(order, M - 1)

    # All indices are available initially
    J = list(range(M))  # unused indices (test set)
    I: list[int] = []   # support point indices

    # Select first support point: maximum |f| (or median)
    idx0 = int(np.argmax(np.abs(f)))
    I.append(idx0)
    J.remove(idx0)

    weights = np.zeros(M, dtype=np.complex128)

    for m in range(n_max):
        # Current support points
        zI = z[I]
        fI = f[I]

        # Cauchy matrix on test set: C[j, i] = 1/(z_j - z_i) for j in J
        zJ = z[J]
        fJ = f[J]

        if len(I) == 0:
            break

        C = 1.0 / (zJ[:, None] - zI[None, :] + 1e-300)

        # Loewner matrix: L[j, i] = (f_j - f_i) / (z_j - z_i)
        L = (fJ[:, None] - fI[None, :]) * C

        # SVD of Loewner matrix to find weights
        if L.shape[0] == 0 or L.shape[1] == 0:
            break

        _, sigma, Vh = np.linalg.svd(L, full_matrices=False)

        # Weights = last right singular vector
        w = Vh[-1, :].conj()

        # Evaluate rational approximant at test points
        num = C @ (w * fI)
        den = C @ w
        r_J = num / (den + 1e-300)

        # Residual
        err = np.abs(fJ - r_J)

        if np.max(err) < 1e-13 * np.max(np.abs(f)):
            # Converged
            weights_I = w
            break

        # Add point with largest error to support set
        worst = int(np.argmax(err))
        idx_new = J[worst]
        I.append(idx_new)
        J.remove(idx_new)
    else:
        # Used all iterations
        if len(I) > 0:
            zI = z[I]
            fI = f[I]
            zJ = z[J] if J else z[I[:1]]
            fJ = f[J] if J else f[I[:1]]
            C_all = 1.0 / (zJ[:, None] - zI[None, :] + 1e-300)
            L_all = (fJ[:, None] - fI[None, :]) * C_all
            if L_all.shape[0] > 0 and L_all.shape[1] > 0:
                _, _, Vh = np.linalg.svd(L_all, full_matrices=False)
                w = Vh[-1, :].conj()
            else:
                w = np.ones(len(I), dtype=np.complex128)
        else:
            w = np.ones(1, dtype=np.complex128)
        weights_I = w

    # Final support points and weights
    zI = z[I]
    fI = f[I]
    wI = weights_I if 'weights_I' in dir() else w

    # Evaluate at k0_eval using barycentric form
    k_eval = np.array(k0_eval, dtype=np.complex128)
    result = np.empty_like(k_eval)

    for idx, k in enumerate(k_eval):
        diffs = k - zI
        # Handle exact support point matches
        exact_match = np.abs(diffs) < 1e-15
        if np.any(exact_match):
            match_idx = int(np.argmin(np.abs(diffs)))
            result[idx] = fI[match_idx]
        else:
            c = wI / diffs
            result[idx] = np.sum(c * fI) / np.sum(c)

    return result


# =====================================================================
# Section 7: Analytical Reference Sweeps
# =====================================================================

def fresnel_slab_sweep(
    k0_min: float,
    k0_max: float,
    n_freq: int,
    eps_slab: complex,
    thickness: float,
    eps_background: complex = 1.0 + 0j,
) -> tuple[NDArray, NDArray, NDArray]:
    """Analytical Fresnel reflection/transmission vs frequency.

    Computes exact slab reflection and transmission coefficients
    at uniformly spaced wavenumbers for comparison with numerical
    sweep results.

    Parameters
    ----------
    k0_min : float
        Minimum wavenumber.
    k0_max : float
        Maximum wavenumber.
    n_freq : int
        Number of frequency points.
    eps_slab : complex
        Slab relative permittivity.
    thickness : float
        Slab thickness (normalised coordinates).
    eps_background : complex
        Background permittivity.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        (k0_array, R_array, T_array) — wavenumber, reflection,
        transmission coefficient arrays.
    """
    k0_arr = np.linspace(k0_min, k0_max, n_freq)
    R = np.array([
        fresnel_slab_reflection(k, eps_slab, thickness, eps_background)
        for k in k0_arr
    ], dtype=np.complex128)
    T = np.array([
        fresnel_slab_transmission(k, eps_slab, thickness, eps_background)
        for k in k0_arr
    ], dtype=np.complex128)
    return k0_arr, R, T


def quarter_wave_resonance_k0(
    eps_slab: complex,
    thickness: float,
    n_harmonic: int = 1,
) -> float:
    """Wavenumber for quarter-wave resonance of a dielectric slab.

    At the resonant frequency, the slab thickness equals an odd
    multiple of λ_slab/4, giving maximum transmission (minimum S₁₁).

    .. math::

        k_0 = \\frac{(2n-1)\\pi}{2 \\cdot d \\cdot \\sqrt{\\varepsilon_r}}

    Parameters
    ----------
    eps_slab : complex
        Slab permittivity.
    thickness : float
        Slab thickness (normalised).
    n_harmonic : int
        Resonance harmonic number (1 = fundamental, 2 = 3λ/4, …).

    Returns
    -------
    float
        Resonant wavenumber k₀.
    """
    n_slab = np.sqrt(complex(eps_slab))
    return float((2 * n_harmonic - 1) * math.pi / (2.0 * thickness * n_slab.real))


def half_wave_resonance_k0(
    eps_slab: complex,
    thickness: float,
    n_harmonic: int = 1,
) -> float:
    """Wavenumber for half-wave resonance of a dielectric slab.

    At half-wave resonance, the slab thickness equals an integer
    multiple of λ_slab/2, giving **minimum** transmission (zero
    added reflection, Γ passes through zero).

    .. math::

        k_0 = \\frac{n \\pi}{d \\cdot \\sqrt{\\varepsilon_r}}

    Parameters
    ----------
    eps_slab : complex
        Slab permittivity.
    thickness : float
        Slab thickness.
    n_harmonic : int
        Harmonic number (1, 2, 3, …).

    Returns
    -------
    float
        Resonant wavenumber k₀.
    """
    n_slab = np.sqrt(complex(eps_slab))
    return float(n_harmonic * math.pi / (thickness * n_slab.real))


# =====================================================================
# Section 8: Sweep Comparison Utilities
# =====================================================================

def compare_sweep_to_analytical(
    sweep: FrequencySweepResult,
    eps_slab: complex,
    thickness: float,
    port_idx: int = 0,
    eps_background: complex = 1.0 + 0j,
) -> dict:
    """Compare numerical sweep to exact Fresnel slab solution.

    Parameters
    ----------
    sweep : FrequencySweepResult
        Numerical sweep result.
    eps_slab : complex
        Slab permittivity.
    thickness : float
        Slab thickness.
    port_idx : int
        Port index for S₁₁ comparison.
    eps_background : complex
        Background permittivity.

    Returns
    -------
    dict
        Comparison metrics: 'max_error_mag', 'rms_error_mag',
        'max_error_db', 'rms_error_db', 'k0_worst'.
    """
    k0_arr = sweep.k0_array
    s11_num = sweep.s_parameter(port_idx, port_idx)

    s11_ana = np.array([
        fresnel_slab_reflection(k, eps_slab, thickness, eps_background)
        for k in k0_arr
    ], dtype=np.complex128)

    # Magnitude errors
    err_mag = np.abs(np.abs(s11_num) - np.abs(s11_ana))
    rms_mag = float(np.sqrt(np.mean(err_mag ** 2)))
    max_mag = float(np.max(err_mag))
    worst_idx = int(np.argmax(err_mag))

    # dB errors
    db_num = np.array([s_to_db(v) for v in s11_num])
    db_ana = np.array([s_to_db(v) for v in s11_ana])
    err_db = np.abs(db_num - db_ana)
    rms_db = float(np.sqrt(np.mean(err_db ** 2)))
    max_db = float(np.max(err_db))

    return {
        "max_error_mag": max_mag,
        "rms_error_mag": rms_mag,
        "max_error_db": max_db,
        "rms_error_db": rms_db,
        "k0_worst": float(k0_arr[worst_idx]),
        "n_points": len(k0_arr),
    }


# =====================================================================
# Section 9: Internal Helpers
# =====================================================================

def _geometry_summary(geometry: Geometry1D) -> str:
    """Build a human-readable summary of the geometry."""
    parts: list[str] = [f"{geometry.n_bits}-bit QTT (N={2 ** geometry.n_bits})"]
    parts.append(f"ε_bg={geometry.background_eps}")
    if geometry.regions:
        for r in geometry.regions:
            parts.append(f"{r.label or 'region'} ε={r.eps_r} "
                         f"x∈[{r.x_start:.3f},{r.x_end:.3f}]")
    pml = geometry.pml
    parts.append(f"PML: {pml.n_cells} cells, σ_max={pml.sigma_max}")
    if geometry.conductors:
        parts.append(f"PEC: {len(geometry.conductors)} conductor(s)")
    return "; ".join(parts)
