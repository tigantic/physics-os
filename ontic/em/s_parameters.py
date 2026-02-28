"""QTT S-Parameter Extraction for 1D Frequency-Domain EM.

Phase 4 of the QTT Frequency-Domain Maxwell program.

Implements
----------
1. **Port definition** — excitation location, reference plane,
   propagation direction, local impedance.
2. **Port mode source** — localized current source in QTT format
   that launches a wave toward the device-under-test.
3. **Mode decomposition** — extracts forward (A) and backward (B)
   wave amplitudes from the field at a reference plane using
   two-point or multi-point least-squares mode matching.
4. **S-parameter computation** — S₁₁ = B₁/A₁ (reflection) and
   S₂₁ = A₂/A₁ (transmission) for 2-port problems.
5. **Input impedance** — Z_in from S₁₁ and reference impedance Z₀.
6. **Analytical references** — Fresnel slab reflection for
   validation against exact closed-form results.

Physics
-------
In a uniform 1D section with wavenumber ``k``, the electric field
decomposes as

.. math::

    E(x) = A \\, e^{-j k x} \\;+\\; B \\, e^{+j k x}

where *A* is the forward (rightward) amplitude and *B* is the
backward (leftward) amplitude.  The S-parameters are

.. math::

    S_{11} = B_1 / A_1, \\qquad S_{21} = A_2 / A_1

measured at port reference planes.

Mode Matching
-------------
Given *E* at probe points *x₁, x₂, …, x_M*, we solve the
(possibly overdetermined) linear system

.. math::

    \\begin{bmatrix}
    e^{-jkx_1} & e^{+jkx_1} \\\\
    e^{-jkx_2} & e^{+jkx_2} \\\\
    \\vdots & \\vdots
    \\end{bmatrix}
    \\begin{bmatrix} A \\\\ B \\end{bmatrix}
    =
    \\begin{bmatrix} E(x_1) \\\\ E(x_2) \\\\ \\vdots \\end{bmatrix}

via least squares.  Probe spacing is chosen ~λ/4 for numerical
stability (avoids ill-conditioned 2×2 system when kΔx ≈ nπ).

Dependencies
------------
- ``ontic.em.qtt_helmholtz``: ``array_to_tt``, ``reconstruct_1d``,
  ``tt_amen_solve``, ``gaussian_source_tt``
- ``ontic.em.boundaries``: ``Geometry1D``, ``helmholtz_mpo_with_bc``,
  ``PMLConfig``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ontic.em.qtt_helmholtz import (
    array_to_tt,
    reconstruct_1d,
    tt_amen_solve,
)


# =====================================================================
# Section 1: Port Definition
# =====================================================================

@dataclass
class Port:
    """Electromagnetic excitation / measurement port.

    A 1-D port specifies where to inject current and where to
    measure forward / backward wave amplitudes.

    Parameters
    ----------
    position : float
        Source location in normalised coordinates [0, 1].
    ref_position : float
        Reference plane for mode extraction (between source and DUT).
        Must lie in a uniform-material region.
    direction : int
        Propagation direction of the incident wave:
        +1 = rightward (toward increasing x),
        -1 = leftward (toward decreasing x).
    eps_r : complex
        Relative permittivity at the reference section.  Determines
        the local propagation constant ``k_ref = k · sqrt(eps_r)``.
    width : float
        Source Gaussian width σ.  Controls spectral bandwidth and
        QTT rank.  Default 0.02 works well for broadband excitation.
    label : str
        Human-readable port label (e.g. "Port 1").
    """

    position: float = 0.3
    ref_position: float = 0.35
    direction: int = 1
    eps_r: complex = 1.0 + 0j
    width: float = 0.02
    label: str = "Port 1"

    def k_local(self, k0: float) -> complex:
        """Local propagation constant at the reference section.

        Parameters
        ----------
        k0 : float
            Free-space wavenumber (2πf/c).

        Returns
        -------
        complex
            ``k0 * sqrt(eps_r)``  (complex if lossy).
        """
        return k0 * np.sqrt(complex(self.eps_r))

    def wavelength_local(self, k0: float) -> float:
        """Local wavelength at the reference section.

        Returns
        -------
        float
            ``2π / Re(k_local)``.
        """
        k_ref = self.k_local(k0)
        return 2.0 * math.pi / max(abs(k_ref.real), 1e-30)

    def z0(self, eta0: float = 376.73) -> complex:
        """Reference impedance at the port.

        Parameters
        ----------
        eta0 : float
            Free-space impedance (Ω).

        Returns
        -------
        complex
            ``eta0 / sqrt(eps_r)``.
        """
        return eta0 / np.sqrt(complex(self.eps_r))


# =====================================================================
# Section 2: Port Source Construction
# =====================================================================

def port_source_tt(
    n_bits: int,
    k0: float,
    port: Port,
    amplitude: complex = 1.0,
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> list[NDArray]:
    r"""Build port excitation source as QTT cores.

    Creates a smooth localized current source ``J(x)`` centred at
    ``port.position`` that launches a wave toward the DUT.  The
    source is a Gaussian envelope:

    .. math::

        J(x) = \frac{A}{\sqrt{2\pi}\,\sigma}
               \exp\!\left(-\frac{(x - x_0)^2}{2\sigma^2}\right)

    The solver RHS is ``-J`` (sign convention: ∇²E + k²εE = -J).

    Parameters
    ----------
    n_bits : int
        QTT resolution (N = 2^n_bits grid points).
    k0 : float
        Free-space wavenumber.
    port : Port
        Port specification.
    amplitude : complex
        Source amplitude scaling.
    max_rank : int
        Maximum QTT rank for output cores.
    cutoff : float
        SVD truncation threshold.

    Returns
    -------
    list[NDArray]
        Complex QTT cores for ``-J(x)`` (Helmholtz RHS).
    """
    N = 2 ** n_bits
    h = 1.0 / N
    x = np.linspace(h / 2, 1.0 - h / 2, N)

    sigma = port.width
    x0 = port.position

    # Gaussian envelope
    J = amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    J = J / (np.sqrt(2.0 * np.pi) * sigma)

    # Enforce zero at boundaries (inside PML)
    J[0] = 0.0
    J[-1] = 0.0

    # RHS = -J
    rhs = -J.astype(np.complex128)
    return array_to_tt(rhs, max_rank=max_rank, cutoff=cutoff)


# =====================================================================
# Section 3: Mode Decomposition
# =====================================================================

def extract_mode_coefficients(
    E_dense: NDArray,
    k: complex,
    x_probes: NDArray,
) -> tuple[complex, complex]:
    r"""Extract forward/backward wave amplitudes via two-point matching.

    Convention (exp(-jωt) time-harmonic ansatz):
    - Forward (rightward) wave: ``exp(+jkx)``
    - Backward (leftward) wave: ``exp(-jkx)``

    Solves the 2×2 system:

    .. math::

        \begin{bmatrix}
        e^{+jkx_1} & e^{-jkx_1} \\
        e^{+jkx_2} & e^{-jkx_2}
        \end{bmatrix}
        \begin{bmatrix} A_{\text{fwd}} \\ A_{\text{bwd}} \end{bmatrix}
        =
        \begin{bmatrix} E(x_1) \\ E(x_2) \end{bmatrix}

    Parameters
    ----------
    E_dense : NDArray
        Dense electric field array of length N.
    k : complex
        Propagation constant at the reference section.
    x_probes : NDArray
        Exactly 2 probe coordinates in [0, 1].

    Returns
    -------
    tuple[complex, complex]
        (A_fwd, A_bwd) — forward (exp(+jkx)) and backward
        (exp(-jkx)) wave amplitudes.

    Raises
    ------
    ValueError
        If ``|det| < 1e-12`` (probe spacing causes singularity).
    """
    if len(x_probes) != 2:
        raise ValueError(f"Need exactly 2 probe points, got {len(x_probes)}")

    x1, x2 = x_probes[0], x_probes[1]

    # 2×2 system matrix: [exp(+jkx), exp(-jkx)]
    M = np.array([
        [np.exp(1j * k * x1), np.exp(-1j * k * x1)],
        [np.exp(1j * k * x2), np.exp(-1j * k * x2)],
    ], dtype=np.complex128)

    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < 1e-12:
        raise ValueError(
            f"Singular mode matrix: |det|={abs(det):.2e}. "
            f"Spacing k·Δx = {abs(k * (x2 - x1)):.4f} ≈ nπ. "
            f"Choose different probe spacing."
        )

    # Interpolate E at probe locations
    N = len(E_dense)
    h = 1.0 / N
    x_grid = np.linspace(h / 2, 1.0 - h / 2, N)

    E_vals = np.interp(x_probes, x_grid, E_dense.real) + \
             1j * np.interp(x_probes, x_grid, E_dense.imag)

    # Solve via Cramer's rule
    A_fwd = (E_vals[0] * M[1, 1] - E_vals[1] * M[0, 1]) / det
    A_bwd = (E_vals[1] * M[0, 0] - E_vals[0] * M[1, 0]) / det

    return complex(A_fwd), complex(A_bwd)


def extract_mode_coefficients_lsq(
    E_dense: NDArray,
    k: complex,
    x_start: float,
    x_end: float,
    n_probes: int = 8,
) -> tuple[complex, complex, float]:
    r"""Extract forward/backward amplitudes via least-squares fitting.

    Uses *n_probes* evenly spaced points in [x_start, x_end] to
    overdetermine the 2-unknown system.  More robust than two-point
    extraction when the field contains evanescent tails or numerical
    noise.

    Convention (exp(-jωt) time-harmonic ansatz):
    - Forward (rightward) wave: ``exp(+jkx)``
    - Backward (leftward) wave: ``exp(-jkx)``

    Parameters
    ----------
    E_dense : NDArray
        Dense electric field array of length N.
    k : complex
        Propagation constant at the reference section.
    x_start : float
        Start of the fitting region (normalised, [0, 1]).
    x_end : float
        End of the fitting region.
    n_probes : int
        Number of fitting points (≥ 2).

    Returns
    -------
    tuple[complex, complex, float]
        (A_fwd, A_bwd, residual_norm) — forward amplitude (exp(+jkx)),
        backward amplitude (exp(-jkx)), and ‖Mx − e‖₂.
    """
    if n_probes < 2:
        raise ValueError(f"Need at least 2 probes, got {n_probes}")

    N = len(E_dense)
    h = 1.0 / N
    x_grid = np.linspace(h / 2, 1.0 - h / 2, N)

    # Probe locations
    x_probes = np.linspace(x_start, x_end, n_probes)

    # Interpolate E
    E_vals = np.interp(x_probes, x_grid, E_dense.real) + \
             1j * np.interp(x_probes, x_grid, E_dense.imag)

    # Design matrix: [exp(+jkx), exp(-jkx)]  (forward, backward)
    M = np.column_stack([
        np.exp(+1j * k * x_probes),
        np.exp(-1j * k * x_probes),
    ])

    # Least-squares solve
    coeffs, residuals, _, _ = np.linalg.lstsq(M, E_vals, rcond=None)
    A_fwd, A_bwd = complex(coeffs[0]), complex(coeffs[1])

    # Compute residual norm
    res = E_vals - M @ coeffs
    res_norm = float(np.linalg.norm(res))

    return A_fwd, A_bwd, res_norm


# =====================================================================
# Section 4: S-Parameter Computation
# =====================================================================

def compute_s11(
    E_cores: list[NDArray],
    k0: float,
    port: Port,
    damping: float = 0.01,
    n_probes: int = 8,
    probe_span: float = 0.0,
) -> complex:
    """Compute S₁₁ (reflection coefficient) at port reference plane.

    Decomposes the field into forward (exp(+jkx)) and backward
    (exp(-jkx)) waves at the reference section:

    - **direction = +1** (rightward port): incident = A_fwd,
      reflected = A_bwd → S₁₁ = A_bwd / A_fwd.
    - **direction = -1** (leftward port): incident = A_bwd,
      reflected = A_fwd → S₁₁ = A_fwd / A_bwd.

    Parameters
    ----------
    E_cores : list[NDArray]
        Solution QTT cores from Helmholtz solver.
    k0 : float
        Free-space wavenumber.
    port : Port
        Port specification with reference plane position.
    damping : float
        Helmholtz damping parameter (adds small imaginary part to ε).
        Must match the value used in the solver.
    n_probes : int
        Number of fitting points for mode extraction.
    probe_span : float
        Length of fitting region.  If 0, auto-computed as
        ``min(λ/2, 0.1)`` where λ is the local wavelength.

    Returns
    -------
    complex
        S₁₁ at the reference plane.
    """
    E_dense = reconstruct_1d(E_cores)

    # Local k with damping
    eps_local = complex(port.eps_r) * (1.0 + 1j * damping)
    k_ref = k0 * np.sqrt(eps_local)

    # Auto probe span  (~λ/2 but capped)
    if probe_span <= 0.0:
        lam_local = 2.0 * math.pi / max(abs(k_ref.real), 1e-30)
        probe_span = min(lam_local / 2.0, 0.1)

    # Fitting region centred on reference plane
    x_ref = port.ref_position
    if port.direction > 0:
        # Port on the left, ref between source and DUT (to the right of source)
        x_start = x_ref
        x_end = x_ref + probe_span
    else:
        # Port on the right
        x_start = x_ref - probe_span
        x_end = x_ref

    # Clamp to valid domain
    x_start = max(x_start, 0.01)
    x_end = min(x_end, 0.99)

    A_fwd, A_bwd, _ = extract_mode_coefficients_lsq(
        E_dense, k_ref, x_start, x_end, n_probes=n_probes
    )

    # S₁₁ = reflected / incident
    if port.direction > 0:
        # Rightward port: incident=A_fwd(exp+jkx), reflected=A_bwd(exp-jkx)
        incident = A_fwd
        reflected = A_bwd
    else:
        # Leftward port: incident=A_bwd(exp-jkx), reflected=A_fwd(exp+jkx)
        incident = A_bwd
        reflected = A_fwd

    if abs(incident) < 1e-30:
        return complex(0.0)

    return reflected / incident


def compute_s21(
    E_cores: list[NDArray],
    k0: float,
    port_in: Port,
    port_out: Port,
    damping: float = 0.01,
    n_probes: int = 8,
    probe_span: float = 0.0,
) -> complex:
    """Compute S₂₁ (transmission coefficient) between two ports.

    Extracts the incident wave at port_in and the outgoing
    (transmitted) wave at port_out.

    - For port_in (direction=+1), incident is A_fwd (rightward).
    - For port_out (direction=+1), the transmitted wave arrives as
      A_fwd (rightward, having passed through the DUT).

    Parameters
    ----------
    E_cores : list[NDArray]
        Solution QTT cores from Helmholtz solver.
    k0 : float
        Free-space wavenumber.
    port_in : Port
        Input port (source side).
    port_out : Port
        Output port (transmission side).
    damping : float
        Helmholtz damping parameter.
    n_probes : int
        Number of fitting points per port.
    probe_span : float
        Fitting region length (0 = auto).

    Returns
    -------
    complex
        S₂₁ = transmitted / incident.
    """
    E_dense = reconstruct_1d(E_cores)

    # --- Port 1: extract incident amplitude ---
    eps1 = complex(port_in.eps_r) * (1.0 + 1j * damping)
    k1 = k0 * np.sqrt(eps1)
    if probe_span <= 0.0:
        lam1 = 2.0 * math.pi / max(abs(k1.real), 1e-30)
        span1 = min(lam1 / 2.0, 0.1)
    else:
        span1 = probe_span

    x_ref1 = port_in.ref_position
    if port_in.direction > 0:
        x1_start, x1_end = x_ref1, x_ref1 + span1
    else:
        x1_start, x1_end = x_ref1 - span1, x_ref1
    x1_start = max(x1_start, 0.01)
    x1_end = min(x1_end, 0.99)

    A1_fwd, A1_bwd, _ = extract_mode_coefficients_lsq(
        E_dense, k1, x1_start, x1_end, n_probes=n_probes
    )
    # Incident at port_in
    incident = A1_fwd if port_in.direction > 0 else A1_bwd

    # --- Port 2: extract transmitted amplitude ---
    eps2 = complex(port_out.eps_r) * (1.0 + 1j * damping)
    k2 = k0 * np.sqrt(eps2)
    if probe_span <= 0.0:
        lam2 = 2.0 * math.pi / max(abs(k2.real), 1e-30)
        span2 = min(lam2 / 2.0, 0.1)
    else:
        span2 = probe_span

    x_ref2 = port_out.ref_position
    if port_out.direction > 0:
        x2_start, x2_end = x_ref2, x_ref2 + span2
    else:
        x2_start, x2_end = x_ref2 - span2, x_ref2
    x2_start = max(x2_start, 0.01)
    x2_end = min(x2_end, 0.99)

    A2_fwd, A2_bwd, _ = extract_mode_coefficients_lsq(
        E_dense, k2, x2_start, x2_end, n_probes=n_probes
    )
    # Transmitted at port_out: the wave that passed through the DUT
    # For port_in direction=+1 → DUT→right: transmitted = fwd at port_out
    if port_in.direction > 0:
        transmitted = A2_fwd
    else:
        transmitted = A2_bwd

    if abs(incident) < 1e-30:
        return complex(0.0)

    return transmitted / incident


# =====================================================================
# Section 5: S-Matrix (Full 2-Port)
# =====================================================================

@dataclass
class SParameterResult:
    """Result of S-parameter extraction.

    Attributes
    ----------
    S : NDArray
        Complex S-matrix (n_ports × n_ports).
    ports : list[Port]
        Port definitions.
    k0 : float
        Free-space wavenumber used.
    frequency_hz : float
        Frequency in Hz (if known).
    Z_in : NDArray
        Input impedance at each port in Ω (diagonal of Z-matrix).
    solve_residuals : list[float]
        Final residuals from each port excitation solve.
    E_solutions : list[list[NDArray]]
        QTT field solutions for each port excitation.
    """

    S: NDArray
    ports: list[Port]
    k0: float
    frequency_hz: float
    Z_in: NDArray
    solve_residuals: list[float]
    E_solutions: list[list[NDArray]]


def compute_impedance(
    s11: complex,
    z0: complex = 50.0 + 0j,
) -> complex:
    """Input impedance from S₁₁ and reference impedance.

    .. math::

        Z_{\\text{in}} = Z_0 \\, \\frac{1 + S_{11}}{1 - S_{11}}

    Parameters
    ----------
    s11 : complex
        Reflection coefficient at the port.
    z0 : complex
        Reference impedance (default 50 Ω).

    Returns
    -------
    complex
        Input impedance.
    """
    denom = 1.0 - s11
    if abs(denom) < 1e-30:
        return complex(float("inf"))
    return z0 * (1.0 + s11) / denom


def compute_s_matrix_1d(
    geometry_builder,
    k0: float,
    ports: list[Port],
    max_rank: int = 128,
    solver_tol: float = 1e-4,
    n_sweeps: int = 40,
    damping: float = 0.01,
    n_probes: int = 8,
    verbose: bool = True,
) -> SParameterResult:
    """Compute full n-port S-matrix for a 1D geometry.

    For each port *j*, excites a source at port *j* and extracts
    the mode coefficients at all ports.  Requires *n* solves for
    an n-port network.

    Parameters
    ----------
    geometry_builder : callable
        Function ``(port: Port) -> Geometry1D`` that returns the
        geometry with source configuration for the given port.
        The geometry should include PML and all materials.
    k0 : float
        Free-space wavenumber.
    ports : list[Port]
        List of port definitions.
    max_rank : int
        Maximum QTT rank for solver.
    solver_tol : float
        Solver convergence tolerance.
    n_sweeps : int
        Number of DMRG sweeps.
    damping : float
        Helmholtz damping regularisation.
    n_probes : int
        Number of probe points for mode extraction.
    verbose : bool
        Print progress.

    Returns
    -------
    SParameterResult
    """
    from ontic.em.boundaries import helmholtz_mpo_with_bc

    n_ports = len(ports)
    S = np.zeros((n_ports, n_ports), dtype=np.complex128)
    residuals: list[float] = []
    solutions: list[list[NDArray]] = []

    for j, port_j in enumerate(ports):
        if verbose:
            print(f"  Exciting {port_j.label} at x={port_j.position:.3f}")

        # Build geometry for this excitation
        geo = geometry_builder(port_j)
        n_bits = geo.n_bits
        N = 2 ** n_bits
        h = 1.0 / N

        # Build operator
        H = helmholtz_mpo_with_bc(geo, k=k0, max_rank=max_rank)

        # Build source for port j
        rhs = port_source_tt(
            n_bits, k0, port_j,
            max_rank=max_rank,
        )

        # Solve
        result = tt_amen_solve(
            H, rhs,
            max_rank=max_rank,
            n_sweeps=n_sweeps,
            tol=solver_tol,
            verbose=False,
        )

        if verbose:
            status = "✓" if result.converged else "✗"
            print(f"    {status} residual={result.final_residual:.2e}")

        residuals.append(result.final_residual)
        solutions.append(result.x)

        # Extract E for mode matching
        E_dense = reconstruct_1d(result.x)

        # Mode coefficients at excitation port j
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
            E_dense, k_j, x_start_j, x_end_j, n_probes=n_probes
        )

        # Incident amplitude at excitation port j
        if port_j.direction > 0:
            incident_j = A_j_fwd
            reflected_j = A_j_bwd
        else:
            incident_j = A_j_bwd
            reflected_j = A_j_fwd

        # S_jj = reflection at port j
        if abs(incident_j) > 1e-30:
            S[j, j] = reflected_j / incident_j

        # S_ij = transmission from port j to port i
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
                E_dense, k_i, x_start_i, x_end_i, n_probes=n_probes
            )

            # Transmitted wave at port i (outgoing from DUT)
            if port_j.direction > 0:
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

    freq_hz = k0 * 299792458.0 / (2.0 * math.pi)

    return SParameterResult(
        S=S,
        ports=ports,
        k0=k0,
        frequency_hz=freq_hz,
        Z_in=Z_in,
        solve_residuals=residuals,
        E_solutions=solutions,
    )


# =====================================================================
# Section 6: Single-Shot S₁₁ Extraction (convenience)
# =====================================================================

def solve_and_extract_s11(
    geometry,
    k0: float,
    port: Port,
    max_rank: int = 128,
    solver_tol: float = 1e-4,
    n_sweeps: int = 40,
    damping: float = 0.01,
    n_probes: int = 8,
    verbose: bool = True,
) -> tuple[complex, "NDArray", float]:
    """One-shot: solve Helmholtz and extract S₁₁ at a single port.

    Convenience function that assembles MPO, solves, and returns
    the reflection coefficient in one call.

    Parameters
    ----------
    geometry : Geometry1D
        Full geometry with PML and materials.
    k0 : float
        Free-space wavenumber.
    port : Port
        Port specification.
    max_rank : int
        Solver max QTT rank.
    solver_tol : float
        DMRG convergence tolerance.
    n_sweeps : int
        Number of DMRG sweeps.
    damping : float
        Helmholtz damping parameter (must match geometry PML damping).
    n_probes : int
        Mode extraction probe count.
    verbose : bool
        Print solver output.

    Returns
    -------
    tuple[complex, NDArray, float]
        (S₁₁, E_dense, solver_residual).
    """
    from ontic.em.boundaries import helmholtz_mpo_with_bc

    n_bits = geometry.n_bits
    N = 2 ** n_bits
    h = 1.0 / N

    # Build operator
    H = helmholtz_mpo_with_bc(geometry, k=k0, max_rank=max_rank)

    # Build source
    rhs = port_source_tt(n_bits, k0, port, max_rank=max_rank)

    # Solve via DMRG
    result = tt_amen_solve(
        H, rhs,
        max_rank=max_rank,
        n_sweeps=n_sweeps,
        tol=solver_tol,
        verbose=verbose,
    )

    # Extract S₁₁
    s11 = compute_s11(
        result.x, k0, port,
        damping=damping, n_probes=n_probes,
    )

    E_dense = reconstruct_1d(result.x)

    return s11, E_dense, result.final_residual


# =====================================================================
# Section 7: Analytical Reference (Fresnel Slab)
# =====================================================================

def fresnel_slab_reflection(
    k0: float,
    eps_slab: complex,
    thickness: float,
    eps_background: complex = 1.0 + 0j,
) -> complex:
    """Exact Fresnel reflection for a dielectric slab at normal incidence.

    For a slab of permittivity ``eps_slab`` and thickness ``d``
    embedded in a background medium:

    .. math::

        \\Gamma = r_{12}\\,
        \\frac{1 - e^{-2j\\beta d}}
             {1 - r_{12}^2 \\, e^{-2j\\beta d}}

    where ``r₁₂ = (n₁ - n₂)/(n₁ + n₂)`` and ``β = k₀ n₂``.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber.
    eps_slab : complex
        Slab permittivity.
    thickness : float
        Slab thickness (normalised coordinates).
    eps_background : complex
        Background permittivity.

    Returns
    -------
    complex
        Fresnel reflection coefficient Γ.
    """
    n1 = np.sqrt(complex(eps_background))
    n2 = np.sqrt(complex(eps_slab))
    r12 = (n1 - n2) / (n1 + n2)

    beta = k0 * n2
    phase = np.exp(-2j * beta * thickness)

    gamma = r12 * (1.0 - phase) / (1.0 - r12 ** 2 * phase)
    return complex(gamma)


def fresnel_slab_transmission(
    k0: float,
    eps_slab: complex,
    thickness: float,
    eps_background: complex = 1.0 + 0j,
) -> complex:
    """Exact Fresnel transmission for a dielectric slab at normal incidence.

    .. math::

        T = \\frac{t_{12}\\,t_{21}\\,e^{-j\\beta d}}
                  {1 - r_{12}^2\\,e^{-2j\\beta d}}

    Parameters
    ----------
    k0 : float
        Free-space wavenumber.
    eps_slab : complex
        Slab permittivity.
    thickness : float
        Slab thickness.
    eps_background : complex
        Background permittivity.

    Returns
    -------
    complex
        Fresnel transmission coefficient T.
    """
    n1 = np.sqrt(complex(eps_background))
    n2 = np.sqrt(complex(eps_slab))
    r12 = (n1 - n2) / (n1 + n2)
    t12 = 1.0 + r12
    t21 = 1.0 - r12

    beta = k0 * n2
    single_pass = np.exp(-1j * beta * thickness)
    denom = 1.0 - r12 ** 2 * np.exp(-2j * beta * thickness)

    return complex(t12 * t21 * single_pass / denom)


# =====================================================================
# Section 8: Utility / Diagnostic Functions
# =====================================================================

def s_to_db(s: complex) -> float:
    """Convert S-parameter to dB (20·log₁₀|S|).

    Parameters
    ----------
    s : complex
        S-parameter value.

    Returns
    -------
    float
        Magnitude in dB.
    """
    return 20.0 * math.log10(max(abs(s), 1e-30))


def s_to_vswr(s11: complex) -> float:
    """Convert S₁₁ to Voltage Standing Wave Ratio.

    .. math::

        \\text{VSWR} = \\frac{1 + |S_{11}|}{1 - |S_{11}|}

    Parameters
    ----------
    s11 : complex
        Reflection coefficient.

    Returns
    -------
    float
        VSWR ∈ [1, ∞).
    """
    rho = abs(s11)
    if rho >= 1.0:
        return float("inf")
    return (1.0 + rho) / (1.0 - rho)


def return_loss(s11: complex) -> float:
    """Return loss in dB (= -|S₁₁|_dB).

    Parameters
    ----------
    s11 : complex
        Reflection coefficient.

    Returns
    -------
    float
        Return loss ≥ 0 (higher = better matching).
    """
    return -s_to_db(s11)


def validate_s_matrix_passivity(S: NDArray, tol: float = 0.01) -> bool:
    """Check that S-matrix satisfies passivity: I - S^H·S ≥ 0.

    For a lossless network, S should be unitary (I - S^H·S = 0).
    For a lossy network, I - S^H·S should be positive semi-definite.

    Parameters
    ----------
    S : NDArray
        Complex S-matrix (n × n).
    tol : float
        Tolerance for eigenvalue positivity check.

    Returns
    -------
    bool
        True if S-matrix is passive.
    """
    n = S.shape[0]
    I = np.eye(n, dtype=np.complex128)
    residual = I - S.conj().T @ S
    eigenvalues = np.linalg.eigvalsh(residual)
    return bool(np.all(eigenvalues > -tol))


def validate_s_matrix_reciprocity(S: NDArray, tol: float = 0.01) -> bool:
    """Check that S-matrix is symmetric (reciprocal network).

    For reciprocal media (no magneto-optic effects), S = S^T.

    Parameters
    ----------
    S : NDArray
        Complex S-matrix (n × n).
    tol : float
        Maximum allowed |S_ij - S_ji|.

    Returns
    -------
    bool
        True if S is symmetric within tolerance.
    """
    return bool(np.allclose(S, S.T, atol=tol))
