"""QTT Physics VM — Far-field pattern extraction.

Computes far-field radiation pattern (gain, directivity, efficiency)
from frequency-domain E/B fields via near-to-far-field transformation.

Physics
-------
The equivalence principle converts tangential E and H (=B in normalised
units) on a closed surface S into far-field radiation via:

    N(θ,φ) = ∮_S J_s × r̂  e^{jk r̂·r'} dS'     (electric current)
    L(θ,φ) = ∮_S M_s × r̂  e^{jk r̂·r'} dS'     (magnetic current)

    J_s = n̂ × H     (electric surface current)
    M_s = −n̂ × E    (magnetic surface current)

In the far field:
    E_θ = −jk/(4πr) (L_φ + η₀ N_θ)
    E_φ = +jk/(4πr) (L_θ − η₀ N_φ)

This module evaluates the QTT frequency-domain fields on a box surface
and numerically integrates to get the far-field pattern.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class FarFieldResult:
    """Far-field radiation pattern result.

    All pattern arrays have shape ``(n_theta, n_phi)``.
    """

    theta: NDArray
    """Elevation angles (radians), shape ``(n_theta,)``."""

    phi: NDArray
    """Azimuth angles (radians), shape ``(n_phi,)``."""

    e_theta: NDArray
    """Far-field E_θ component (complex), shape ``(n_theta, n_phi)``."""

    e_phi: NDArray
    """Far-field E_φ component (complex), shape ``(n_theta, n_phi)``."""

    frequency: float
    """Frequency at which the pattern was computed."""

    radiated_power: float
    """Total radiated power (integrated over sphere)."""

    accepted_power: float
    """Power accepted into the antenna (from input current)."""

    @property
    def gain_total(self) -> NDArray:
        """Total gain (linear) = 4π U(θ,φ) / P_rad.

        Shape ``(n_theta, n_phi)``.
        """
        u = self.radiation_intensity
        if self.radiated_power < 1e-30:
            return np.zeros_like(u)
        return 4.0 * np.pi * u / self.radiated_power

    @property
    def gain_total_dbi(self) -> NDArray:
        """Total gain in dBi."""
        g = self.gain_total
        g = np.where(g > 1e-30, g, 1e-30)
        return 10.0 * np.log10(g)

    @property
    def directivity(self) -> NDArray:
        """Directivity = 4π U(θ,φ) / P_rad (same as gain if lossless)."""
        return self.gain_total

    @property
    def radiation_intensity(self) -> NDArray:
        """Radiation intensity U(θ,φ) = r²|E|²/(2η₀).

        At r=1 (normalised far field), U = (|E_θ|² + |E_φ|²) / 2.
        Shape ``(n_theta, n_phi)``.
        """
        return 0.5 * (np.abs(self.e_theta) ** 2 + np.abs(self.e_phi) ** 2)

    @property
    def radiation_efficiency(self) -> float:
        """Radiation efficiency η = P_rad / P_accepted."""
        if self.accepted_power < 1e-30:
            return 0.0
        return self.radiated_power / self.accepted_power

    @property
    def peak_gain_dbi(self) -> float:
        """Peak realised gain in dBi."""
        return float(np.max(self.gain_total_dbi))

    @property
    def peak_directivity_dbi(self) -> float:
        """Peak directivity in dBi."""
        d = self.directivity
        d_max = np.max(d)
        if d_max < 1e-30:
            return -np.inf
        return float(10.0 * np.log10(d_max))

    def pattern_cut(
        self,
        plane: str = "E",
        phi_deg: float = 0.0,
    ) -> tuple[NDArray, NDArray]:
        """Extract a principal-plane pattern cut.

        Parameters
        ----------
        plane : str
            ``"E"`` for E-plane (φ = phi_deg) or ``"H"`` for H-plane
            (φ = phi_deg + 90°).
        phi_deg : float
            Reference azimuth for E-plane.

        Returns
        -------
        theta_deg : NDArray
            Elevation angles in degrees.
        gain_db : NDArray
            Gain in dBi along the cut.
        """
        if plane == "H":
            phi_deg = phi_deg + 90.0
        phi_rad = np.deg2rad(phi_deg) % (2.0 * np.pi)
        # Find nearest phi index
        phi_idx = int(np.argmin(np.abs(self.phi - phi_rad)))
        gain_db = self.gain_total_dbi[:, phi_idx]
        return np.rad2deg(self.theta), gain_db

    def summary(self) -> dict[str, Any]:
        """Summary metrics for reporting."""
        gain_grid = self.gain_total
        peak_idx = np.unravel_index(np.argmax(gain_grid), gain_grid.shape)
        return {
            "peak_gain_dBi": self.peak_gain_dbi,
            "peak_directivity_dBi": self.peak_directivity_dbi,
            "radiation_efficiency": self.radiation_efficiency,
            "radiated_power": self.radiated_power,
            "peak_theta_deg": float(np.rad2deg(self.theta[peak_idx[0]])),
            "peak_phi_deg": float(np.rad2deg(self.phi[peak_idx[1]])),
            "frequency": self.frequency,
        }


class FarFieldExtractor:
    """Extract far-field patterns from DFT-accumulated E/B fields.

    Uses near-to-far-field transformation on a box surface placed
    just inside the simulation domain boundary.

    Parameters
    ----------
    frequency : float
        Frequency at which the DFT fields were accumulated.
    domain_size : float
        Physical domain size (assumed cubic [0, L]³).
    n_surface_samples : int
        Number of sample points per edge on each surface face.
    n_theta : int
        Number of elevation angle samples.
    n_phi : int
        Number of azimuth angle samples.
    surface_margin : float
        Fractional margin from domain boundary for the integration
        surface (e.g. 0.1 = surface at 10% and 90% of domain).
    """

    def __init__(
        self,
        frequency: float = 1.0,
        domain_size: float = 1.0,
        n_surface_samples: int = 32,
        n_theta: int = 91,
        n_phi: int = 72,
        surface_margin: float = 0.1,
    ) -> None:
        self._freq = frequency
        self._L = domain_size
        self._n_surf = n_surface_samples
        self._n_theta = n_theta
        self._n_phi = n_phi
        self._margin = surface_margin

    def extract(
        self,
        dft_fields: dict[str, Any],
        accepted_power: float = 1.0,
    ) -> FarFieldResult:
        """Compute far-field pattern from frequency-domain QTT fields.

        Parameters
        ----------
        dft_fields : dict
            Must contain QTT tensors:
            ``dft_re_Ex``, ``dft_im_Ex``, ..., ``dft_re_Bz``, ``dft_im_Bz``
            These are the real and imaginary parts of the DFT-accumulated
            fields at the target frequency.
        accepted_power : float
            Power accepted into the antenna (from S-parameter data).
            Used for gain computation.

        Returns
        -------
        FarFieldResult
        """
        k = 2.0 * np.pi * self._freq  # wave number (c=1)

        # ── Build complex frequency-domain fields ───────────────────
        # Evaluate on the 6 faces of a box surface
        margin = self._margin * self._L
        lo = margin
        hi = self._L - margin
        n = self._n_surf

        # Surface sample coordinates along each edge
        edge = np.linspace(lo, hi, n)

        # ── Observation angles ──────────────────────────────────────
        theta = np.linspace(0, np.pi, self._n_theta)
        phi = np.linspace(0, 2.0 * np.pi, self._n_phi, endpoint=False)
        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

        # Far-field direction unit vectors
        sin_t = np.sin(THETA)
        cos_t = np.cos(THETA)
        sin_p = np.sin(PHI)
        cos_p = np.cos(PHI)

        # r̂ components
        rx = sin_t * cos_p
        ry = sin_t * sin_p
        rz = cos_t

        # θ̂ components
        tx = cos_t * cos_p
        ty = cos_t * sin_p
        tz = -sin_t

        # φ̂ components
        px = -sin_p
        py = cos_p
        pz = np.zeros_like(THETA)

        # ── Accumulate N and L integrands over 6 faces ──────────────
        # N = ∮ J_s × r̂ e^{jkr̂·r'} dS   (J_s = n̂ × H = n̂ × B)
        # L = ∮ M_s × r̂ e^{jkr̂·r'} dS   (M_s = −n̂ × E)
        #
        # We compute N_θ, N_φ, L_θ, L_φ components directly.
        n_t = self._n_theta
        n_p = self._n_phi
        N_theta = np.zeros((n_t, n_p), dtype=complex)
        N_phi = np.zeros((n_t, n_p), dtype=complex)
        L_theta = np.zeros((n_t, n_p), dtype=complex)
        L_phi = np.zeros((n_t, n_p), dtype=complex)

        dA = ((hi - lo) / (n - 1)) ** 2  # surface element area

        # Evaluate fields at surface sample points.
        # For QTT tensors: use evaluate_at_point.
        # For efficiency, we evaluate all needed points in one pass per face.
        def _eval_complex(
            fields: dict[str, Any],
            component: str,
            point: tuple[float, float, float],
        ) -> complex:
            """Evaluate complex DFT field at a point."""
            re_key = f"dft_re_{component}"
            im_key = f"dft_im_{component}"
            re_field = fields[re_key]
            im_field = fields[im_key]
            re_val = re_field.evaluate_at_point(point)
            im_val = im_field.evaluate_at_point(point)
            return complex(re_val, im_val)

        # Process each face of the integration box
        faces = [
            # (normal_axis, normal_sign, fixed_coord, var1_axis, var2_axis)
            (0, -1, lo, 1, 2),  # x = lo, n̂ = (−1, 0, 0)
            (0, +1, hi, 1, 2),  # x = hi, n̂ = (+1, 0, 0)
            (1, -1, lo, 0, 2),  # y = lo, n̂ = (0, −1, 0)
            (1, +1, hi, 0, 2),  # y = hi, n̂ = (0, +1, 0)
            (2, -1, lo, 0, 1),  # z = lo, n̂ = (0, 0, −1)
            (2, +1, hi, 0, 1),  # z = hi, n̂ = (0, 0, +1)
        ]

        for norm_ax, norm_sign, fixed_val, ax1, ax2 in faces:
            n_hat = np.zeros(3)
            n_hat[norm_ax] = float(norm_sign)

            for i1, c1 in enumerate(edge):
                for i2, c2 in enumerate(edge):
                    # Build 3D coordinate
                    pt = [0.0, 0.0, 0.0]
                    pt[norm_ax] = fixed_val
                    pt[ax1] = c1
                    pt[ax2] = c2
                    pt_tuple = (pt[0], pt[1], pt[2])

                    # Evaluate E and B at this point
                    Ex = _eval_complex(dft_fields, "Ex", pt_tuple)
                    Ey = _eval_complex(dft_fields, "Ey", pt_tuple)
                    Ez = _eval_complex(dft_fields, "Ez", pt_tuple)
                    Bx = _eval_complex(dft_fields, "Bx", pt_tuple)
                    By = _eval_complex(dft_fields, "By", pt_tuple)
                    Bz = _eval_complex(dft_fields, "Bz", pt_tuple)

                    E_vec = np.array([Ex, Ey, Ez])
                    B_vec = np.array([Bx, By, Bz])

                    # Surface currents
                    # J_s = n̂ × B  (H = B in normalised units)
                    J_s = np.cross(n_hat, B_vec)
                    # M_s = −n̂ × E
                    M_s = -np.cross(n_hat, E_vec)

                    # Phase: exp(jk r̂ · r')
                    r_dot_rp = rx * pt[0] + ry * pt[1] + rz * pt[2]
                    phase = np.exp(1j * k * r_dot_rp)

                    # Project currents onto θ̂ and φ̂
                    J_theta = J_s[0] * tx + J_s[1] * ty + J_s[2] * tz
                    J_phi = J_s[0] * px + J_s[1] * py + J_s[2] * pz
                    M_theta = M_s[0] * tx + M_s[1] * ty + M_s[2] * tz
                    M_phi = M_s[0] * px + M_s[1] * py + M_s[2] * pz

                    # Accumulate
                    N_theta += J_theta * phase * dA
                    N_phi += J_phi * phase * dA
                    L_theta += M_theta * phase * dA
                    L_phi += M_phi * phase * dA

        # ── Far-field E ─────────────────────────────────────────────
        # E_θ = −jk/(4π) (L_φ + N_θ)   [η₀ = 1 in normalised units]
        # E_φ = +jk/(4π) (L_θ − N_φ)
        prefactor = -1j * k / (4.0 * np.pi)
        e_theta = prefactor * (L_phi + N_theta)
        e_phi = -prefactor * (L_theta - N_phi)

        # ── Radiated power ──────────────────────────────────────────
        # P_rad = ∮ U(θ,φ) sin(θ) dθ dφ
        U = 0.5 * (np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2)
        d_theta = theta[1] - theta[0] if len(theta) > 1 else np.pi
        d_phi = phi[1] - phi[0] if len(phi) > 1 else 2.0 * np.pi
        SIN_T = np.sin(THETA)
        p_rad = float(np.sum(U * SIN_T * d_theta * d_phi))

        return FarFieldResult(
            theta=theta,
            phi=phi,
            e_theta=e_theta,
            e_phi=e_phi,
            frequency=self._freq,
            radiated_power=p_rad,
            accepted_power=accepted_power,
        )
