"""
Physical optics — diffraction, polarization, beam propagation.

Upgrades domain IV.1 from EUV-only (euv_quantum_well_resist.py) to
general physical optics with:
  - Fresnel diffraction (chirp-z / convolution)
  - Fraunhofer (far-field) diffraction via FFT
  - Angular spectrum propagation (exact within paraxial)
  - Jones vector/matrix polarization calculus
  - Mueller/Stokes polarization calculus
  - Thin-film multilayer (transfer matrix method)
  - Gaussian beam propagation (ABCD matrix)

All in SI: wavelength λ [m], distances [m], electric field [V/m].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_LIGHT: float = 299_792_458.0          # m/s
H_PLANCK: float = 6.62607015e-34        # J·s
EPSILON_0: float = 8.854187817e-12      # F/m
MU_0: float = 1.2566370614e-6           # H/m


# ===================================================================
#  Scalar Diffraction Propagators
# ===================================================================

def _make_grid(N: int, dx: float) -> Tuple[NDArray, NDArray]:
    """Generate centred 2D coordinate grid."""
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    return X, Y


class FresnelPropagator:
    r"""
    Fresnel (near-field) scalar diffraction propagation.

    Fresnel diffraction integral (convolution form):
    $$
    U(x,y,z) = \frac{e^{ikz}}{i\lambda z}
        \iint U_0(x',y') \exp\!\left[
            \frac{i\pi}{\lambda z}\left((x-x')^2+(y-y')^2\right)
        \right] dx'\,dy'
    $$

    Implemented as multiplication in Fourier space (transfer function):
    $$
    H(f_x,f_y) = \exp\!\left[i\pi\lambda z(f_x^2+f_y^2)\right]
    $$
    when $\lambda z f_{\max}^2 \ll 1$; otherwise uses single-FFT chirp method.
    """

    def __init__(self, wavelength: float, grid_size: int, pixel_pitch: float) -> None:
        """
        Parameters
        ----------
        wavelength : Wavelength λ [m].
        grid_size : Number of samples N (must be power of 2 for FFT efficiency).
        pixel_pitch : Spatial sampling Δx [m].
        """
        self.lam = wavelength
        self.k = 2.0 * np.pi / wavelength
        self.N = grid_size
        self.dx = pixel_pitch

        # Frequency grid
        fx = np.fft.fftfreq(grid_size, d=pixel_pitch)
        FX, FY = np.meshgrid(fx, fx, indexing="ij")
        self._freq_r2 = FX**2 + FY**2

    def propagate(self, U0: NDArray[np.complex128],
                  z: float) -> NDArray[np.complex128]:
        """
        Propagate scalar field U0 a distance z using transfer function method.

        Parameters
        ----------
        U0 : (N, N) complex input field.
        z : Propagation distance [m].

        Returns
        -------
        U : (N, N) propagated complex field.
        """
        H = np.exp(1j * np.pi * self.lam * z * self._freq_r2)
        # Global phase
        H *= np.exp(1j * self.k * z)
        U_hat = np.fft.fft2(U0)
        return np.fft.ifft2(U_hat * H)

    def propagate_chirp(self, U0: NDArray[np.complex128],
                        z: float) -> Tuple[NDArray[np.complex128], float]:
        """
        Single-FFT chirp-z Fresnel propagation.

        Returns (U, dx_out) where dx_out is the output pixel pitch,
        which differs from input pitch: dx_out = λz / (N·dx_in).
        """
        X, Y = _make_grid(self.N, self.dx)
        # Quadratic phase in input plane
        phase_in = np.exp(1j * self.k / (2.0 * z) * (X**2 + Y**2))
        U_prod = U0 * phase_in

        # FFT gives output on frequency grid
        U_hat = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_prod)))

        dx_out = self.lam * z / (self.N * self.dx)
        Xo, Yo = _make_grid(self.N, dx_out)
        phase_out = np.exp(1j * self.k / (2.0 * z) * (Xo**2 + Yo**2))
        prefactor = np.exp(1j * self.k * z) / (1j * self.lam * z) * self.dx**2
        U = prefactor * phase_out * U_hat

        return U, dx_out


class FraunhoferPropagator:
    r"""
    Fraunhofer (far-field) diffraction.

    $$
    U(x,y) = \frac{e^{ikz}}{i\lambda z}
        \exp\!\left(\frac{ik}{2z}(x^2+y^2)\right)
        \tilde{U}_0\!\left(\frac{x}{\lambda z},\frac{y}{\lambda z}\right)
    $$

    Valid when Fresnel number $N_F = a^2/\lambda z \ll 1$.
    """

    def __init__(self, wavelength: float, grid_size: int,
                 pixel_pitch: float) -> None:
        self.lam = wavelength
        self.k = 2.0 * np.pi / wavelength
        self.N = grid_size
        self.dx = pixel_pitch

    def propagate(self, U0: NDArray[np.complex128],
                  z: float) -> Tuple[NDArray[np.complex128], float]:
        """
        Far-field diffraction pattern.

        Returns
        -------
        U : (N, N) far-field complex amplitude.
        dx_out : Output plane pixel pitch [m].
        """
        dx_out = self.lam * z / (self.N * self.dx)
        Xo, Yo = _make_grid(self.N, dx_out)

        U_hat = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U0)))
        phase = np.exp(1j * self.k * z) * np.exp(
            1j * self.k / (2.0 * z) * (Xo**2 + Yo**2))
        prefactor = self.dx**2 / (1j * self.lam * z)
        U = prefactor * phase * U_hat

        return U, dx_out

    @staticmethod
    def fresnel_number(aperture_radius: float, wavelength: float,
                       distance: float) -> float:
        """Fresnel number N_F = a² / (λz)."""
        return aperture_radius**2 / (wavelength * distance)


class AngularSpectrumPropagator:
    r"""
    Angular Spectrum Method (ASM) — exact within scalar diffraction.

    Transfer function:
    $$
    H(f_x,f_y;z) = \exp\!\left[
        i 2\pi z \sqrt{1/\lambda^2 - f_x^2 - f_y^2}
    \right]
    $$

    Evanescent waves ($f_x^2 + f_y^2 > 1/\lambda^2$) are damped.
    No paraxial approximation — exact for all propagating angles.
    """

    def __init__(self, wavelength: float, grid_size: int,
                 pixel_pitch: float) -> None:
        self.lam = wavelength
        self.k = 2.0 * np.pi / wavelength
        self.N = grid_size
        self.dx = pixel_pitch

        fx = np.fft.fftfreq(grid_size, d=pixel_pitch)
        FX, FY = np.meshgrid(fx, fx, indexing="ij")
        self._freq_r2 = FX**2 + FY**2
        self._max_freq2 = 1.0 / wavelength**2

    def propagate(self, U0: NDArray[np.complex128],
                  z: float) -> NDArray[np.complex128]:
        """
        Propagate field U0 distance z via angular spectrum.

        Parameters
        ----------
        U0 : (N, N) complex input field.
        z : Propagation distance [m] (positive = forward).

        Returns
        -------
        U : (N, N) propagated field (same grid).
        """
        prop_mask = self._freq_r2 <= self._max_freq2
        kz = np.zeros_like(self._freq_r2)
        kz[prop_mask] = 2.0 * np.pi * np.sqrt(
            self._max_freq2 - self._freq_r2[prop_mask])

        # Evanescent decay
        evan_mask = ~prop_mask
        kz_evan = 2.0 * np.pi * np.sqrt(
            self._freq_r2[evan_mask] - self._max_freq2)

        H = np.zeros_like(self._freq_r2, dtype=np.complex128)
        H[prop_mask] = np.exp(1j * kz[prop_mask] * z)
        H[evan_mask] = np.exp(-kz_evan * abs(z))

        U_hat = np.fft.fft2(U0)
        return np.fft.ifft2(U_hat * H)

    def tilt(self, U0: NDArray[np.complex128],
             theta_x: float, theta_y: float) -> NDArray[np.complex128]:
        """
        Apply plane-wave tilt to field (steer beam by angles θx, θy).

        Parameters
        ----------
        theta_x, theta_y : Tilt angles [radians].
        """
        X, Y = _make_grid(self.N, self.dx)
        kx = self.k * np.sin(theta_x)
        ky = self.k * np.sin(theta_y)
        return U0 * np.exp(1j * (kx * X + ky * Y))


# ===================================================================
#  Jones Polarization Calculus
# ===================================================================

class JonesVector:
    r"""
    Jones vector for fully polarised light.

    $$\mathbf{E} = \begin{pmatrix} E_x \\ E_y \end{pmatrix} = E_0 \begin{pmatrix} \cos\alpha \\ e^{i\delta}\sin\alpha \end{pmatrix}$$
    """

    __slots__ = ("_vec",)

    def __init__(self, ex: complex, ey: complex) -> None:
        self._vec = np.array([ex, ey], dtype=np.complex128)

    @property
    def vec(self) -> NDArray[np.complex128]:
        return self._vec.copy()

    @property
    def intensity(self) -> float:
        """I = |E_x|² + |E_y|²."""
        return float(np.real(np.conj(self._vec) @ self._vec))

    def normalised(self) -> "JonesVector":
        """Unit-amplitude Jones vector."""
        amp = np.sqrt(self.intensity)
        if amp < 1e-30:
            return JonesVector(0.0, 0.0)
        v = self._vec / amp
        return JonesVector(v[0], v[1])

    def to_stokes(self) -> "StokesVector":
        """Convert to Stokes parameters."""
        Ex, Ey = self._vec
        S0 = float(np.abs(Ex)**2 + np.abs(Ey)**2)
        S1 = float(np.abs(Ex)**2 - np.abs(Ey)**2)
        S2 = float(2.0 * np.real(Ex * np.conj(Ey)))
        S3 = float(2.0 * np.imag(Ex * np.conj(Ey)))
        return StokesVector(S0, S1, S2, S3)

    # Factory methods for standard polarisation states
    @staticmethod
    def horizontal() -> "JonesVector":
        return JonesVector(1.0, 0.0)

    @staticmethod
    def vertical() -> "JonesVector":
        return JonesVector(0.0, 1.0)

    @staticmethod
    def diagonal() -> "JonesVector":
        s = 1.0 / math.sqrt(2.0)
        return JonesVector(s, s)

    @staticmethod
    def antidiagonal() -> "JonesVector":
        s = 1.0 / math.sqrt(2.0)
        return JonesVector(s, -s)

    @staticmethod
    def right_circular() -> "JonesVector":
        s = 1.0 / math.sqrt(2.0)
        return JonesVector(s, -1j * s)

    @staticmethod
    def left_circular() -> "JonesVector":
        s = 1.0 / math.sqrt(2.0)
        return JonesVector(s, 1j * s)

    @staticmethod
    def elliptical(alpha: float, delta: float,
                   amplitude: float = 1.0) -> "JonesVector":
        """
        Elliptical polarisation.

        Parameters
        ----------
        alpha : Orientation angle [rad] (tan α = Ey/Ex amplitude ratio).
        delta : Phase difference δ [rad].
        amplitude : Overall amplitude.
        """
        return JonesVector(
            amplitude * math.cos(alpha),
            amplitude * math.sin(alpha) * np.exp(1j * delta),
        )


class JonesMatrix:
    r"""
    2×2 Jones matrix for linear optical elements.

    Action: $\mathbf{E}_{\text{out}} = \mathbf{J}\,\mathbf{E}_{\text{in}}$

    Standard matrices:
    - Linear polariser at angle θ
    - Quarter/half-wave plate at angle θ
    - Rotator (Faraday, optical activity)
    - General phase retarder
    """

    __slots__ = ("_mat",)

    def __init__(self, matrix: NDArray[np.complex128]) -> None:
        self._mat = np.array(matrix, dtype=np.complex128).reshape(2, 2)

    @property
    def mat(self) -> NDArray[np.complex128]:
        return self._mat.copy()

    def apply(self, jones: JonesVector) -> JonesVector:
        """Apply element to Jones vector."""
        v = self._mat @ jones.vec
        return JonesVector(v[0], v[1])

    def __matmul__(self, other: Union["JonesMatrix", JonesVector]) -> Union["JonesMatrix", JonesVector]:
        if isinstance(other, JonesVector):
            return self.apply(other)
        elif isinstance(other, JonesMatrix):
            return JonesMatrix(self._mat @ other._mat)
        return NotImplemented

    @property
    def transmission(self) -> float:
        """Average power transmission (eigenvalue-based)."""
        return float(np.real(np.trace(self._mat.conj().T @ self._mat)) / 2.0)

    # Factory: standard optical elements

    @staticmethod
    def linear_polariser(theta: float = 0.0) -> "JonesMatrix":
        """Linear polariser at angle θ from horizontal."""
        c, s = math.cos(theta), math.sin(theta)
        M = np.array([[c * c, c * s], [c * s, s * s]], dtype=np.complex128)
        return JonesMatrix(M)

    @staticmethod
    def half_wave_plate(theta: float = 0.0) -> "JonesMatrix":
        """Half-wave plate (retardance π) with fast axis at angle θ."""
        return JonesMatrix._retarder(np.pi, theta)

    @staticmethod
    def quarter_wave_plate(theta: float = 0.0) -> "JonesMatrix":
        """Quarter-wave plate (retardance π/2) with fast axis at angle θ."""
        return JonesMatrix._retarder(np.pi / 2.0, theta)

    @staticmethod
    def phase_retarder(delta: float, theta: float = 0.0) -> "JonesMatrix":
        """General phase retarder with retardance δ, fast axis at θ."""
        return JonesMatrix._retarder(delta, theta)

    @staticmethod
    def _retarder(delta: float, theta: float) -> "JonesMatrix":
        c, s = math.cos(theta), math.sin(theta)
        ed = np.exp(1j * delta / 2.0)
        edn = np.exp(-1j * delta / 2.0)
        M = np.array([
            [c * c * edn + s * s * ed, c * s * (edn - ed)],
            [c * s * (edn - ed), s * s * edn + c * c * ed],
        ], dtype=np.complex128)
        return JonesMatrix(M)

    @staticmethod
    def rotator(theta: float) -> "JonesMatrix":
        """Polarisation rotator (Faraday / optical activity) by angle θ."""
        c, s = math.cos(theta), math.sin(theta)
        return JonesMatrix(np.array([[c, -s], [s, c]], dtype=np.complex128))

    @staticmethod
    def identity() -> "JonesMatrix":
        return JonesMatrix(np.eye(2, dtype=np.complex128))


# ===================================================================
#  Stokes / Mueller Polarization Calculus
# ===================================================================

class StokesVector:
    r"""
    Stokes vector for partially polarised light.

    $$\mathbf{S} = \begin{pmatrix} S_0 \\ S_1 \\ S_2 \\ S_3 \end{pmatrix}
      = \begin{pmatrix} I \\ I_H - I_V \\ I_D - I_A \\ I_R - I_L \end{pmatrix}$$

    Degree of polarisation (DOP): $\text{DOP} = \sqrt{S_1^2+S_2^2+S_3^2}/S_0$
    """

    __slots__ = ("_vec",)

    def __init__(self, S0: float, S1: float, S2: float, S3: float) -> None:
        self._vec = np.array([S0, S1, S2, S3], dtype=np.float64)

    @property
    def vec(self) -> NDArray[np.float64]:
        return self._vec.copy()

    @property
    def intensity(self) -> float:
        return float(self._vec[0])

    @property
    def dop(self) -> float:
        """Degree of polarisation."""
        S0 = self._vec[0]
        if S0 < 1e-30:
            return 0.0
        return float(np.sqrt(np.sum(self._vec[1:]**2)) / S0)

    @property
    def dolp(self) -> float:
        """Degree of linear polarisation."""
        S0 = self._vec[0]
        if S0 < 1e-30:
            return 0.0
        return float(np.sqrt(self._vec[1]**2 + self._vec[2]**2) / S0)

    @property
    def docp(self) -> float:
        """Degree of circular polarisation."""
        S0 = self._vec[0]
        if S0 < 1e-30:
            return 0.0
        return float(abs(self._vec[3]) / S0)

    @property
    def orientation_angle(self) -> float:
        """Polarisation ellipse orientation angle ψ [rad]."""
        return 0.5 * math.atan2(self._vec[2], self._vec[1])

    @property
    def ellipticity_angle(self) -> float:
        """Polarisation ellipse ellipticity angle χ [rad]."""
        dolp = np.sqrt(self._vec[1]**2 + self._vec[2]**2)
        return 0.5 * math.atan2(self._vec[3], dolp)

    @staticmethod
    def unpolarised(intensity: float = 1.0) -> "StokesVector":
        return StokesVector(intensity, 0.0, 0.0, 0.0)

    @staticmethod
    def horizontal(intensity: float = 1.0) -> "StokesVector":
        return StokesVector(intensity, intensity, 0.0, 0.0)

    @staticmethod
    def right_circular(intensity: float = 1.0) -> "StokesVector":
        return StokesVector(intensity, 0.0, 0.0, intensity)


class MuellerMatrix:
    r"""
    4×4 Mueller matrix for transformation of Stokes vectors.

    $$\mathbf{S}_{\text{out}} = \mathbf{M}\,\mathbf{S}_{\text{in}}$$

    Includes standard optical elements and depolarisers.
    """

    __slots__ = ("_mat",)

    def __init__(self, matrix: NDArray[np.float64]) -> None:
        self._mat = np.array(matrix, dtype=np.float64).reshape(4, 4)

    @property
    def mat(self) -> NDArray[np.float64]:
        return self._mat.copy()

    def apply(self, stokes: StokesVector) -> StokesVector:
        v = self._mat @ stokes.vec
        return StokesVector(v[0], v[1], v[2], v[3])

    def __matmul__(self, other: Union["MuellerMatrix", StokesVector]) -> Union["MuellerMatrix", StokesVector]:
        if isinstance(other, StokesVector):
            return self.apply(other)
        elif isinstance(other, MuellerMatrix):
            return MuellerMatrix(self._mat @ other._mat)
        return NotImplemented

    @property
    def diattenuation(self) -> float:
        r"""Diattenuation $D = \sqrt{m_{01}^2+m_{02}^2+m_{03}^2}/m_{00}$."""
        m = self._mat
        if abs(m[0, 0]) < 1e-30:
            return 0.0
        return float(np.sqrt(m[0, 1]**2 + m[0, 2]**2 + m[0, 3]**2) / m[0, 0])

    @property
    def depolarisation_index(self) -> float:
        r"""Gil-Bernabeu depolarisation index $P_\Delta$."""
        m = self._mat
        tr = float(np.trace(m.T @ m))
        m00_sq = m[0, 0]**2
        if m00_sq < 1e-30:
            return 0.0
        return float(np.sqrt((tr - m00_sq) / (3.0 * m00_sq)))

    # Factory: standard Mueller matrices

    @staticmethod
    def from_jones(J: JonesMatrix) -> "MuellerMatrix":
        """Convert Jones matrix to Mueller matrix via Pauli decomposition."""
        A = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, -1],
            [0, 1, 1, 0],
            [0, 1j, -1j, 0],
        ], dtype=np.complex128)
        A_inv = np.linalg.inv(A)
        JJ = np.kron(J.mat, np.conj(J.mat))
        M = np.real(A @ JJ @ A_inv)
        return MuellerMatrix(M)

    @staticmethod
    def linear_polariser(theta: float = 0.0) -> "MuellerMatrix":
        """Ideal linear polariser at angle θ from horizontal."""
        c2 = math.cos(2.0 * theta)
        s2 = math.sin(2.0 * theta)
        M = 0.5 * np.array([
            [1, c2, s2, 0],
            [c2, c2 * c2, c2 * s2, 0],
            [s2, c2 * s2, s2 * s2, 0],
            [0, 0, 0, 0],
        ])
        return MuellerMatrix(M)

    @staticmethod
    def quarter_wave_plate(theta: float = 0.0) -> "MuellerMatrix":
        return MuellerMatrix.from_jones(JonesMatrix.quarter_wave_plate(theta))

    @staticmethod
    def half_wave_plate(theta: float = 0.0) -> "MuellerMatrix":
        return MuellerMatrix.from_jones(JonesMatrix.half_wave_plate(theta))

    @staticmethod
    def ideal_depolariser(transmittance: float = 1.0) -> "MuellerMatrix":
        """Ideal depolariser: output is unpolarised regardless of input."""
        M = np.zeros((4, 4))
        M[0, 0] = transmittance
        return MuellerMatrix(M)

    @staticmethod
    def rotator(theta: float) -> "MuellerMatrix":
        """Rotation Mueller matrix (coordinate rotation by angle θ)."""
        c2 = math.cos(2.0 * theta)
        s2 = math.sin(2.0 * theta)
        M = np.array([
            [1, 0, 0, 0],
            [0, c2, s2, 0],
            [0, -s2, c2, 0],
            [0, 0, 0, 1],
        ])
        return MuellerMatrix(M)

    @staticmethod
    def identity() -> "MuellerMatrix":
        return MuellerMatrix(np.eye(4))


# ===================================================================
#  Thin-Film Multilayer (Transfer Matrix Method)
# ===================================================================

@dataclass
class ThinFilmLayer:
    """Single layer in a thin-film stack."""
    refractive_index: complex      # n + ik (complex for absorbing)
    thickness: float               # Physical thickness [m]


class ThinFilmStack:
    r"""
    Thin-film interference via the transfer matrix method.

    Interface matrix (Fresnel):
    $$
    t_s = \frac{2n_1\cos\theta_1}{n_1\cos\theta_1 + n_2\cos\theta_2}, \quad
    r_s = \frac{n_1\cos\theta_1 - n_2\cos\theta_2}{n_1\cos\theta_1 + n_2\cos\theta_2}
    $$

    Propagation matrix:
    $$
    P = \begin{pmatrix} e^{i\delta} & 0 \\ 0 & e^{-i\delta} \end{pmatrix}, \quad
    \delta = \frac{2\pi}{\lambda} n d \cos\theta
    $$

    Total system matrix $M = I_{01} P_1 I_{12} P_2 \cdots I_{(N-1)N}$,
    reflectance $R = |M_{10}/M_{00}|^2$, transmittance $T = |1/M_{00}|^2$.
    """

    def __init__(self, layers: Sequence[ThinFilmLayer],
                 n_incident: complex = 1.0,
                 n_substrate: complex = 1.5) -> None:
        self.layers = list(layers)
        self.n_inc = complex(n_incident)
        self.n_sub = complex(n_substrate)

    @staticmethod
    def _snell_angle(n1: complex, theta1: float, n2: complex) -> complex:
        """Snell's law for complex refractive indices."""
        sin_t2 = n1 * np.sin(theta1) / n2
        return np.arcsin(sin_t2)

    @staticmethod
    def _interface_matrix_s(n1: complex, theta1: complex,
                            n2: complex, theta2: complex) -> NDArray:
        """TE (s-polarisation) interface transfer matrix."""
        eta1 = n1 * np.cos(theta1)
        eta2 = n2 * np.cos(theta2)
        r = (eta1 - eta2) / (eta1 + eta2)
        t = 2.0 * eta1 / (eta1 + eta2)
        M = np.array([[1.0, r], [r, 1.0]], dtype=np.complex128) / t
        return M

    @staticmethod
    def _interface_matrix_p(n1: complex, theta1: complex,
                            n2: complex, theta2: complex) -> NDArray:
        """TM (p-polarisation) interface transfer matrix."""
        eta1 = n1 / np.cos(theta1)
        eta2 = n2 / np.cos(theta2)
        r = (eta1 - eta2) / (eta1 + eta2)
        t = 2.0 * eta1 / (eta1 + eta2)
        M = np.array([[1.0, r], [r, 1.0]], dtype=np.complex128) / t
        return M

    @staticmethod
    def _propagation_matrix(n: complex, d: float,
                            theta: complex, wavelength: float) -> NDArray:
        """Phase propagation matrix through layer."""
        delta = 2.0 * np.pi / wavelength * n * d * np.cos(theta)
        return np.array([[np.exp(1j * delta), 0],
                         [0, np.exp(-1j * delta)]], dtype=np.complex128)

    def reflectance_transmittance(
        self,
        wavelength: float,
        theta_inc: float = 0.0,
        polarisation: str = "s",
    ) -> Tuple[float, float]:
        """
        Compute reflectance R and transmittance T for given wavelength and angle.

        Parameters
        ----------
        wavelength : λ [m].
        theta_inc : Incidence angle [rad].
        polarisation : 's' (TE) or 'p' (TM).

        Returns
        -------
        R, T : Power reflectance and transmittance.
        """
        interface_fn = (self._interface_matrix_s if polarisation == "s"
                        else self._interface_matrix_p)

        # Build angle list through layers
        n_list = [self.n_inc] + [L.refractive_index for L in self.layers] + [self.n_sub]
        theta_list = [complex(theta_inc)]
        for i in range(len(n_list) - 1):
            theta_list.append(self._snell_angle(n_list[i], theta_list[i], n_list[i + 1]))

        # System matrix = product of interface and propagation matrices
        M_sys = interface_fn(n_list[0], theta_list[0], n_list[1], theta_list[1])

        for i, layer in enumerate(self.layers):
            P = self._propagation_matrix(
                layer.refractive_index, layer.thickness,
                theta_list[i + 1], wavelength)
            M_sys = M_sys @ P
            M_sys = M_sys @ interface_fn(
                n_list[i + 1], theta_list[i + 1],
                n_list[i + 2], theta_list[i + 2])

        r = M_sys[1, 0] / M_sys[0, 0]
        t = 1.0 / M_sys[0, 0]

        R = float(np.abs(r)**2)

        # Transmittance corrected for refractive index ratio
        n_ratio = np.real(n_list[-1] * np.cos(theta_list[-1])) / np.real(
            n_list[0] * np.cos(theta_list[0]))
        T = float(np.abs(t)**2 * n_ratio)

        return R, T

    def spectrum(
        self,
        wavelengths: NDArray[np.float64],
        theta_inc: float = 0.0,
        polarisation: str = "s",
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute R and T spectra over array of wavelengths."""
        R_arr = np.empty(len(wavelengths))
        T_arr = np.empty(len(wavelengths))
        for i, lam in enumerate(wavelengths):
            R_arr[i], T_arr[i] = self.reflectance_transmittance(
                lam, theta_inc, polarisation)
        return R_arr, T_arr


# ===================================================================
#  Gaussian Beam (ABCD Matrix Method)
# ===================================================================

@dataclass
class GaussianBeam:
    r"""
    Fundamental Gaussian beam TEM₀₀ with ABCD matrix propagation.

    Complex beam parameter:
    $$\frac{1}{q} = \frac{1}{R} - \frac{i\lambda}{\pi w^2}$$

    ABCD propagation:
    $$q_2 = \frac{Aq_1 + B}{Cq_1 + D}$$

    Beam waist: $w_0 = \sqrt{\lambda z_R / \pi}$
    Rayleigh range: $z_R = \pi w_0^2 / \lambda$
    Divergence: $\theta = \lambda / (\pi w_0)$
    """

    wavelength: float       # λ [m]
    waist: float            # w₀ [m]
    waist_position: float = 0.0  # z position of waist [m]

    @property
    def rayleigh_range(self) -> float:
        """Rayleigh range z_R = π w₀² / λ."""
        return np.pi * self.waist**2 / self.wavelength

    @property
    def divergence(self) -> float:
        """Far-field half-angle divergence θ = λ/(π w₀) [rad]."""
        return self.wavelength / (np.pi * self.waist)

    @property
    def confocal_parameter(self) -> float:
        """Confocal parameter b = 2·z_R."""
        return 2.0 * self.rayleigh_range

    def complex_beam_parameter(self, z: float) -> complex:
        """q(z) = (z - z₀) + i·z_R."""
        return complex(z - self.waist_position) + 1j * self.rayleigh_range

    def spot_size(self, z: float) -> float:
        """Beam radius w(z) = w₀ √(1 + ((z-z₀)/z_R)²)."""
        zr = self.rayleigh_range
        dz = z - self.waist_position
        return self.waist * np.sqrt(1.0 + (dz / zr)**2)

    def radius_of_curvature(self, z: float) -> float:
        """Wavefront radius R(z) = (z-z₀)(1 + (z_R/(z-z₀))²)."""
        dz = z - self.waist_position
        if abs(dz) < 1e-30:
            return np.inf
        zr = self.rayleigh_range
        return dz * (1.0 + (zr / dz)**2)

    def gouy_phase(self, z: float) -> float:
        """Gouy phase ζ(z) = arctan((z-z₀)/z_R)."""
        return np.arctan((z - self.waist_position) / self.rayleigh_range)

    def intensity_profile(self, z: float, r: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Intensity profile I(r, z) = I₀ (w₀/w(z))² exp(-2r²/w(z)²).
        """
        w = self.spot_size(z)
        I0 = (self.waist / w)**2
        return I0 * np.exp(-2.0 * r**2 / w**2)

    def propagate_abcd(self, A: float, B: float,
                       C: float, D: float, z: float) -> "GaussianBeam":
        """
        Propagate beam through ABCD optical system.

        Returns new GaussianBeam with updated parameters.
        """
        q_in = self.complex_beam_parameter(z)
        q_out = (A * q_in + B) / (C * q_in + D)

        # Extract w₀' and z₀' from q_out
        inv_q = 1.0 / q_out
        R_out = 1.0 / np.real(inv_q) if abs(np.real(inv_q)) > 1e-30 else np.inf
        w_sq = -self.wavelength / (np.pi * np.imag(inv_q))
        w_out = np.sqrt(abs(w_sq))

        zR_out = np.pi * w_out**2 / self.wavelength
        # New waist position relative to output plane
        if np.isinf(R_out):
            z0_out = 0.0
        else:
            z0_out = -R_out / (1.0 + (zR_out / R_out)**2) if abs(R_out) > 1e-30 else 0.0

        return GaussianBeam(
            wavelength=self.wavelength,
            waist=w_out,
            waist_position=z0_out,
        )

    # Standard ABCD elements
    @staticmethod
    def abcd_free_space(d: float) -> Tuple[float, float, float, float]:
        """Free-space propagation of distance d."""
        return 1.0, d, 0.0, 1.0

    @staticmethod
    def abcd_thin_lens(f: float) -> Tuple[float, float, float, float]:
        """Thin lens of focal length f."""
        return 1.0, 0.0, -1.0 / f, 1.0

    @staticmethod
    def abcd_curved_mirror(R: float) -> Tuple[float, float, float, float]:
        """Concave mirror of radius R (R > 0 concave)."""
        return 1.0, 0.0, -2.0 / R, 1.0

    @staticmethod
    def abcd_flat_interface(n1: float, n2: float) -> Tuple[float, float, float, float]:
        """Refraction at flat interface from n1 to n2."""
        return 1.0, 0.0, 0.0, n1 / n2

    @staticmethod
    def abcd_curved_interface(n1: float, n2: float,
                              R: float) -> Tuple[float, float, float, float]:
        """Refraction at curved interface (radius R) from n1 to n2."""
        return 1.0, 0.0, (n1 - n2) / (n2 * R), n1 / n2
