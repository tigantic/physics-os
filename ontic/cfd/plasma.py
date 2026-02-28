"""
Plasma and Ionization Model for Hypersonic Reentry
===================================================

Phase 22: Plasma/Ionization modeling for communications blackout prediction.

This module implements the physics of plasma sheath formation during hypersonic
reentry, enabling prediction of RF signal attenuation and communications blackout.

Key Physics:
- Saha ionization equation for thermal equilibrium plasma
- Plasma frequency calculation for RF propagation cutoff
- Electron density field computation from CFD solutions
- Attenuation modeling for signal loss estimation

References:
    - Rybkin & Shutov, "Plasma Chemistry and Catalysis" (2019)
    - Hartunian et al., "Reentry Plasma Physics" (1962)
    - NASA RP-1232, "Radiation from Hypersonic Entry Plasmas" (1990)

Constitution Compliance: Article II.1, Article V
"""

import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

# =============================================================================
# Physical Constants
# =============================================================================

# Fundamental constants (SI units)
K_BOLTZMANN = 1.380649e-23  # J/K
ELECTRON_MASS = 9.10938e-31  # kg
ELECTRON_CHARGE = 1.60218e-19  # C
PLANCK = 6.62607e-34  # J·s
EPSILON_0 = 8.854188e-12  # F/m
C_LIGHT = 2.99792e8  # m/s
AVOGADRO = 6.02214e23  # mol^-1

# Ionization energies in eV (first ionization)
IONIZATION_ENERGY = {
    "N": 14.534,  # Nitrogen
    "O": 13.618,  # Oxygen
    "N2": 15.581,  # Molecular nitrogen
    "O2": 12.070,  # Molecular oxygen
    "NO": 9.264,  # Nitric oxide (lowest, dominant at moderate T)
    "Ar": 15.760,  # Argon
    "H": 13.598,  # Hydrogen
    "He": 24.587,  # Helium
}

# Partition function degeneracy (g_0 for neutral, g_1 for ion)
PARTITION_DEGENERACY = {
    "N": (4, 9),
    "O": (9, 4),
    "N2": (1, 2),
    "O2": (3, 2),
    "NO": (2, 1),
    "Ar": (1, 6),
    "H": (2, 1),
    "He": (1, 2),
}

# Molecular mass in kg
MOLECULAR_MASS = {
    "N": 14.007e-3 / AVOGADRO,
    "O": 15.999e-3 / AVOGADRO,
    "N2": 28.014e-3 / AVOGADRO,
    "O2": 31.998e-3 / AVOGADRO,
    "NO": 30.006e-3 / AVOGADRO,
    "Ar": 39.948e-3 / AVOGADRO,
    "H": 1.008e-3 / AVOGADRO,
    "He": 4.003e-3 / AVOGADRO,
}


class Species(Enum):
    """Gas species for plasma calculations."""

    N = "N"
    O = "O"
    N2 = "N2"
    O2 = "O2"
    NO = "NO"
    Ar = "Ar"
    H = "H"
    He = "He"


@dataclass
class PlasmaState:
    """
    State of the plasma at a point or field.

    Attributes:
        electron_density: Electron number density (m^-3)
        ion_density: Ion number density by species (m^-3)
        temperature: Temperature (K)
        pressure: Pressure (Pa)
        plasma_frequency: Plasma frequency (rad/s)
        debye_length: Debye shielding length (m)
    """

    electron_density: Tensor
    ion_density: dict[str, Tensor]
    temperature: Tensor
    pressure: Tensor
    plasma_frequency: Tensor
    debye_length: Tensor


@dataclass
class PlasmaSheath:
    """
    Container for plasma sheath properties around a vehicle.

    Attributes:
        thickness: Sheath thickness field (m)
        electron_density: Electron density field (m^-3)
        plasma_frequency: Plasma frequency field (rad/s)
        attenuation: Signal attenuation field (dB)
        blackout_mask: Boolean mask where signal is lost
    """

    thickness: Tensor
    electron_density: Tensor
    plasma_frequency: Tensor
    attenuation: Tensor
    blackout_mask: Tensor

    @property
    def max_electron_density(self) -> float:
        """Maximum electron density in the sheath."""
        return self.electron_density.max().item()

    @property
    def critical_frequency(self) -> float:
        """Critical frequency below which signals are blocked (Hz)."""
        return self.plasma_frequency.max().item() / (2 * math.pi)


# =============================================================================
# Saha Ionization Equation
# =============================================================================


def saha_ionization(T: Tensor, p: Tensor, species: str = "N") -> Tensor:
    """
    Compute electron density using the Saha ionization equation.

    The Saha equation describes thermal equilibrium ionization:

        n_e * n_i / n_0 = (2/Λ³) * (g_1/g_0) * exp(-E_ion / kT)

    where Λ is the thermal de Broglie wavelength.

    For a single species with charge neutrality (n_e = n_i):
        n_e² / n_0 = K_saha(T)
        n_e = sqrt(n_0 * K_saha)

    Args:
        T: Temperature (K), can be scalar or field
        p: Pressure (Pa), can be scalar or field
        species: Species name ('N', 'O', 'NO', etc.)

    Returns:
        Electron density (m^-3)

    References:
        Saha, M.N. (1920). "Ionization in the solar chromosphere"
    """
    # Get species properties
    E_ion_eV = IONIZATION_ENERGY.get(species, 14.0)
    E_ion_J = E_ion_eV * ELECTRON_CHARGE  # Convert to Joules

    g_0, g_1 = PARTITION_DEGENERACY.get(species, (1, 1))
    m_species = MOLECULAR_MASS.get(species, 28e-3 / AVOGADRO)

    # Ensure tensors
    T = torch.as_tensor(T, dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64)

    # Clamp temperature to avoid numerical issues
    T = torch.clamp(T, min=300.0)

    # Thermal de Broglie wavelength for electrons
    # Λ = h / sqrt(2πm_e kT)
    Lambda_e = PLANCK / torch.sqrt(2 * math.pi * ELECTRON_MASS * K_BOLTZMANN * T)

    # Saha constant (partition function ratio)
    # K_saha = (2/Λ³) * (g_1/g_0) * exp(-E_ion/kT)
    K_saha = (2 / Lambda_e**3) * (g_1 / g_0) * torch.exp(-E_ion_J / (K_BOLTZMANN * T))

    # Total number density from ideal gas law: n = p / (kT)
    n_total = p / (K_BOLTZMANN * T)

    # Ionization fraction from quadratic: n_e² + K_saha*n_e - K_saha*n_total = 0
    # For weak ionization (n_e << n_total): n_e ≈ sqrt(n_total * K_saha)
    # For strong ionization: solve quadratic

    # Discriminant for quadratic
    discriminant = K_saha**2 + 4 * K_saha * n_total

    # Electron density (positive root of quadratic)
    n_e = (-K_saha + torch.sqrt(discriminant)) / 2

    # Ensure non-negative
    n_e = torch.clamp(n_e, min=0.0)

    return n_e


def saha_multi_species(
    T: Tensor, p: Tensor, mass_fractions: dict[str, float]
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Compute electron density for a multi-species gas mixture.

    Uses Saha equation for each species and sums contributions,
    accounting for the fact that NO ionizes at lower temperatures.

    Args:
        T: Temperature (K)
        p: Pressure (Pa)
        mass_fractions: Mass fraction of each species

    Returns:
        (total_n_e, species_n_e): Total and per-species electron densities
    """
    T = torch.as_tensor(T, dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64)

    total_n_e = torch.zeros_like(T)
    species_n_e = {}

    # Normalize mass fractions
    total_mass = sum(mass_fractions.values())

    for species, Y in mass_fractions.items():
        if species not in IONIZATION_ENERGY:
            continue

        # Partial pressure
        p_species = p * (Y / total_mass)

        # Electron density from this species
        n_e_species = saha_ionization(T, p_species, species)

        species_n_e[species] = n_e_species
        total_n_e = total_n_e + n_e_species

    return total_n_e, species_n_e


# =============================================================================
# Plasma Frequency and RF Propagation
# =============================================================================


def plasma_frequency(n_e: Tensor) -> Tensor:
    """
    Compute the plasma frequency from electron density.

    The plasma frequency is the natural oscillation frequency of electrons:

        ω_pe = sqrt(n_e * e² / (ε_0 * m_e))

    RF signals with frequency ω < ω_pe cannot propagate through the plasma.

    Args:
        n_e: Electron density (m^-3)

    Returns:
        Plasma frequency (rad/s)
    """
    n_e = torch.as_tensor(n_e, dtype=torch.float64)
    n_e = torch.clamp(n_e, min=0.0)

    omega_pe = torch.sqrt(n_e * ELECTRON_CHARGE**2 / (EPSILON_0 * ELECTRON_MASS))

    return omega_pe


def critical_frequency(n_e: Tensor) -> Tensor:
    """
    Compute the critical frequency (Hz) for RF propagation.

    Signals below this frequency cannot penetrate the plasma.

    Args:
        n_e: Electron density (m^-3)

    Returns:
        Critical frequency (Hz)
    """
    omega_pe = plasma_frequency(n_e)
    f_c = omega_pe / (2 * math.pi)
    return f_c


def rf_attenuation(
    omega_signal: Tensor,
    omega_pe: Tensor,
    nu_collision: Tensor | None = None,
    path_length: float = 0.1,
) -> Tensor:
    """
    Compute RF signal attenuation through a plasma layer.

    Uses the Appleton-Hartree equation for a collisional plasma:

        n² = 1 - (ω_pe²/ω²) / (1 - j*ν/ω)

    For ω > ω_pe (propagating): attenuation from collisions
    For ω < ω_pe (evanescent): exponential decay

    Args:
        omega_signal: Signal angular frequency (rad/s)
        omega_pe: Plasma frequency (rad/s)
        nu_collision: Collision frequency (Hz), estimated if None
        path_length: Path length through plasma (m)

    Returns:
        Attenuation (dB)
    """
    omega_signal = torch.as_tensor(omega_signal, dtype=torch.float64)
    omega_pe = torch.as_tensor(omega_pe, dtype=torch.float64)

    # Estimate collision frequency if not provided
    # ν ≈ 10^11 * n_e^(1/2) for typical reentry conditions
    if nu_collision is None:
        n_e = omega_pe**2 * EPSILON_0 * ELECTRON_MASS / ELECTRON_CHARGE**2
        nu_collision = 1e4 * torch.sqrt(n_e + 1e10)  # Empirical estimate

    # Ratio X = (ω_pe/ω)²
    X = (omega_pe / (omega_signal + 1e-10)) ** 2

    # Collision term Z = ν/ω
    Z = nu_collision / (omega_signal + 1e-10)

    # Complex refractive index squared
    # n² = 1 - X / (1 - jZ) for no magnetic field
    denom = 1 + Z**2
    n_sq_real = 1 - X / denom
    n_sq_imag = -X * Z / denom

    # Attenuation coefficient α = (ω/c) * Im(n)
    # For n² = n_r² - n_i² + 2j*n_r*n_i
    # When n_sq_real < 0 (cutoff), signal is evanescent

    # Compute |n| and phase
    n_sq_mag = torch.sqrt(n_sq_real**2 + n_sq_imag**2)

    # Real part of n
    n_real = torch.sqrt((n_sq_mag + n_sq_real) / 2)
    # Imaginary part of n
    n_imag = torch.sqrt((n_sq_mag - n_sq_real) / 2)

    # Handle evanescent case (X > 1)
    evanescent = X > 1
    n_imag = torch.where(evanescent, torch.sqrt(X - 1), n_imag)

    # Attenuation coefficient (Np/m)
    alpha = omega_signal * n_imag / C_LIGHT

    # Total attenuation (dB)
    attenuation_nepers = alpha * path_length
    attenuation_dB = 8.686 * attenuation_nepers  # 1 Np = 8.686 dB

    return attenuation_dB


def is_blackout(
    omega_signal: Tensor, omega_pe: Tensor, threshold_dB: float = 30.0
) -> Tensor:
    """
    Determine if communication blackout occurs.

    Blackout occurs when:
    1. Signal frequency is below plasma frequency (ω < ω_pe)
    2. Attenuation exceeds threshold even above cutoff

    Args:
        omega_signal: Signal angular frequency (rad/s)
        omega_pe: Plasma frequency (rad/s)
        threshold_dB: Attenuation threshold for blackout

    Returns:
        Boolean tensor indicating blackout condition
    """
    omega_signal = torch.as_tensor(omega_signal, dtype=torch.float64)
    omega_pe = torch.as_tensor(omega_pe, dtype=torch.float64)

    # Primary cutoff condition
    cutoff = omega_signal < omega_pe

    # Attenuation-based condition
    atten = rf_attenuation(omega_signal, omega_pe)
    high_attenuation = atten > threshold_dB

    return cutoff | high_attenuation


# =============================================================================
# Electron Density Field from CFD
# =============================================================================


def electron_density_field(
    temperature_field: Tensor,
    pressure_field: Tensor,
    mass_fractions: dict[str, float] | None = None,
) -> Tensor:
    """
    Compute electron density field from CFD temperature and pressure.

    This function maps CFD solution fields to electron density for
    plasma sheath characterization.

    Args:
        temperature_field: Temperature field (K)
        pressure_field: Pressure field (Pa)
        mass_fractions: Species mass fractions, defaults to air

    Returns:
        Electron density field (m^-3)
    """
    # Default to air composition at high temperature
    # (equilibrium dissociation products)
    if mass_fractions is None:
        mass_fractions = {
            "N": 0.30,  # Atomic nitrogen
            "O": 0.25,  # Atomic oxygen
            "N2": 0.20,  # Molecular nitrogen
            "O2": 0.05,  # Molecular oxygen
            "NO": 0.20,  # Nitric oxide (important for ionization)
        }

    n_e, _ = saha_multi_species(temperature_field, pressure_field, mass_fractions)

    return n_e


def compute_plasma_sheath(
    temperature_field: Tensor,
    pressure_field: Tensor,
    density_field: Tensor,
    signal_frequency: float = 2.4e9,  # 2.4 GHz (typical comm frequency)
    mass_fractions: dict[str, float] | None = None,
) -> PlasmaSheath:
    """
    Compute complete plasma sheath properties from CFD solution.

    Args:
        temperature_field: Temperature (K)
        pressure_field: Pressure (Pa)
        density_field: Density (kg/m³)
        signal_frequency: Communication signal frequency (Hz)
        mass_fractions: Species mass fractions

    Returns:
        PlasmaSheath with all relevant properties
    """
    # Electron density
    n_e = electron_density_field(temperature_field, pressure_field, mass_fractions)

    # Plasma frequency
    omega_pe = plasma_frequency(n_e)

    # Signal angular frequency
    omega_signal = 2 * math.pi * signal_frequency

    # Attenuation
    atten = rf_attenuation(torch.tensor(omega_signal), omega_pe)

    # Blackout mask
    blackout = is_blackout(torch.tensor(omega_signal), omega_pe)

    # Estimate sheath thickness from density gradient
    # (simplified: use characteristic length scale)
    rho_min = density_field.min()
    rho_max = density_field.max()
    thickness = torch.ones_like(temperature_field) * 0.01  # 1 cm estimate

    # Debye length for reference
    # λ_D = sqrt(ε_0 * k_B * T / (n_e * e²))
    T_clamped = torch.clamp(temperature_field, min=1000.0)
    n_e_clamped = torch.clamp(n_e, min=1e10)
    debye = torch.sqrt(
        EPSILON_0 * K_BOLTZMANN * T_clamped / (n_e_clamped * ELECTRON_CHARGE**2)
    )

    return PlasmaSheath(
        thickness=thickness,
        electron_density=n_e,
        plasma_frequency=omega_pe,
        attenuation=atten,
        blackout_mask=blackout,
    )


# =============================================================================
# Plasma State Computation
# =============================================================================


def compute_plasma_state(
    T: Tensor, p: Tensor, mass_fractions: dict[str, float] | None = None
) -> PlasmaState:
    """
    Compute complete plasma state from thermodynamic conditions.

    Args:
        T: Temperature (K)
        p: Pressure (Pa)
        mass_fractions: Species mass fractions

    Returns:
        PlasmaState with electron/ion densities and derived quantities
    """
    if mass_fractions is None:
        mass_fractions = {"NO": 0.5, "N": 0.25, "O": 0.25}

    # Multi-species ionization
    n_e, ion_densities = saha_multi_species(T, p, mass_fractions)

    # Plasma frequency
    omega_pe = plasma_frequency(n_e)

    # Debye length
    T_clamped = torch.clamp(T, min=1000.0)
    n_e_clamped = torch.clamp(n_e, min=1e10)
    debye = torch.sqrt(
        EPSILON_0 * K_BOLTZMANN * T_clamped / (n_e_clamped * ELECTRON_CHARGE**2)
    )

    return PlasmaState(
        electron_density=n_e,
        ion_density=ion_densities,
        temperature=T,
        pressure=p,
        plasma_frequency=omega_pe,
        debye_length=debye,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def frequency_to_angular(f: float) -> float:
    """Convert frequency (Hz) to angular frequency (rad/s)."""
    return 2 * math.pi * f


def angular_to_frequency(omega: float) -> float:
    """Convert angular frequency (rad/s) to frequency (Hz)."""
    return omega / (2 * math.pi)


def electron_density_to_critical_freq(n_e: float) -> float:
    """
    Convert electron density to critical frequency.

    f_c ≈ 9 * sqrt(n_e) Hz for n_e in m^-3

    Args:
        n_e: Electron density (m^-3)

    Returns:
        Critical frequency (Hz)
    """
    return 9.0 * math.sqrt(n_e)


def critical_freq_to_electron_density(f_c: float) -> float:
    """
    Convert critical frequency to required electron density.

    Args:
        f_c: Critical frequency (Hz)

    Returns:
        Electron density (m^-3)
    """
    return (f_c / 9.0) ** 2


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Constants
    "K_BOLTZMANN",
    "ELECTRON_MASS",
    "ELECTRON_CHARGE",
    "IONIZATION_ENERGY",
    "MOLECULAR_MASS",
    # Types
    "Species",
    "PlasmaState",
    "PlasmaSheath",
    # Core functions
    "saha_ionization",
    "saha_multi_species",
    "plasma_frequency",
    "critical_frequency",
    "rf_attenuation",
    "is_blackout",
    # Field computations
    "electron_density_field",
    "compute_plasma_sheath",
    "compute_plasma_state",
    # Utilities
    "frequency_to_angular",
    "angular_to_frequency",
    "electron_density_to_critical_freq",
    "critical_freq_to_electron_density",
]
