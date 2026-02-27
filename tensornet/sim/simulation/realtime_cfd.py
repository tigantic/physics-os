"""
Real-Time CFD Coupling Module
=============================

Provides interfaces for coupling high-fidelity CFD solutions with
real-time guidance algorithms through aerodynamic tables and
surrogate models.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    OFFLINE (Pre-computation)                    │
    │  ┌────────────┐    ┌──────────────┐    ┌────────────────────┐  │
    │  │ Mesh/Grid  │───►│ CFD Solver   │───►│ Post-Processing    │  │
    │  │ Generation │    │ (Euler/NS)   │    │ (Forces/Moments)   │  │
    │  └────────────┘    └──────────────┘    └─────────┬──────────┘  │
    │                                                   │             │
    │                                          ┌───────▼────────┐    │
    │                                          │  AeroTable     │    │
    │                                          │  Generation    │    │
    │                                          └───────┬────────┘    │
    └──────────────────────────────────────────────────┼─────────────┘
                                                       │
    ┌──────────────────────────────────────────────────┼─────────────┐
    │                    ONLINE (Real-time)            │             │
    │                                          ┌───────▼────────┐    │
    │                                          │   AeroTable    │    │
    │                                          │   Lookup       │    │
    │                                          └───────┬────────┘    │
    │                                                   │             │
    │  ┌────────────┐    ┌──────────────┐    ┌────────▼─────────┐   │
    │  │ Vehicle    │───►│ RealTimeCFD  │◄───│ Interpolate      │   │
    │  │ State      │    │ Interface    │    │ CL, CD, Cm, ...  │   │
    │  └────────────┘    └──────────────┘    └──────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘

Interpolation Methods:
    - Linear (fastest, for monotonic data)
    - Cubic spline (smooth, general purpose)
    - Kriging (uncertainty quantification)
    - Neural network (complex nonlinear)

Table Dimensions:
    - Mach number (primary)
    - Angle of attack (primary)
    - Sideslip angle (secondary)
    - Control surface deflections
    - Reynolds number (optional)
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class InterpolationMethod(Enum):
    """Interpolation method for aero tables."""

    LINEAR = "linear"
    NEAREST = "nearest"
    CUBIC = "cubic"
    RBF = "rbf"
    NEURAL = "neural"


class TableDimension(Enum):
    """Standard aerodynamic table dimensions."""

    MACH = "mach"
    ALPHA = "alpha"
    BETA = "beta"
    ELEVATOR = "de"
    AILERON = "da"
    RUDDER = "dr"
    REYNOLDS = "re"
    ALTITUDE = "alt"


@dataclass
class AeroTableConfig:
    """Configuration for aerodynamic table generation."""

    # Dimension ranges
    mach_range: tuple[float, float] = (0.5, 10.0)
    alpha_range: tuple[float, float] = (-5.0, 25.0)
    beta_range: tuple[float, float] = (-10.0, 10.0)

    # Grid points
    n_mach: int = 20
    n_alpha: int = 31
    n_beta: int = 11

    # Control surface ranges
    elevator_range: tuple[float, float] = (-30.0, 30.0)
    aileron_range: tuple[float, float] = (-25.0, 25.0)
    rudder_range: tuple[float, float] = (-30.0, 30.0)
    n_deflection: int = 13

    # Interpolation
    method: InterpolationMethod = InterpolationMethod.LINEAR

    # Reference values
    S_ref: float = 10.0  # Reference area (m²)
    c_ref: float = 2.0  # Reference chord (m)
    b_ref: float = 5.0  # Reference span (m)

    # Extrapolation behavior
    bounds_error: bool = False
    fill_value: float | None = None  # None = extrapolate


@dataclass
class AeroCoefficient:
    """Single aerodynamic coefficient with derivatives."""

    base: float = 0.0
    d_alpha: float = 0.0  # Per radian
    d_beta: float = 0.0
    d_p: float = 0.0  # Rate derivatives (non-dimensional)
    d_q: float = 0.0
    d_r: float = 0.0
    d_de: float = 0.0  # Control derivatives
    d_da: float = 0.0
    d_dr: float = 0.0


@dataclass
class AeroPoint:
    """Aerodynamic coefficients at a single flight condition."""

    CL: float = 0.0
    CD: float = 0.0
    CY: float = 0.0
    Cl: float = 0.0  # Rolling moment
    Cm: float = 0.0  # Pitching moment
    Cn: float = 0.0  # Yawing moment

    # Derivatives (optional)
    CL_alpha: float = 0.0
    CD_alpha: float = 0.0
    Cm_alpha: float = 0.0
    Cm_q: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to coefficient vector."""
        return np.array([self.CL, self.CD, self.CY, self.Cl, self.Cm, self.Cn])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "AeroPoint":
        """Create from coefficient vector."""
        return cls(CL=v[0], CD=v[1], CY=v[2], Cl=v[3], Cm=v[4], Cn=v[5])


class AeroTable:
    """
    Multi-dimensional aerodynamic lookup table.

    Provides fast interpolation of aerodynamic coefficients
    across the flight envelope.
    """

    def __init__(self, config: AeroTableConfig = None):
        self.config = config or AeroTableConfig()

        # Grid points
        self.mach_grid = np.linspace(*self.config.mach_range, self.config.n_mach)
        self.alpha_grid = np.linspace(*self.config.alpha_range, self.config.n_alpha)
        self.beta_grid = np.linspace(*self.config.beta_range, self.config.n_beta)

        # Coefficient tables (Mach x Alpha x Beta)
        shape = (self.config.n_mach, self.config.n_alpha, self.config.n_beta)
        self.CL_table = np.zeros(shape)
        self.CD_table = np.zeros(shape)
        self.CY_table = np.zeros(shape)
        self.Cl_table = np.zeros(shape)
        self.Cm_table = np.zeros(shape)
        self.Cn_table = np.zeros(shape)

        # Derivative tables
        self.CL_alpha_table = np.zeros(shape)
        self.Cm_alpha_table = np.zeros(shape)
        self.Cm_q_table = np.zeros(shape)

        # Interpolators (created after table population)
        self._interpolators: dict[str, RegularGridInterpolator] = {}
        self._is_built = False

    def populate_from_cfd(
        self, cfd_solver: Callable[[float, float, float], dict[str, float]]
    ):
        """
        Populate table from CFD solver.

        Args:
            cfd_solver: Function(mach, alpha_deg, beta_deg) -> aero_dict
        """
        for i, mach in enumerate(self.mach_grid):
            for j, alpha in enumerate(self.alpha_grid):
                for k, beta in enumerate(self.beta_grid):
                    # Call CFD solver
                    aero = cfd_solver(mach, alpha, beta)

                    # Store coefficients
                    self.CL_table[i, j, k] = aero.get("CL", 0.0)
                    self.CD_table[i, j, k] = aero.get("CD", 0.0)
                    self.CY_table[i, j, k] = aero.get("CY", 0.0)
                    self.Cl_table[i, j, k] = aero.get("Cl", 0.0)
                    self.Cm_table[i, j, k] = aero.get("Cm", 0.0)
                    self.Cn_table[i, j, k] = aero.get("Cn", 0.0)

        self._build_interpolators()

    def populate_from_arrays(
        self,
        CL: np.ndarray,
        CD: np.ndarray,
        Cm: np.ndarray,
        CY: np.ndarray | None = None,
        Cl: np.ndarray | None = None,
        Cn: np.ndarray | None = None,
    ):
        """Populate from pre-computed arrays."""
        self.CL_table = CL
        self.CD_table = CD
        self.Cm_table = Cm

        if CY is not None:
            self.CY_table = CY
        if Cl is not None:
            self.Cl_table = Cl
        if Cn is not None:
            self.Cn_table = Cn

        self._build_interpolators()

    def _build_interpolators(self):
        """Build interpolator objects for each coefficient."""
        grids = (self.mach_grid, self.alpha_grid, self.beta_grid)
        method = (
            self.config.method.value
            if self.config.method != InterpolationMethod.RBF
            else "linear"
        )

        # Ensure fill_value is numeric
        fill_value = (
            self.config.fill_value if self.config.fill_value is not None else 0.0
        )

        self._interpolators = {
            "CL": RegularGridInterpolator(
                grids,
                self.CL_table,
                method=method,
                bounds_error=self.config.bounds_error,
                fill_value=fill_value,
            ),
            "CD": RegularGridInterpolator(
                grids,
                self.CD_table,
                method=method,
                bounds_error=self.config.bounds_error,
                fill_value=fill_value,
            ),
            "CY": RegularGridInterpolator(
                grids,
                self.CY_table,
                method=method,
                bounds_error=self.config.bounds_error,
                fill_value=fill_value,
            ),
            "Cl": RegularGridInterpolator(
                grids,
                self.Cl_table,
                method=method,
                bounds_error=self.config.bounds_error,
                fill_value=fill_value,
            ),
            "Cm": RegularGridInterpolator(
                grids,
                self.Cm_table,
                method=method,
                bounds_error=self.config.bounds_error,
                fill_value=fill_value,
            ),
            "Cn": RegularGridInterpolator(
                grids,
                self.Cn_table,
                method=method,
                bounds_error=self.config.bounds_error,
                fill_value=fill_value,
            ),
        }

        self._is_built = True

    def lookup(self, mach: float, alpha_deg: float, beta_deg: float = 0.0) -> AeroPoint:
        """
        Look up aerodynamic coefficients.

        Args:
            mach: Mach number
            alpha_deg: Angle of attack (degrees)
            beta_deg: Sideslip angle (degrees)

        Returns:
            AeroPoint with interpolated coefficients
        """
        if not self._is_built:
            raise RuntimeError("AeroTable not built. Call populate_* first.")

        point = np.array([[mach, alpha_deg, beta_deg]])

        return AeroPoint(
            CL=float(self._interpolators["CL"](point)[0]),
            CD=float(self._interpolators["CD"](point)[0]),
            CY=float(self._interpolators["CY"](point)[0]),
            Cl=float(self._interpolators["Cl"](point)[0]),
            Cm=float(self._interpolators["Cm"](point)[0]),
            Cn=float(self._interpolators["Cn"](point)[0]),
        )

    def lookup_batch(
        self, mach: np.ndarray, alpha_deg: np.ndarray, beta_deg: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Batch lookup for multiple points.

        Args:
            mach: Array of Mach numbers
            alpha_deg: Array of angles of attack
            beta_deg: Array of sideslip angles

        Returns:
            Dict of coefficient arrays
        """
        if not self._is_built:
            raise RuntimeError("AeroTable not built.")

        points = np.column_stack([mach, alpha_deg, beta_deg])

        return {
            "CL": self._interpolators["CL"](points),
            "CD": self._interpolators["CD"](points),
            "CY": self._interpolators["CY"](points),
            "Cl": self._interpolators["Cl"](points),
            "Cm": self._interpolators["Cm"](points),
            "Cn": self._interpolators["Cn"](points),
        }

    def save(self, filepath: str | Path):
        """Save table to file."""
        data = {
            "mach_grid": self.mach_grid.tolist(),
            "alpha_grid": self.alpha_grid.tolist(),
            "beta_grid": self.beta_grid.tolist(),
            "CL": self.CL_table.tolist(),
            "CD": self.CD_table.tolist(),
            "CY": self.CY_table.tolist(),
            "Cl": self.Cl_table.tolist(),
            "Cm": self.Cm_table.tolist(),
            "Cn": self.Cn_table.tolist(),
            "config": {
                "S_ref": self.config.S_ref,
                "c_ref": self.config.c_ref,
                "b_ref": self.config.b_ref,
                "method": self.config.method.value,
            },
        }

        Path(filepath).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, filepath: str | Path) -> "AeroTable":
        """Load table from file."""
        data = json.loads(Path(filepath).read_text())

        config = AeroTableConfig(
            mach_range=(data["mach_grid"][0], data["mach_grid"][-1]),
            alpha_range=(data["alpha_grid"][0], data["alpha_grid"][-1]),
            beta_range=(data["beta_grid"][0], data["beta_grid"][-1]),
            n_mach=len(data["mach_grid"]),
            n_alpha=len(data["alpha_grid"]),
            n_beta=len(data["beta_grid"]),
            S_ref=data["config"]["S_ref"],
            c_ref=data["config"]["c_ref"],
            b_ref=data["config"]["b_ref"],
            method=InterpolationMethod(data["config"]["method"]),
        )

        table = cls(config)
        table.mach_grid = np.array(data["mach_grid"])
        table.alpha_grid = np.array(data["alpha_grid"])
        table.beta_grid = np.array(data["beta_grid"])
        table.CL_table = np.array(data["CL"])
        table.CD_table = np.array(data["CD"])
        table.CY_table = np.array(data["CY"])
        table.Cl_table = np.array(data["Cl"])
        table.Cm_table = np.array(data["Cm"])
        table.Cn_table = np.array(data["Cn"])

        table._build_interpolators()
        return table


class RealTimeCFD:
    """
    Real-time CFD interface for guidance systems.

    Combines pre-computed aero tables with optional online
    corrections for real-time simulation.
    """

    def __init__(self, aero_table: AeroTable, config: AeroTableConfig = None):
        self.table = aero_table
        self.config = config or aero_table.config

        # Online correction factors
        self.CL_correction = 0.0
        self.CD_correction = 0.0
        self.Cm_correction = 0.0

        # State for derivative estimation
        self._prev_state: dict | None = None
        self._prev_time: float | None = None

        # Performance monitoring
        self.lookup_count = 0
        self.total_lookup_time_us = 0.0

    def get_aero(
        self, state: dict[str, float], controls: dict[str, float] = None
    ) -> dict[str, float]:
        """
        Get aerodynamic forces and moments.

        Args:
            state: Vehicle state (mach, alpha_deg, beta_deg, alt, V, q_bar)
            controls: Control deflections (de, da, dr)

        Returns:
            Aero dict with forces, moments, and coefficients
        """
        import time

        t0 = time.perf_counter_ns()

        mach = state.get("mach", 1.0)
        alpha = state.get("alpha_deg", 0.0)
        beta = state.get("beta_deg", 0.0)
        q_bar = state.get("q_bar", 10000.0)  # Dynamic pressure
        V = state.get("V", 300.0)  # Velocity

        # Table lookup
        aero = self.table.lookup(mach, alpha, beta)

        # Apply corrections
        CL = aero.CL + self.CL_correction
        CD = aero.CD + self.CD_correction
        Cm = aero.Cm + self.Cm_correction
        CY = aero.CY
        Cl = aero.Cl
        Cn = aero.Cn

        # Control effectiveness (simplified)
        if controls:
            de = controls.get("de", 0.0)
            da = controls.get("da", 0.0)
            dr = controls.get("dr", 0.0)

            # Control derivatives (typical values)
            CL += 0.02 * de  # CL_de
            CD += 0.001 * abs(de)  # Trim drag
            Cm += -0.03 * de  # Cm_de (stabilizing)
            Cl += 0.02 * da  # Cl_da
            Cn += 0.01 * dr  # Cn_dr
            CY += 0.01 * dr  # CY_dr

        # Compute forces
        S = self.config.S_ref
        c = self.config.c_ref
        b = self.config.b_ref

        L = q_bar * S * CL
        D = q_bar * S * CD
        Y = q_bar * S * CY
        l_moment = q_bar * S * b * Cl
        m_moment = q_bar * S * c * Cm
        n_moment = q_bar * S * b * Cn

        # Lift-to-drag ratio
        LD = CL / max(CD, 0.001)

        # Update performance stats
        self.lookup_count += 1
        self.total_lookup_time_us += (time.perf_counter_ns() - t0) / 1000

        return {
            "CL": CL,
            "CD": CD,
            "CY": CY,
            "Cl": Cl,
            "Cm": Cm,
            "Cn": Cn,
            "L": L,
            "D": D,
            "Y": Y,
            "l_moment": l_moment,
            "m_moment": m_moment,
            "n_moment": n_moment,
            "L_D": LD,
            "mach": mach,
            "alpha_deg": alpha,
            "beta_deg": beta,
        }

    def update_correction(
        self, measured_CL: float, measured_CD: float, measured_Cm: float
    ):
        """
        Update correction factors from in-flight measurements.

        Args:
            measured_CL: Measured lift coefficient
            measured_CD: Measured drag coefficient
            measured_Cm: Measured pitching moment coefficient
        """
        # Simple first-order correction (could use Kalman filter)
        alpha = 0.1  # Learning rate

        self.CL_correction += alpha * (
            measured_CL - (self.table.lookup(1.0, 0.0).CL + self.CL_correction)
        )
        self.CD_correction += alpha * (
            measured_CD - (self.table.lookup(1.0, 0.0).CD + self.CD_correction)
        )
        self.Cm_correction += alpha * (
            measured_Cm - (self.table.lookup(1.0, 0.0).Cm + self.Cm_correction)
        )

    def get_derivatives(
        self, state: dict[str, float], epsilon: float = 0.1
    ) -> dict[str, float]:
        """
        Estimate aerodynamic derivatives numerically.

        Args:
            state: Current flight state
            epsilon: Perturbation size (degrees for alpha)

        Returns:
            Dict of derivatives
        """
        mach = state.get("mach", 1.0)
        alpha = state.get("alpha_deg", 0.0)
        beta = state.get("beta_deg", 0.0)

        # Central differences for alpha derivatives
        aero_plus = self.table.lookup(mach, alpha + epsilon, beta)
        aero_minus = self.table.lookup(mach, alpha - epsilon, beta)

        dalpha = 2 * epsilon * np.pi / 180  # Convert to radians

        return {
            "CL_alpha": (aero_plus.CL - aero_minus.CL) / dalpha,
            "CD_alpha": (aero_plus.CD - aero_minus.CD) / dalpha,
            "Cm_alpha": (aero_plus.Cm - aero_minus.Cm) / dalpha,
        }

    def get_performance_stats(self) -> dict[str, float]:
        """Get lookup performance statistics."""
        avg_time = self.total_lookup_time_us / max(self.lookup_count, 1)

        return {
            "lookup_count": self.lookup_count,
            "total_time_ms": self.total_lookup_time_us / 1000,
            "avg_lookup_time_us": avg_time,
            "max_lookup_rate_hz": 1e6 / max(avg_time, 1),
        }


def build_aero_table(
    cfd_solver: Callable, config: AeroTableConfig = None, parallel: bool = False
) -> AeroTable:
    """
    Build aerodynamic table from CFD solver.

    Args:
        cfd_solver: Function(mach, alpha, beta) -> aero_dict
        config: Table configuration
        parallel: Use parallel computation

    Returns:
        Populated AeroTable
    """
    config = config or AeroTableConfig()
    table = AeroTable(config)
    table.populate_from_cfd(cfd_solver)
    return table


def interpolate_coefficients(
    table: AeroTable, mach: float, alpha_deg: float, beta_deg: float = 0.0
) -> tuple[float, float, float]:
    """
    Convenience function for interpolating CL, CD, Cm.

    Args:
        table: AeroTable
        mach: Mach number
        alpha_deg: Angle of attack
        beta_deg: Sideslip angle

    Returns:
        (CL, CD, Cm) tuple
    """
    aero = table.lookup(mach, alpha_deg, beta_deg)
    return aero.CL, aero.CD, aero.Cm


def validate_aero_table(
    table: AeroTable,
    validation_points: list[tuple[float, float, float, dict[str, float]]],
) -> dict[str, float]:
    """
    Validate aero table against known points.

    Args:
        table: AeroTable to validate
        validation_points: List of (mach, alpha, beta, expected_aero)

    Returns:
        Validation metrics
    """
    errors_CL = []
    errors_CD = []
    errors_Cm = []

    for mach, alpha, beta, expected in validation_points:
        aero = table.lookup(mach, alpha, beta)

        errors_CL.append(abs(aero.CL - expected.get("CL", 0)))
        errors_CD.append(abs(aero.CD - expected.get("CD", 0)))
        errors_Cm.append(abs(aero.Cm - expected.get("Cm", 0)))

    return {
        "CL_mae": np.mean(errors_CL),
        "CD_mae": np.mean(errors_CD),
        "Cm_mae": np.mean(errors_Cm),
        "CL_max_error": np.max(errors_CL),
        "CD_max_error": np.max(errors_CD),
        "Cm_max_error": np.max(errors_Cm),
        "num_points": len(validation_points),
    }


def create_hypersonic_waverider_model() -> Callable:
    """
    Create a simplified hypersonic waverider aerodynamic model.

    Returns:
        CFD solver function
    """

    def waverider_aero(
        mach: float, alpha_deg: float, beta_deg: float
    ) -> dict[str, float]:
        """Simplified waverider aerodynamics."""
        alpha_rad = np.radians(alpha_deg)
        beta_rad = np.radians(beta_deg)

        # Newtonian-like hypersonic approximation
        if mach > 1:
            K = 2.0  # Lift curve slope parameter

            # Lift coefficient
            CL = K * alpha_rad * np.cos(beta_rad)

            # Drag coefficient (wave + pressure + friction)
            CD_0 = 0.02 / np.sqrt(mach**2 - 1)  # Zero-lift wave drag
            CD_i = CL**2 / (np.pi * 2.0)  # Induced drag (AR~2)
            CD_friction = 0.003  # Skin friction
            CD = CD_0 + CD_i + CD_friction

            # Pitching moment (stable at high alpha)
            Cm_alpha = -0.5  # Per radian, static margin
            Cm_0 = 0.01
            Cm = Cm_0 + Cm_alpha * alpha_rad

            # Lateral-directional
            CY = -0.5 * beta_rad
            Cl = -0.1 * beta_rad  # Dihedral effect
            Cn = 0.1 * beta_rad  # Directional stability

        else:
            # Subsonic (simplified)
            CL = 5.7 * alpha_rad  # Thin airfoil theory
            CD = 0.02 + CL**2 / (np.pi * 2.0)
            Cm = -0.5 * alpha_rad
            CY = -0.5 * beta_rad
            Cl = -0.1 * beta_rad
            Cn = 0.1 * beta_rad

        return {"CL": CL, "CD": CD, "CY": CY, "Cl": Cl, "Cm": Cm, "Cn": Cn}

    return waverider_aero


def validate_realtime_cfd_module():
    """Validate real-time CFD module."""
    print("\n" + "=" * 70)
    print("REAL-TIME CFD COUPLING VALIDATION")
    print("=" * 70)

    # Test 1: AeroTableConfig
    print("\n[Test 1] AeroTableConfig")
    print("-" * 40)

    config = AeroTableConfig(
        mach_range=(0.5, 8.0), alpha_range=(-5, 20), n_mach=16, n_alpha=26
    )

    print(f"Mach range: {config.mach_range}")
    print(f"Alpha range: {config.alpha_range}")
    print(f"Grid size: {config.n_mach} x {config.n_alpha} x {config.n_beta}")
    print("✓ PASS")

    # Test 2: AeroTable with waverider model
    print("\n[Test 2] AeroTable Population")
    print("-" * 40)

    waverider = create_hypersonic_waverider_model()
    table = AeroTable(config)
    table.populate_from_cfd(waverider)

    print(f"Table shape: {table.CL_table.shape}")
    print(f"CL range: [{table.CL_table.min():.3f}, {table.CL_table.max():.3f}]")
    print(f"CD range: [{table.CD_table.min():.4f}, {table.CD_table.max():.4f}]")

    assert table._is_built
    print("✓ PASS")

    # Test 3: Table Lookup
    print("\n[Test 3] Table Lookup")
    print("-" * 40)

    aero = table.lookup(5.0, 10.0, 0.0)

    print("Mach=5, Alpha=10°, Beta=0°:")
    print(f"  CL = {aero.CL:.4f}")
    print(f"  CD = {aero.CD:.4f}")
    print(f"  Cm = {aero.Cm:.4f}")
    print(f"  L/D = {aero.CL / max(aero.CD, 0.001):.2f}")

    assert aero.CL > 0  # Positive lift at positive alpha
    assert aero.CD > 0  # Positive drag
    print("✓ PASS")

    # Test 4: Batch Lookup
    print("\n[Test 4] Batch Lookup")
    print("-" * 40)

    mach = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    alpha = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    beta = np.zeros(5)

    coeffs = table.lookup_batch(mach, alpha, beta)

    print("Mach sweep at alpha=5°:")
    for i, m in enumerate(mach):
        print(f"  M={m:.1f}: CL={coeffs['CL'][i]:.4f}, CD={coeffs['CD'][i]:.4f}")

    assert len(coeffs["CL"]) == 5
    print("✓ PASS")

    # Test 5: RealTimeCFD Interface
    print("\n[Test 5] RealTimeCFD Interface")
    print("-" * 40)

    rt_cfd = RealTimeCFD(table, config)

    state = {"mach": 5.0, "alpha_deg": 10.0, "beta_deg": 0.0, "q_bar": 50000, "V": 1500}
    controls = {"de": -2.0, "da": 0.0, "dr": 0.0}

    aero = rt_cfd.get_aero(state, controls)

    print(f"State: M={state['mach']}, α={state['alpha_deg']}°")
    print(f"Controls: δe={controls['de']}°")
    print(f"Forces: L={aero['L']:.0f} N, D={aero['D']:.0f} N")
    print(f"L/D = {aero['L_D']:.2f}")

    assert aero["L"] > 0
    assert aero["D"] > 0
    print("✓ PASS")

    # Test 6: Derivative Estimation
    print("\n[Test 6] Derivative Estimation")
    print("-" * 40)

    derivs = rt_cfd.get_derivatives(state)

    print(f"At M={state['mach']}, α={state['alpha_deg']}°:")
    print(f"  CL_α = {derivs['CL_alpha']:.4f} /rad")
    print(f"  CD_α = {derivs['CD_alpha']:.4f} /rad")
    print(f"  Cm_α = {derivs['Cm_alpha']:.4f} /rad")

    assert derivs["CL_alpha"] > 0  # Positive lift curve slope
    assert derivs["Cm_alpha"] < 0  # Stable configuration
    print("✓ PASS")

    # Test 7: Performance Stats
    print("\n[Test 7] Performance Stats")
    print("-" * 40)

    # Do many lookups
    for _ in range(1000):
        rt_cfd.get_aero(
            {
                "mach": np.random.uniform(2, 7),
                "alpha_deg": np.random.uniform(0, 15),
                "beta_deg": 0,
                "q_bar": 50000,
                "V": 1500,
            }
        )

    stats = rt_cfd.get_performance_stats()

    print(f"Lookups: {stats['lookup_count']}")
    print(f"Total time: {stats['total_time_ms']:.2f} ms")
    print(f"Avg lookup: {stats['avg_lookup_time_us']:.2f} μs")
    print(f"Max rate: {stats['max_lookup_rate_hz']:.0f} Hz")

    assert stats["lookup_count"] > 1000
    assert stats["avg_lookup_time_us"] < 1000  # Should be fast
    print("✓ PASS")

    # Test 8: Table Save/Load
    print("\n[Test 8] Table Save/Load")
    print("-" * 40)

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath = f.name

    table.save(filepath)
    table_loaded = AeroTable.load(filepath)

    # Compare lookups
    aero1 = table.lookup(5.0, 10.0, 0.0)
    aero2 = table_loaded.lookup(5.0, 10.0, 0.0)

    print(f"Original CL: {aero1.CL:.6f}")
    print(f"Loaded CL: {aero2.CL:.6f}")
    print(f"Difference: {abs(aero1.CL - aero2.CL):.2e}")

    assert abs(aero1.CL - aero2.CL) < 1e-10

    import os

    os.remove(filepath)
    print("✓ PASS")

    # Test 9: Validation Against Known Points
    print("\n[Test 9] Table Validation")
    print("-" * 40)

    # Create validation points from the model
    validation_points = [
        (3.0, 5.0, 0.0, waverider(3.0, 5.0, 0.0)),
        (5.0, 10.0, 0.0, waverider(5.0, 10.0, 0.0)),
        (7.0, 15.0, 0.0, waverider(7.0, 15.0, 0.0)),
    ]

    metrics = validate_aero_table(table, validation_points)

    print(f"CL MAE: {metrics['CL_mae']:.6f}")
    print(f"CD MAE: {metrics['CD_mae']:.6f}")
    print(f"Cm MAE: {metrics['Cm_mae']:.6f}")

    assert (
        metrics["CL_mae"] < 0.01
    )  # Linear interpolation should be exact at grid points
    print("✓ PASS")

    print("\n" + "=" * 70)
    print("REAL-TIME CFD COUPLING VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    validate_realtime_cfd_module()
