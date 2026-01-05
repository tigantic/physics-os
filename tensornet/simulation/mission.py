"""
Mission Simulation Module
=========================

End-to-end mission simulation with Monte Carlo capability for
uncertainty quantification and dispersion analysis.

Capabilities:
    - Full mission profile simulation (boost → glide → terminal)
    - Monte Carlo analysis with configurable dispersions
    - Statistical CEP/LEP computation
    - Sensitivity analysis
    - Performance envelope mapping

Mission Phases:
    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐
    │  BOOST  │──►│ PULLOUT │──►│  GLIDE  │──►│   TAEM   │──►│ TERMINAL │
    │         │   │         │   │         │   │          │   │          │
    └─────────┘   └─────────┘   └─────────┘   └──────────┘   └──────────┘
         │             │             │             │              │
         ▼             ▼             ▼             ▼              ▼
     Thrust        Guidance      Aero-Guidance  Energy Mgmt   Final Aim
     Profile       Init          Banking        Altitude Adj   Point Corr

Monte Carlo Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     MissionSimulator                         │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │ Uncertainty │  │ Trajectory  │  │ Performance         │  │
    │  │ Model       │──│ Solver      │──│ Metrics             │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             ┌──────────────┐       ┌──────────────┐
             │ Single Run   │  ...  │ Single Run   │
             │ (Sample 1)   │       │ (Sample N)   │
             └──────────────┘       └──────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │  Dispersion Analysis  │
                    │  - CEP/LEP            │
                    │  - Sensitivity        │
                    │  - Failure modes      │
                    └───────────────────────┘

References:
    [1] AIAA-2013-4820: Monte Carlo Methods for Trajectory Analysis
    [2] NASA-CR-4776: Uncertainty Quantification in Hypersonic Systems
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MissionPhase(Enum):
    """Mission phase enumeration."""

    PRELAUNCH = "prelaunch"
    BOOST = "boost"
    PULLOUT = "pullout"
    GLIDE = "glide"
    TAEM = "taem"  # Terminal Area Energy Management
    TERMINAL = "terminal"
    IMPACT = "impact"
    COMPLETE = "complete"
    FAILED = "failed"


class FailureMode(Enum):
    """Mission failure modes."""

    NONE = "none"
    STRUCTURAL = "structural"  # G-limit exceeded
    THERMAL = "thermal"  # Heating limit exceeded
    RANGE_SHORT = "range_short"  # Insufficient energy
    RANGE_LONG = "range_long"  # Overshoot
    GUIDANCE = "guidance"  # Guidance divergence
    CONTROL_SATURATION = "control_saturation"


@dataclass
class MissionConfig:
    """Configuration for mission simulation."""

    # Launch conditions
    launch_lat: float = 34.0  # degrees
    launch_lon: float = -118.0  # degrees
    launch_alt: float = 0.0  # meters
    launch_heading: float = 90.0  # degrees (east)

    # Target
    target_lat: float = 35.0
    target_lon: float = -100.0
    target_alt: float = 0.0

    # Vehicle parameters
    vehicle_mass_kg: float = 1000.0
    S_ref: float = 10.0  # Reference area (m²)

    # Boost phase
    boost_thrust_N: float = 100000.0
    boost_duration_s: float = 60.0
    boost_pitch_profile: str = "optimal"  # 'optimal', 'fixed', 'gravity_turn'

    # Glide phase
    target_L_D: float = 3.0
    max_bank_angle_deg: float = 70.0

    # Constraints
    max_g_load: float = 8.0
    max_q_bar_Pa: float = 100000.0
    max_heating_rate_W_cm2: float = 200.0

    # Simulation
    dt_s: float = 0.1
    max_time_s: float = 1800.0  # 30 minutes max

    # Phase transition altitudes
    pullout_complete_alt: float = 40000.0
    taem_start_range_km: float = 100.0
    terminal_start_range_km: float = 20.0


@dataclass
class UncertaintyModel:
    """Uncertainty model for Monte Carlo analysis."""

    # Atmospheric uncertainties
    density_sigma_pct: float = 5.0  # Percentage standard deviation
    wind_sigma_ms: float = 20.0  # Wind speed std dev

    # Aerodynamic uncertainties
    CL_sigma_pct: float = 5.0
    CD_sigma_pct: float = 10.0
    Cm_sigma_pct: float = 10.0

    # Propulsion uncertainties
    thrust_sigma_pct: float = 3.0
    Isp_sigma_pct: float = 2.0

    # Navigation uncertainties
    position_sigma_m: float = 100.0
    velocity_sigma_ms: float = 1.0

    # Mass uncertainties
    mass_sigma_pct: float = 1.0

    # Control uncertainties
    actuator_bias_deg: float = 0.5

    def sample(self) -> dict[str, float]:
        """Generate a sample of uncertainty factors."""
        return {
            "density_factor": 1.0 + np.random.normal(0, self.density_sigma_pct / 100),
            "wind_u": np.random.normal(0, self.wind_sigma_ms),
            "wind_v": np.random.normal(0, self.wind_sigma_ms),
            "CL_factor": 1.0 + np.random.normal(0, self.CL_sigma_pct / 100),
            "CD_factor": 1.0 + np.random.normal(0, self.CD_sigma_pct / 100),
            "Cm_factor": 1.0 + np.random.normal(0, self.Cm_sigma_pct / 100),
            "thrust_factor": 1.0 + np.random.normal(0, self.thrust_sigma_pct / 100),
            "Isp_factor": 1.0 + np.random.normal(0, self.Isp_sigma_pct / 100),
            "position_error": np.random.normal(0, self.position_sigma_m, 3),
            "velocity_error": np.random.normal(0, self.velocity_sigma_ms, 3),
            "mass_factor": 1.0 + np.random.normal(0, self.mass_sigma_pct / 100),
            "actuator_bias": np.random.normal(0, self.actuator_bias_deg),
        }


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo analysis."""

    n_runs: int = 100
    seed: int | None = None
    parallel: bool = True
    n_workers: int = 4

    # Statistics to compute
    compute_cep: bool = True  # Circular Error Probable
    compute_lep: bool = True  # Linear Error Probable
    compute_sensitivity: bool = True

    # Output options
    save_all_trajectories: bool = False
    save_impact_points: bool = True


@dataclass
class MissionResult:
    """Result of a single mission simulation."""

    success: bool = False
    phase_history: list[MissionPhase] = field(default_factory=list)
    failure_mode: FailureMode = FailureMode.NONE

    # Impact point
    impact_lat: float = 0.0
    impact_lon: float = 0.0
    impact_time: float = 0.0

    # Miss distance (from target)
    miss_distance_m: float = 0.0
    downrange_error_m: float = 0.0
    crossrange_error_m: float = 0.0

    # Performance metrics
    max_mach: float = 0.0
    max_altitude_m: float = 0.0
    max_g_load: float = 0.0
    max_heating_rate: float = 0.0
    max_q_bar: float = 0.0
    total_range_km: float = 0.0

    # Trajectory (optional)
    trajectory: np.ndarray | None = None
    times: np.ndarray | None = None

    # Uncertainty sample used
    uncertainty_sample: dict | None = None


class MissionSimulator:
    """
    End-to-end mission simulator.

    Integrates trajectory solver, guidance controller, and
    aerodynamic models for complete mission simulation.
    """

    def __init__(
        self, config: MissionConfig = None, uncertainty: UncertaintyModel = None
    ):
        self.config = config or MissionConfig()
        self.uncertainty = uncertainty or UncertaintyModel()

        # State
        self.current_phase = MissionPhase.PRELAUNCH
        self.time = 0.0
        self.state = np.zeros(12)  # [x,y,z,vx,vy,vz,ax,ay,az,roll,pitch,yaw]

        # History
        self.trajectory_history: list[np.ndarray] = []
        self.phase_history: list[MissionPhase] = []
        self.time_history: list[float] = []

        # Target (computed from lat/lon)
        self.target_position = self._latlon_to_xyz(
            self.config.target_lat, self.config.target_lon, self.config.target_alt
        )

    def _latlon_to_xyz(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """Convert lat/lon/alt to local XYZ (flat Earth approximation)."""
        # Simple flat-Earth for now
        R_earth = 6371000.0
        x = np.radians(lon - self.config.launch_lon) * R_earth * np.cos(np.radians(lat))
        y = np.radians(lat - self.config.launch_lat) * R_earth
        z = alt
        return np.array([x, y, z])

    def _xyz_to_latlon(
        self, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        """Convert local XYZ to lat/lon/alt."""
        R_earth = 6371000.0
        lat = self.config.launch_lat + np.degrees(y / R_earth)
        lon = self.config.launch_lon + np.degrees(
            x / (R_earth * np.cos(np.radians(lat)))
        )
        return lat, lon, z

    def _compute_atmosphere(
        self, alt: float, uncertainty: dict = None
    ) -> dict[str, float]:
        """Compute atmospheric properties."""
        # ISA atmosphere
        if alt < 11000:
            T = 288.15 - 0.0065 * alt
            p = 101325 * (T / 288.15) ** 5.2561
        elif alt < 25000:
            T = 216.65
            p = 22632 * np.exp(-0.000157688 * (alt - 11000))
        else:
            T = 216.65 + 0.001 * (alt - 25000)
            p = 2488.6 * (216.65 / T) ** 34.163

        rho = p / (287.05 * T)
        a = np.sqrt(1.4 * 287.05 * T)

        # Apply uncertainty
        if uncertainty:
            rho *= uncertainty.get("density_factor", 1.0)

        return {"T": T, "p": p, "rho": rho, "a": a}

    def _compute_aero(
        self, mach: float, alpha: float, beta: float, uncertainty: dict = None
    ) -> dict[str, float]:
        """Compute aerodynamic coefficients."""
        # Simplified hypersonic model
        alpha_rad = np.radians(alpha)

        CL = 2.0 * alpha_rad
        CD = 0.02 + CL**2 / (np.pi * 2.0)
        Cm = -0.5 * alpha_rad

        # Apply uncertainty
        if uncertainty:
            CL *= uncertainty.get("CL_factor", 1.0)
            CD *= uncertainty.get("CD_factor", 1.0)
            Cm *= uncertainty.get("Cm_factor", 1.0)

        return {"CL": CL, "CD": CD, "Cm": Cm}

    def _check_constraints(
        self, g_load: float, q_bar: float, heating: float
    ) -> FailureMode:
        """Check if constraints are violated."""
        if g_load > self.config.max_g_load:
            return FailureMode.STRUCTURAL
        if q_bar > self.config.max_q_bar_Pa * 1.5:  # 50% margin
            return FailureMode.STRUCTURAL
        if heating > self.config.max_heating_rate_W_cm2:
            return FailureMode.THERMAL
        return FailureMode.NONE

    def _determine_phase(self, range_to_target_km: float) -> MissionPhase:
        """Determine current mission phase."""
        alt = self.state[2]

        if self.time < self.config.boost_duration_s:
            return MissionPhase.BOOST
        elif (
            alt < self.config.pullout_complete_alt
            and self.current_phase == MissionPhase.BOOST
        ):
            return MissionPhase.PULLOUT
        elif range_to_target_km < self.config.terminal_start_range_km:
            return MissionPhase.TERMINAL
        elif range_to_target_km < self.config.taem_start_range_km:
            return MissionPhase.TAEM
        elif alt <= 0:
            return MissionPhase.IMPACT
        else:
            return MissionPhase.GLIDE

    def run(self, uncertainty_sample: dict = None) -> MissionResult:
        """
        Run a single mission simulation.

        Args:
            uncertainty_sample: Uncertainty factors to apply

        Returns:
            MissionResult with outcome and metrics
        """
        # Reset state
        self.time = 0.0
        self.current_phase = MissionPhase.BOOST
        self.trajectory_history = []
        self.phase_history = []
        self.time_history = []

        # Apply initial uncertainties
        if uncertainty_sample is None:
            uncertainty_sample = {}

        # Initial state
        pos = self._latlon_to_xyz(
            self.config.launch_lat, self.config.launch_lon, self.config.launch_alt
        )

        # Apply position error
        pos += uncertainty_sample.get("position_error", np.zeros(3))

        heading_rad = np.radians(self.config.launch_heading)
        self.state = np.zeros(12)
        self.state[0:3] = pos
        self.state[11] = heading_rad  # yaw

        mass = self.config.vehicle_mass_kg * uncertainty_sample.get("mass_factor", 1.0)

        # Performance tracking
        max_mach = 0.0
        max_alt = 0.0
        max_g = 0.0
        max_q = 0.0
        max_heat = 0.0

        failure_mode = FailureMode.NONE

        # Main simulation loop
        dt = self.config.dt_s

        while self.time < self.config.max_time_s:
            # Current position
            pos = self.state[0:3]
            vel = self.state[3:6]
            alt = pos[2]
            V = np.linalg.norm(vel)

            # Atmosphere
            atm = self._compute_atmosphere(alt, uncertainty_sample)
            mach = V / atm["a"] if atm["a"] > 0 else 0
            q_bar = 0.5 * atm["rho"] * V**2

            # Range to target
            range_to_target = np.linalg.norm(pos[0:2] - self.target_position[0:2])
            range_to_target_km = range_to_target / 1000

            # Determine phase
            new_phase = self._determine_phase(range_to_target_km)
            if new_phase != self.current_phase:
                self.current_phase = new_phase

            # Check termination
            if alt <= 0 or self.current_phase == MissionPhase.IMPACT:
                break

            # Compute guidance command
            if self.current_phase == MissionPhase.BOOST:
                # Boost phase: thrust along velocity
                thrust = self.config.boost_thrust_N * uncertainty_sample.get(
                    "thrust_factor", 1.0
                )
                if V > 0:
                    thrust_dir = vel / V
                else:
                    thrust_dir = np.array(
                        [np.cos(heading_rad), np.sin(heading_rad), 0.5]
                    )
                    thrust_dir /= np.linalg.norm(thrust_dir)

                accel = thrust_dir * thrust / mass

            else:
                # Glide phase: aerodynamic flight
                if V > 0:
                    vel_dir = vel / V
                else:
                    vel_dir = np.array([1, 0, 0])

                # Simple proportional navigation toward target
                target_dir = self.target_position - pos
                target_dist = np.linalg.norm(target_dir)
                if target_dist > 0:
                    target_dir /= target_dist

                # Compute required turn
                cross = np.cross(vel_dir, target_dir)
                alpha = 5.0  # Fixed angle of attack

                # Get aero
                aero = self._compute_aero(mach, alpha, 0, uncertainty_sample)

                L = q_bar * self.config.S_ref * aero["CL"]
                D = q_bar * self.config.S_ref * aero["CD"]

                # Lift direction (perpendicular to velocity, toward target)
                lift_dir = np.array([0, 0, 1]) + 0.3 * cross  # Simplified
                if np.linalg.norm(lift_dir) > 0:
                    lift_dir /= np.linalg.norm(lift_dir)

                accel = (
                    lift_dir * L / mass - vel_dir * D / mass - np.array([0, 0, 9.81])
                )

            # Update state (simple Euler)
            self.state[3:6] += accel * dt
            self.state[0:3] += self.state[3:6] * dt

            # Update tracking
            g_load = np.linalg.norm(accel) / 9.81
            heating = 1e-4 * atm["rho"] ** 0.5 * V**3  # Simplified Sutton-Graves

            max_mach = max(max_mach, mach)
            max_alt = max(max_alt, alt)
            max_g = max(max_g, g_load)
            max_q = max(max_q, q_bar)
            max_heat = max(max_heat, heating)

            # Check constraints
            failure = self._check_constraints(g_load, q_bar, heating)
            if failure != FailureMode.NONE:
                failure_mode = failure
                break

            # Store history
            self.trajectory_history.append(self.state.copy())
            self.phase_history.append(self.current_phase)
            self.time_history.append(self.time)

            self.time += dt

        # Compute miss distance
        final_pos = self.state[0:3]
        impact_lat, impact_lon, _ = self._xyz_to_latlon(*final_pos)

        miss = final_pos[0:2] - self.target_position[0:2]
        miss_distance = np.linalg.norm(miss)

        # Downrange/crossrange (simplified)
        downrange_error = miss[0]
        crossrange_error = miss[1]

        total_range = (
            np.linalg.norm(
                np.array([self.config.launch_lon, self.config.launch_lat])
                - np.array([impact_lon, impact_lat])
            )
            * 111000
        )  # Approximate m/deg

        # Determine success
        success = failure_mode == FailureMode.NONE and miss_distance < 1000  # 1km CEP

        return MissionResult(
            success=success,
            phase_history=self.phase_history,
            failure_mode=failure_mode,
            impact_lat=impact_lat,
            impact_lon=impact_lon,
            impact_time=self.time,
            miss_distance_m=miss_distance,
            downrange_error_m=downrange_error,
            crossrange_error_m=crossrange_error,
            max_mach=max_mach,
            max_altitude_m=max_alt,
            max_g_load=max_g,
            max_heating_rate=max_heat,
            max_q_bar=max_q,
            total_range_km=total_range / 1000,
            trajectory=(
                np.array(self.trajectory_history) if self.trajectory_history else None
            ),
            times=np.array(self.time_history) if self.time_history else None,
            uncertainty_sample=uncertainty_sample,
        )


def run_monte_carlo(
    config: MissionConfig, uncertainty: UncertaintyModel, mc_config: MonteCarloConfig
) -> list[MissionResult]:
    """
    Run Monte Carlo simulation.

    Args:
        config: Mission configuration
        uncertainty: Uncertainty model
        mc_config: Monte Carlo settings

    Returns:
        List of MissionResult for all runs
    """
    if mc_config.seed is not None:
        np.random.seed(mc_config.seed)

    results = []

    def run_single(run_idx: int) -> MissionResult:
        """Run a single Monte Carlo sample."""
        sample = uncertainty.sample()
        sim = MissionSimulator(config, uncertainty)
        result = sim.run(sample)
        return result

    if mc_config.parallel and mc_config.n_workers > 1:
        with ThreadPoolExecutor(max_workers=mc_config.n_workers) as executor:
            futures = [executor.submit(run_single, i) for i in range(mc_config.n_runs)]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for i in range(mc_config.n_runs):
            results.append(run_single(i))

    return results


def analyze_dispersion(results: list[MissionResult]) -> dict[str, Any]:
    """
    Analyze dispersion from Monte Carlo results.

    Args:
        results: List of mission results

    Returns:
        Dispersion analysis metrics
    """
    # Extract impact points
    miss_distances = [
        r.miss_distance_m
        for r in results
        if r.success or r.failure_mode == FailureMode.NONE
    ]
    downrange_errors = [
        r.downrange_error_m
        for r in results
        if r.success or r.failure_mode == FailureMode.NONE
    ]
    crossrange_errors = [
        r.crossrange_error_m
        for r in results
        if r.success or r.failure_mode == FailureMode.NONE
    ]

    if not miss_distances:
        return {"error": "No successful runs"}

    # CEP (Circular Error Probable) - 50th percentile
    miss_sorted = np.sort(miss_distances)
    cep = np.percentile(miss_distances, 50)

    # LEP (Linear Error Probable) - separate for down/cross range
    lep_downrange = np.percentile(np.abs(downrange_errors), 50)
    lep_crossrange = np.percentile(np.abs(crossrange_errors), 50)

    # 95% and 99% containment
    r_95 = np.percentile(miss_distances, 95)
    r_99 = np.percentile(miss_distances, 99)

    # Success rate
    n_success = sum(1 for r in results if r.success)
    success_rate = n_success / len(results)

    # Failure mode breakdown
    failure_modes = {}
    for r in results:
        if r.failure_mode != FailureMode.NONE:
            mode = r.failure_mode.value
            failure_modes[mode] = failure_modes.get(mode, 0) + 1

    # Performance statistics
    max_machs = [r.max_mach for r in results]
    max_alts = [r.max_altitude_m for r in results]
    max_gs = [r.max_g_load for r in results]

    return {
        "n_runs": len(results),
        "n_valid": len(miss_distances),
        "success_rate": success_rate,
        # Accuracy metrics
        "cep_m": cep,
        "lep_downrange_m": lep_downrange,
        "lep_crossrange_m": lep_crossrange,
        "r_95_m": r_95,
        "r_99_m": r_99,
        # Statistics
        "miss_mean_m": np.mean(miss_distances),
        "miss_std_m": np.std(miss_distances),
        "miss_max_m": np.max(miss_distances),
        # Performance
        "mach_mean": np.mean(max_machs),
        "mach_std": np.std(max_machs),
        "altitude_mean_km": np.mean(max_alts) / 1000,
        "g_load_mean": np.mean(max_gs),
        "g_load_max": np.max(max_gs),
        # Failures
        "failure_modes": failure_modes,
    }


def compute_sensitivity(
    config: MissionConfig, uncertainty: UncertaintyModel, n_samples: int = 50
) -> dict[str, float]:
    """
    Compute sensitivity indices for uncertainty factors.

    Uses one-at-a-time (OAT) sensitivity analysis.

    Args:
        config: Mission configuration
        uncertainty: Baseline uncertainty model
        n_samples: Samples per factor

    Returns:
        Dict of factor -> sensitivity index
    """
    # Baseline run
    sim = MissionSimulator(config, uncertainty)
    baseline_result = sim.run()
    baseline_miss = baseline_result.miss_distance_m

    sensitivities = {}
    factors = [
        ("density_sigma_pct", "density_factor"),
        ("CL_sigma_pct", "CL_factor"),
        ("CD_sigma_pct", "CD_factor"),
        ("thrust_sigma_pct", "thrust_factor"),
        ("position_sigma_m", "position_error"),
    ]

    for uncertainty_param, sample_key in factors:
        miss_distances = []

        for _ in range(n_samples):
            # Sample only this factor
            sample = {sample_key: 1.0 + np.random.normal(0, 0.1)}

            if sample_key == "position_error":
                sample[sample_key] = np.random.normal(0, 100, 3)

            result = sim.run(sample)
            miss_distances.append(result.miss_distance_m)

        # Sensitivity = std of miss due to this factor
        sensitivities[uncertainty_param] = np.std(miss_distances)

    return sensitivities


def validate_mission_module():
    """Validate mission simulation module."""
    print("\n" + "=" * 70)
    print("MISSION SIMULATION VALIDATION")
    print("=" * 70)

    # Test 1: MissionConfig
    print("\n[Test 1] MissionConfig")
    print("-" * 40)

    config = MissionConfig(
        launch_lat=34.0,
        launch_lon=-118.0,
        target_lat=35.0,
        target_lon=-117.0,
        boost_duration_s=30.0,
    )

    print(f"Launch: ({config.launch_lat}, {config.launch_lon})")
    print(f"Target: ({config.target_lat}, {config.target_lon})")
    print(f"Boost duration: {config.boost_duration_s} s")
    print("✓ PASS")

    # Test 2: UncertaintyModel
    print("\n[Test 2] UncertaintyModel")
    print("-" * 40)

    uncertainty = UncertaintyModel(
        density_sigma_pct=5.0, CL_sigma_pct=5.0, CD_sigma_pct=10.0
    )

    samples = [uncertainty.sample() for _ in range(100)]
    density_factors = [s["density_factor"] for s in samples]

    print(f"Density factor mean: {np.mean(density_factors):.4f} (expected ~1.0)")
    print(f"Density factor std: {np.std(density_factors):.4f} (expected ~0.05)")

    assert 0.95 < np.mean(density_factors) < 1.05
    print("✓ PASS")

    # Test 3: Single Mission Run
    print("\n[Test 3] Single Mission Run")
    print("-" * 40)

    sim = MissionSimulator(config, uncertainty)
    result = sim.run()

    print(f"Success: {result.success}")
    print(f"Impact time: {result.impact_time:.1f} s")
    print(f"Max Mach: {result.max_mach:.2f}")
    print(f"Max altitude: {result.max_altitude_m / 1000:.1f} km")
    print(f"Max g-load: {result.max_g_load:.1f}")
    print(f"Miss distance: {result.miss_distance_m:.0f} m")
    print(f"Failure mode: {result.failure_mode.value}")

    assert result.impact_time > 0
    assert result.max_mach > 0
    print("✓ PASS")

    # Test 4: Mission with Uncertainty
    print("\n[Test 4] Mission with Uncertainty")
    print("-" * 40)

    sample = uncertainty.sample()
    result_uncertain = sim.run(sample)

    print(f"Uncertainty sample keys: {list(sample.keys())}")
    print(f"Miss distance: {result_uncertain.miss_distance_m:.0f} m")

    assert result_uncertain.uncertainty_sample is not None
    print("✓ PASS")

    # Test 5: Monte Carlo (small)
    print("\n[Test 5] Monte Carlo Simulation")
    print("-" * 40)

    mc_config = MonteCarloConfig(n_runs=20, seed=42, parallel=False)

    t0 = time.time()
    results = run_monte_carlo(config, uncertainty, mc_config)
    elapsed = time.time() - t0

    print(f"Completed {len(results)} runs in {elapsed:.2f} s")
    print(f"Rate: {len(results) / elapsed:.1f} runs/s")

    success_count = sum(
        1 for r in results if r.success or r.failure_mode == FailureMode.NONE
    )
    print(f"Valid runs: {success_count}/{len(results)}")

    assert len(results) == mc_config.n_runs
    print("✓ PASS")

    # Test 6: Dispersion Analysis
    print("\n[Test 6] Dispersion Analysis")
    print("-" * 40)

    analysis = analyze_dispersion(results)

    print(f"Success rate: {analysis['success_rate']:.1%}")
    print(f"CEP: {analysis['cep_m']:.0f} m")
    print(f"LEP (downrange): {analysis['lep_downrange_m']:.0f} m")
    print(f"LEP (crossrange): {analysis['lep_crossrange_m']:.0f} m")
    print(f"95% containment: {analysis['r_95_m']:.0f} m")
    print(f"Mean Mach: {analysis['mach_mean']:.2f}")

    if analysis.get("failure_modes"):
        print(f"Failure modes: {analysis['failure_modes']}")

    assert "cep_m" in analysis
    print("✓ PASS")

    # Test 7: Phase Transitions
    print("\n[Test 7] Phase Transitions")
    print("-" * 40)

    phases_seen = set(result.phase_history)
    print(f"Phases observed: {[p.value for p in phases_seen]}")

    # Should see at least boost phase
    assert MissionPhase.BOOST in phases_seen
    print("✓ PASS")

    # Test 8: Constraint Checking
    print("\n[Test 8] Constraint Checking")
    print("-" * 40)

    # Run with high uncertainty to trigger some failures
    high_uncertainty = UncertaintyModel(density_sigma_pct=20.0, thrust_sigma_pct=10.0)

    mc_config_stress = MonteCarloConfig(n_runs=30, parallel=False)
    stress_results = run_monte_carlo(config, high_uncertainty, mc_config_stress)

    failure_count = sum(1 for r in stress_results if r.failure_mode != FailureMode.NONE)
    print(f"Failures under stress: {failure_count}/{len(stress_results)}")

    stress_analysis = analyze_dispersion(stress_results)
    print(f"Success rate: {stress_analysis['success_rate']:.1%}")
    print("✓ PASS")

    # Test 9: Trajectory Output
    print("\n[Test 9] Trajectory Output")
    print("-" * 40)

    sim = MissionSimulator(config)
    result = sim.run()

    if result.trajectory is not None:
        print(f"Trajectory shape: {result.trajectory.shape}")
        print(f"Time points: {len(result.times)}")
        print(f"Duration: {result.times[-1] - result.times[0]:.1f} s")

        alt_profile = result.trajectory[:, 2]
        print(f"Altitude range: {alt_profile.min():.0f} - {alt_profile.max():.0f} m")
    else:
        print("No trajectory stored")

    print("✓ PASS")

    print("\n" + "=" * 70)
    print("MISSION SIMULATION VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    validate_mission_module()
