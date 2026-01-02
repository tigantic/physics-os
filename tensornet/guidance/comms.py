"""
Communications Blackout Management and Smart Antenna Switching
===============================================================

Phase 22: Cognitive communications for hypersonic reentry.

This module implements antenna management and frequency selection
strategies to maintain communication during plasma blackout conditions.

Key Features:
- Antenna array modeling with radiation patterns
- Blackout map computation from plasma density fields
- Cognitive antenna switching based on local attenuation
- Frequency hopping to stay above plasma cutoff
- Maneuver recommendations to expose low-attenuation antennas

References:
    - Rybak & Churchill, "Progress in Reentry Communications" (1971)
    - NASA TM-2014-216634, "Plasma Blackout Mitigation" (2014)
    - Hartunian et al., "Implications of the SHARP Flight" (1962)

Constitution Compliance: Article II.1, Article V
"""

import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

# =============================================================================
# Data Structures
# =============================================================================


class AntennaType(Enum):
    """Types of antennas on hypersonic vehicle."""

    PATCH = "patch"  # Conformal patch antenna
    SLOT = "slot"  # Slot antenna (flush mount)
    BLADE = "blade"  # Blade antenna (if subsonic portion)
    HORN = "horn"  # Horn antenna
    PHASED_ARRAY = "phased"  # Phased array


@dataclass
class Antenna:
    """
    Single antenna specification.

    Attributes:
        id: Unique identifier
        position: (x, y, z) position on vehicle body frame (m)
        normal: (nx, ny, nz) antenna boresight direction
        antenna_type: Type of antenna
        frequency_range: (f_min, f_max) operating frequencies (Hz)
        gain_dB: Peak gain (dB)
        beamwidth_deg: Half-power beamwidth (degrees)
    """

    id: str
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    antenna_type: AntennaType = AntennaType.PATCH
    frequency_range: tuple[float, float] = (1e9, 10e9)
    gain_dB: float = 5.0
    beamwidth_deg: float = 60.0


@dataclass
class AntennaArray:
    """
    Collection of antennas on a vehicle.

    Typical hypersonic vehicle has antennas at:
    - Nose (forward comm)
    - Aft body (rear comm)
    - Sides (crosslink)
    - Sometimes in windows/hatches
    """

    antennas: list[Antenna]

    def __post_init__(self):
        """Build lookup table."""
        self._by_id = {a.id: a for a in self.antennas}

    def get(self, antenna_id: str) -> Antenna | None:
        """Get antenna by ID."""
        return self._by_id.get(antenna_id)

    @property
    def positions(self) -> Tensor:
        """Get all antenna positions as tensor (N, 3)."""
        return torch.tensor([a.position for a in self.antennas], dtype=torch.float64)

    @property
    def normals(self) -> Tensor:
        """Get all antenna normals as tensor (N, 3)."""
        return torch.tensor([a.normal for a in self.antennas], dtype=torch.float64)

    @classmethod
    def typical_reentry_vehicle(cls) -> "AntennaArray":
        """Create typical antenna array for reentry vehicle."""
        return cls(
            antennas=[
                Antenna(
                    id="nose_fwd",
                    position=(2.0, 0.0, 0.0),
                    normal=(1.0, 0.0, 0.0),
                    antenna_type=AntennaType.PATCH,
                    frequency_range=(2e9, 8e9),
                    gain_dB=6.0,
                ),
                Antenna(
                    id="aft_port",
                    position=(-1.5, -0.3, 0.0),
                    normal=(-0.707, -0.707, 0.0),
                    antenna_type=AntennaType.SLOT,
                    frequency_range=(1e9, 6e9),
                    gain_dB=4.0,
                ),
                Antenna(
                    id="aft_starboard",
                    position=(-1.5, 0.3, 0.0),
                    normal=(-0.707, 0.707, 0.0),
                    antenna_type=AntennaType.SLOT,
                    frequency_range=(1e9, 6e9),
                    gain_dB=4.0,
                ),
                Antenna(
                    id="dorsal",
                    position=(0.0, 0.0, 0.3),
                    normal=(0.0, 0.0, 1.0),
                    antenna_type=AntennaType.PATCH,
                    frequency_range=(2e9, 12e9),
                    gain_dB=5.0,
                ),
                Antenna(
                    id="ventral",
                    position=(0.0, 0.0, -0.3),
                    normal=(0.0, 0.0, -1.0),
                    antenna_type=AntennaType.PATCH,
                    frequency_range=(2e9, 12e9),
                    gain_dB=5.0,
                ),
            ]
        )


@dataclass
class BlackoutMap:
    """
    Attenuation map across all antennas.

    Attributes:
        antenna_ids: List of antenna IDs
        attenuation_dB: Attenuation at each antenna (dB)
        plasma_frequency: Local plasma frequency at each antenna (rad/s)
        is_blackout: Boolean flag for each antenna
        best_antenna_id: ID of antenna with lowest attenuation
        min_attenuation_dB: Minimum attenuation across antennas
    """

    antenna_ids: list[str]
    attenuation_dB: Tensor
    plasma_frequency: Tensor
    is_blackout: Tensor

    @property
    def best_antenna_id(self) -> str:
        """Find antenna with lowest attenuation."""
        idx = torch.argmin(self.attenuation_dB).item()
        return self.antenna_ids[int(idx)]

    @property
    def min_attenuation_dB(self) -> float:
        """Minimum attenuation across all antennas."""
        return self.attenuation_dB.min().item()

    @property
    def all_blackout(self) -> bool:
        """Check if all antennas are in blackout."""
        return self.is_blackout.all().item()

    def ranked_antennas(self) -> list[tuple[str, float]]:
        """Get antennas ranked by attenuation (best first)."""
        indices = torch.argsort(self.attenuation_dB)
        return [
            (self.antenna_ids[int(i)], self.attenuation_dB[i].item()) for i in indices
        ]


@dataclass
class FrequencyRecommendation:
    """Recommended frequency for communication."""

    frequency_Hz: float
    margin_dB: float
    above_cutoff: bool
    reason: str


@dataclass
class ManeuverRecommendation:
    """Recommended attitude maneuver for comm improvement."""

    target_antenna_id: str
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    expected_improvement_dB: float
    reason: str


# =============================================================================
# Blackout Map Computation
# =============================================================================


def compute_blackout_map(
    antenna_array: AntennaArray,
    electron_density_field: Tensor,
    field_positions: Tensor,
    signal_frequency: float = 2.4e9,
) -> BlackoutMap:
    """
    Compute blackout map for antenna array given plasma field.

    Interpolates electron density to antenna positions and computes
    local attenuation for each antenna.

    Args:
        antenna_array: Antenna configuration
        electron_density_field: Electron density at grid points (m^-3)
        field_positions: Positions of grid points (N, 3)
        signal_frequency: Communication frequency (Hz)

    Returns:
        BlackoutMap with attenuation at each antenna
    """
    from tensornet.cfd.plasma import is_blackout, plasma_frequency, rf_attenuation

    n_antennas = len(antenna_array.antennas)
    antenna_positions = antenna_array.positions

    # Interpolate electron density to antenna positions
    # (simplified: nearest neighbor for now)
    n_e_at_antennas = torch.zeros(n_antennas, dtype=torch.float64)

    for i, pos in enumerate(antenna_positions):
        # Find nearest grid point
        distances = torch.norm(field_positions - pos.unsqueeze(0), dim=1)
        nearest_idx = torch.argmin(distances)
        n_e_at_antennas[i] = electron_density_field.flatten()[nearest_idx]

    # Compute plasma frequency at each antenna
    omega_pe = plasma_frequency(n_e_at_antennas)

    # Signal angular frequency
    omega_signal = 2 * math.pi * signal_frequency

    # Compute attenuation
    atten = rf_attenuation(torch.tensor(omega_signal), omega_pe)

    # Determine blackout
    blackout = is_blackout(torch.tensor(omega_signal), omega_pe)

    return BlackoutMap(
        antenna_ids=[a.id for a in antenna_array.antennas],
        attenuation_dB=atten,
        plasma_frequency=omega_pe,
        is_blackout=blackout,
    )


def compute_blackout_map_from_cfd(
    antenna_array: AntennaArray,
    temperature_field: Tensor,
    pressure_field: Tensor,
    field_positions: Tensor,
    signal_frequency: float = 2.4e9,
) -> BlackoutMap:
    """
    Compute blackout map directly from CFD solution.

    Args:
        antenna_array: Antenna configuration
        temperature_field: Temperature at grid points (K)
        pressure_field: Pressure at grid points (Pa)
        field_positions: Positions of grid points (N, 3)
        signal_frequency: Communication frequency (Hz)

    Returns:
        BlackoutMap with attenuation at each antenna
    """
    from tensornet.cfd.plasma import electron_density_field

    n_e = electron_density_field(temperature_field, pressure_field)

    return compute_blackout_map(antenna_array, n_e, field_positions, signal_frequency)


# =============================================================================
# Cognitive Communications
# =============================================================================


class CognitiveComms:
    """
    Cognitive communications controller for blackout mitigation.

    Implements smart antenna selection, frequency hopping, and
    maneuver recommendations to maintain communication during
    plasma blackout conditions.

    Attributes:
        antenna_array: Vehicle antenna configuration
        current_antenna_id: Currently selected antenna
        current_frequency: Current operating frequency (Hz)
        history: Log of switching decisions
    """

    def __init__(
        self,
        antenna_array: AntennaArray,
        default_frequency: float = 2.4e9,
        min_frequency: float = 1e9,
        max_frequency: float = 20e9,
        hysteresis_dB: float = 3.0,
    ):
        """
        Initialize cognitive comms controller.

        Args:
            antenna_array: Antenna configuration
            default_frequency: Default operating frequency (Hz)
            min_frequency: Minimum available frequency (Hz)
            max_frequency: Maximum available frequency (Hz)
            hysteresis_dB: Hysteresis for antenna switching (dB)
        """
        self.antenna_array = antenna_array
        self.current_antenna_id = antenna_array.antennas[0].id
        self.current_frequency = default_frequency
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.hysteresis_dB = hysteresis_dB

        self.history: list[dict] = []

    def select_antenna(self, blackout_map: BlackoutMap) -> str:
        """
        Select best antenna based on current blackout conditions.

        Implements hysteresis to avoid rapid switching.

        Args:
            blackout_map: Current attenuation at each antenna

        Returns:
            Selected antenna ID
        """
        # Get current antenna's attenuation
        current_idx = blackout_map.antenna_ids.index(self.current_antenna_id)
        current_atten = blackout_map.attenuation_dB[current_idx].item()

        # Find best antenna
        best_id = blackout_map.best_antenna_id
        best_idx = blackout_map.antenna_ids.index(best_id)
        best_atten = blackout_map.attenuation_dB[best_idx].item()

        # Switch only if improvement exceeds hysteresis
        if best_atten < current_atten - self.hysteresis_dB:
            self.current_antenna_id = best_id
            self.history.append(
                {
                    "action": "antenna_switch",
                    "from": self.current_antenna_id,
                    "to": best_id,
                    "improvement_dB": current_atten - best_atten,
                }
            )

        return self.current_antenna_id

    def frequency_hop(self, local_omega_pe: float) -> FrequencyRecommendation:
        """
        Recommend frequency to stay above plasma cutoff.

        Args:
            local_omega_pe: Local plasma frequency at current antenna (rad/s)

        Returns:
            Frequency recommendation
        """
        # Critical frequency (Hz)
        f_critical = local_omega_pe / (2 * math.pi)

        # Target frequency: 20% above critical for margin
        f_target = f_critical * 1.2

        # Clamp to available range
        f_target = max(self.min_frequency, min(f_target, self.max_frequency))

        # Check if above cutoff
        above_cutoff = f_target > f_critical

        # Compute margin
        margin_dB = (
            20 * math.log10(f_target / max(f_critical, 1e6)) if above_cutoff else -100
        )

        reason = (
            f"Plasma cutoff at {f_critical/1e9:.2f} GHz, "
            f"recommending {f_target/1e9:.2f} GHz"
        )

        if not above_cutoff:
            reason += " (WARNING: Cannot exceed cutoff with available frequencies)"

        return FrequencyRecommendation(
            frequency_Hz=f_target,
            margin_dB=margin_dB,
            above_cutoff=above_cutoff,
            reason=reason,
        )

    def recommend_maneuver(
        self, blackout_map: BlackoutMap, target_antenna_id: str | None = None
    ) -> ManeuverRecommendation:
        """
        Recommend attitude maneuver to improve communications.

        If all antennas are in blackout, recommend maneuver to expose
        the antenna with lowest attenuation to the relay.

        Args:
            blackout_map: Current blackout conditions
            target_antenna_id: Specific antenna to optimize (or best)

        Returns:
            Maneuver recommendation
        """
        if target_antenna_id is None:
            target_antenna_id = blackout_map.best_antenna_id

        target_antenna = self.antenna_array.get(target_antenna_id)
        if target_antenna is None:
            return ManeuverRecommendation(
                target_antenna_id=target_antenna_id,
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_deg=0.0,
                expected_improvement_dB=0.0,
                reason="Unknown antenna ID",
            )

        # Get current antenna
        current_idx = blackout_map.antenna_ids.index(self.current_antenna_id)
        current_atten = blackout_map.attenuation_dB[current_idx].item()

        target_idx = blackout_map.antenna_ids.index(target_antenna_id)
        target_atten = blackout_map.attenuation_dB[target_idx].item()

        # Simple maneuver: rotate to point target antenna normal toward relay
        # This is a placeholder - real implementation needs relay geometry
        normal = target_antenna.normal

        # Estimate roll/pitch/yaw to expose antenna (simplified)
        roll_deg = math.degrees(math.atan2(normal[1], normal[2]))
        pitch_deg = math.degrees(
            math.atan2(-normal[0], math.sqrt(normal[1] ** 2 + normal[2] ** 2))
        )
        yaw_deg = 0.0  # Would need relay azimuth

        improvement = current_atten - target_atten

        reason = (
            f"Maneuver to expose {target_antenna_id} antenna "
            f"(expected {improvement:.1f} dB improvement)"
        )

        if blackout_map.all_blackout:
            reason = "ALL ANTENNAS IN BLACKOUT. " + reason

        return ManeuverRecommendation(
            target_antenna_id=target_antenna_id,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            expected_improvement_dB=improvement,
            reason=reason,
        )

    def update(self, blackout_map: BlackoutMap) -> dict:
        """
        Full update cycle: select antenna and recommend frequency.

        Args:
            blackout_map: Current blackout conditions

        Returns:
            Dictionary with all recommendations
        """
        # Select antenna
        antenna_id = self.select_antenna(blackout_map)

        # Get local plasma frequency at selected antenna
        antenna_idx = blackout_map.antenna_ids.index(antenna_id)
        local_omega_pe = blackout_map.plasma_frequency[antenna_idx].item()

        # Recommend frequency
        freq_rec = self.frequency_hop(local_omega_pe)
        self.current_frequency = freq_rec.frequency_Hz

        # Recommend maneuver if in blackout
        maneuver_rec = None
        if blackout_map.is_blackout[antenna_idx]:
            maneuver_rec = self.recommend_maneuver(blackout_map)

        return {
            "selected_antenna": antenna_id,
            "frequency_Hz": freq_rec.frequency_Hz,
            "above_cutoff": freq_rec.above_cutoff,
            "attenuation_dB": blackout_map.attenuation_dB[antenna_idx].item(),
            "is_blackout": blackout_map.is_blackout[antenna_idx].item(),
            "maneuver_recommendation": maneuver_rec,
        }


# =============================================================================
# Utility Functions
# =============================================================================


def estimate_blackout_duration(
    trajectory_time: Tensor, blackout_flags: Tensor
) -> tuple[float, float, float]:
    """
    Estimate blackout duration from trajectory data.

    Args:
        trajectory_time: Time points (s)
        blackout_flags: Boolean flags for blackout at each time

    Returns:
        (start_time, end_time, duration) in seconds
    """
    blackout_flags = torch.as_tensor(blackout_flags, dtype=torch.bool)
    trajectory_time = torch.as_tensor(trajectory_time, dtype=torch.float64)

    if not blackout_flags.any():
        return (0.0, 0.0, 0.0)

    # Find first and last blackout indices
    indices = torch.where(blackout_flags)[0]
    start_idx = indices[0].item()
    end_idx = indices[-1].item()

    start_time = trajectory_time[start_idx].item()
    end_time = trajectory_time[end_idx].item()
    duration = end_time - start_time

    return (start_time, end_time, duration)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Types
    "AntennaType",
    "Antenna",
    "AntennaArray",
    "BlackoutMap",
    "FrequencyRecommendation",
    "ManeuverRecommendation",
    # Core functions
    "compute_blackout_map",
    "compute_blackout_map_from_cfd",
    # Controller
    "CognitiveComms",
    # Utilities
    "estimate_blackout_duration",
]
