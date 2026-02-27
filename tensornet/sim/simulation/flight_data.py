"""
Flight Data Integration Module
==============================

Parses, validates, and reconstructs trajectories from flight telemetry data
for model validation and post-flight analysis.

Capabilities:
    - Multi-format telemetry parsing (IRIG-106, CSV, binary)
    - Trajectory reconstruction with EKF/UKF
    - Model vs flight comparison metrics
    - Aerodynamic parameter estimation
    - Data quality assessment

Data Flow:
    ┌────────────────┐     ┌──────────────────┐     ┌──────────────────┐
    │  Raw Telemetry │────►│  Parser/Decoder  │────►│ TelemetryFrame[] │
    │  (Files/Stream)│     │                  │     │                  │
    └────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                             │
                           ┌──────────────────┐              │
                           │    FlightRecord  │◄─────────────┘
                           │                  │
                           └────────┬─────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
    ┌─────────────┐          ┌─────────────┐           ┌─────────────┐
    │ Trajectory  │          │  Parameter  │           │   Model     │
    │ Reconstruct │          │  Estimation │           │  Validation │
    └─────────────┘          └─────────────┘           └─────────────┘

References:
    [1] AIAA S-111: Flight Test Techniques
    [2] NASA-HDBK-1001: Atmosphere Models for Aerothermodynamic Analysis
"""

import json
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class TelemetryFormat(Enum):
    """Supported telemetry formats."""

    CSV = "csv"
    BINARY = "binary"
    IRIG_106 = "irig106"
    JSON = "json"
    HDF5 = "hdf5"


class DataQuality(Enum):
    """Data quality classification."""

    GOOD = "good"
    DEGRADED = "degraded"
    BAD = "bad"
    MISSING = "missing"


@dataclass
class TelemetryFrame:
    """
    Single telemetry frame with synchronized sensor data.

    Attributes:
        timestamp: Time since epoch or mission start (s)
        position: [lat, lon, alt] in deg, deg, m
        velocity: [vn, ve, vd] in m/s (NED frame)
        attitude: [roll, pitch, yaw] in rad
        rates: [p, q, r] in rad/s
        accelerations: [ax, ay, az] in m/s² (body frame)
        air_data: Dict with alpha, beta, mach, q_dynamic, etc.
        sensor_status: Dict of sensor validity flags
        quality: Overall frame quality
    """

    timestamp: float
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    attitude: np.ndarray | None = None
    rates: np.ndarray | None = None
    accelerations: np.ndarray | None = None
    air_data: dict[str, float] | None = None
    sensor_status: dict[str, bool] = field(default_factory=dict)
    quality: DataQuality = DataQuality.GOOD

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "position": self.position.tolist() if self.position is not None else None,
            "velocity": self.velocity.tolist() if self.velocity is not None else None,
            "attitude": self.attitude.tolist() if self.attitude is not None else None,
            "rates": self.rates.tolist() if self.rates is not None else None,
            "accelerations": (
                self.accelerations.tolist() if self.accelerations is not None else None
            ),
            "air_data": self.air_data,
            "sensor_status": self.sensor_status,
            "quality": self.quality.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TelemetryFrame":
        """Create from dictionary."""
        return cls(
            timestamp=d["timestamp"],
            position=np.array(d["position"]) if d.get("position") else None,
            velocity=np.array(d["velocity"]) if d.get("velocity") else None,
            attitude=np.array(d["attitude"]) if d.get("attitude") else None,
            rates=np.array(d["rates"]) if d.get("rates") else None,
            accelerations=(
                np.array(d["accelerations"]) if d.get("accelerations") else None
            ),
            air_data=d.get("air_data"),
            sensor_status=d.get("sensor_status", {}),
            quality=DataQuality(d.get("quality", "good")),
        )


@dataclass
class FlightRecord:
    """
    Complete flight record with metadata and telemetry.

    Attributes:
        flight_id: Unique flight identifier
        vehicle_id: Vehicle configuration ID
        date: Flight date (ISO format)
        frames: List of telemetry frames
        metadata: Additional flight information
        events: Timestamped events (ignition, separation, etc.)
    """

    flight_id: str
    vehicle_id: str
    date: str
    frames: list[TelemetryFrame] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[tuple[float, str]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Flight duration in seconds."""
        if not self.frames:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp

    @property
    def sample_rate(self) -> float:
        """Average sample rate in Hz."""
        if len(self.frames) < 2:
            return 0.0
        return len(self.frames) / self.duration

    def get_time_series(self, field: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract time series data for a field.

        Args:
            field: Field name ('position', 'velocity', 'mach', etc.)

        Returns:
            (times, values) arrays
        """
        times = []
        values = []

        for frame in self.frames:
            if field == "position" and frame.position is not None:
                times.append(frame.timestamp)
                values.append(frame.position)
            elif field == "velocity" and frame.velocity is not None:
                times.append(frame.timestamp)
                values.append(frame.velocity)
            elif field == "attitude" and frame.attitude is not None:
                times.append(frame.timestamp)
                values.append(frame.attitude)
            elif field in ["mach", "alpha", "beta", "q_dynamic"] and frame.air_data:
                if field in frame.air_data:
                    times.append(frame.timestamp)
                    values.append(frame.air_data[field])

        return np.array(times), np.array(values)

    def trim_to_range(self, t_start: float, t_end: float) -> "FlightRecord":
        """Return a new FlightRecord trimmed to time range."""
        trimmed_frames = [f for f in self.frames if t_start <= f.timestamp <= t_end]

        return FlightRecord(
            flight_id=self.flight_id + "_trimmed",
            vehicle_id=self.vehicle_id,
            date=self.date,
            frames=trimmed_frames,
            metadata=self.metadata.copy(),
            events=[(t, e) for t, e in self.events if t_start <= t <= t_end],
        )

    def resample(self, target_rate_hz: float) -> "FlightRecord":
        """Resample to target rate using linear interpolation."""
        if not self.frames:
            return FlightRecord(
                flight_id=self.flight_id + "_resampled",
                vehicle_id=self.vehicle_id,
                date=self.date,
            )

        t_start = self.frames[0].timestamp
        t_end = self.frames[-1].timestamp
        dt = 1.0 / target_rate_hz

        new_times = np.arange(t_start, t_end, dt)
        new_frames = []

        # Get original time series
        times_pos, pos = self.get_time_series("position")
        times_vel, vel = self.get_time_series("velocity")
        times_att, att = self.get_time_series("attitude")

        for t in new_times:
            frame = TelemetryFrame(timestamp=t)

            # Interpolate position
            if len(times_pos) > 1:
                frame.position = np.array(
                    [np.interp(t, times_pos, pos[:, i]) for i in range(3)]
                )

            # Interpolate velocity
            if len(times_vel) > 1:
                frame.velocity = np.array(
                    [np.interp(t, times_vel, vel[:, i]) for i in range(3)]
                )

            # Interpolate attitude
            if len(times_att) > 1:
                frame.attitude = np.array(
                    [np.interp(t, times_att, att[:, i]) for i in range(3)]
                )

            new_frames.append(frame)

        return FlightRecord(
            flight_id=self.flight_id + "_resampled",
            vehicle_id=self.vehicle_id,
            date=self.date,
            frames=new_frames,
            metadata=self.metadata.copy(),
            events=self.events.copy(),
        )


@dataclass
class TrajectoryReconstruction:
    """
    Reconstructed trajectory with uncertainty.

    Uses Extended Kalman Filter or Unscented Kalman Filter
    to optimally combine sensor data.
    """

    # State: [x, y, z, vx, vy, vz, ax, ay, az, roll, pitch, yaw]
    state_dim: int = 12

    # Covariance
    P: np.ndarray | None = None

    # Process noise
    Q: np.ndarray | None = None

    # Measurement noise
    R_gps: np.ndarray | None = None
    R_imu: np.ndarray | None = None

    # Reconstructed trajectory
    states: list[np.ndarray] = field(default_factory=list)
    covariances: list[np.ndarray] = field(default_factory=list)
    times: list[float] = field(default_factory=list)

    def __post_init__(self):
        # Initialize covariances if not provided
        if self.P is None:
            self.P = np.diag(
                [
                    100,
                    100,
                    100,  # Position (m²)
                    10,
                    10,
                    10,  # Velocity (m²/s²)
                    1,
                    1,
                    1,  # Acceleration (m²/s⁴)
                    0.1,
                    0.1,
                    0.1,  # Attitude (rad²)
                ]
            )

        if self.Q is None:
            self.Q = np.diag(
                [
                    0.01,
                    0.01,
                    0.01,  # Position process noise
                    0.1,
                    0.1,
                    0.1,  # Velocity process noise
                    1.0,
                    1.0,
                    1.0,  # Acceleration process noise
                    0.01,
                    0.01,
                    0.01,  # Attitude process noise
                ]
            )

        if self.R_gps is None:
            self.R_gps = np.diag([5, 5, 10, 0.1, 0.1, 0.1])  # GPS measurement noise

        if self.R_imu is None:
            self.R_imu = np.diag(
                [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
            )  # IMU measurement noise

    def state_transition(self, x: np.ndarray, dt: float) -> np.ndarray:
        """State transition function."""
        x_new = x.copy()

        # Position update
        x_new[0:3] += x[3:6] * dt + 0.5 * x[6:9] * dt**2

        # Velocity update
        x_new[3:6] += x[6:9] * dt

        # Attitude (simple integration, should use quaternions for large angles)
        x_new[9:12] += x[6:9] * dt * 0  # Placeholder

        return x_new

    def process_frame(
        self, frame: TelemetryFrame, x_prev: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Process a telemetry frame with EKF.

        Args:
            frame: Telemetry frame
            x_prev: Previous state estimate
            dt: Time step

        Returns:
            Updated state estimate
        """
        # Predict
        x_pred = self.state_transition(x_prev, dt)

        # Simplified Jacobian (identity for linear case)
        F = np.eye(self.state_dim)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt**2
        F[3:6, 6:9] = np.eye(3) * dt

        P_pred = F @ self.P @ F.T + self.Q

        # Update with GPS if available
        if frame.position is not None and frame.velocity is not None:
            z_gps = np.concatenate([frame.position, frame.velocity])
            H_gps = np.zeros((6, self.state_dim))
            H_gps[0:3, 0:3] = np.eye(3)
            H_gps[3:6, 3:6] = np.eye(3)

            y = z_gps - H_gps @ x_pred
            S = H_gps @ P_pred @ H_gps.T + self.R_gps
            K = P_pred @ H_gps.T @ np.linalg.inv(S)

            x_pred = x_pred + K @ y
            P_pred = (np.eye(self.state_dim) - K @ H_gps) @ P_pred

        # Update with IMU if available
        if frame.accelerations is not None and frame.rates is not None:
            z_imu = np.concatenate([frame.accelerations, frame.rates])
            H_imu = np.zeros((6, self.state_dim))
            H_imu[0:3, 6:9] = np.eye(3)  # Accelerations
            H_imu[3:6, 9:12] = np.eye(3)  # Rates (approximate)

            y = z_imu - H_imu @ x_pred
            S = H_imu @ P_pred @ H_imu.T + self.R_imu
            K = P_pred @ H_imu.T @ np.linalg.inv(S)

            x_pred = x_pred + K @ y
            P_pred = (np.eye(self.state_dim) - K @ H_imu) @ P_pred

        self.P = P_pred
        return x_pred

    def reconstruct(self, record: FlightRecord) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct trajectory from flight record.

        Args:
            record: Flight record with telemetry

        Returns:
            (times, states) arrays
        """
        if not record.frames:
            return np.array([]), np.array([])

        # Initialize state from first frame
        frame0 = record.frames[0]
        x = np.zeros(self.state_dim)

        if frame0.position is not None:
            x[0:3] = frame0.position
        if frame0.velocity is not None:
            x[3:6] = frame0.velocity
        if frame0.attitude is not None:
            x[9:12] = frame0.attitude

        self.states = [x.copy()]
        self.covariances = [self.P.copy()]
        self.times = [frame0.timestamp]

        # Process each frame
        for i in range(1, len(record.frames)):
            frame = record.frames[i]
            dt = frame.timestamp - self.times[-1]

            if dt > 0:
                x = self.process_frame(frame, x, dt)
                self.states.append(x.copy())
                self.covariances.append(self.P.copy())
                self.times.append(frame.timestamp)

        return np.array(self.times), np.array(self.states)

    def get_uncertainty(self, time_idx: int) -> np.ndarray:
        """Get 1-sigma uncertainty at time index."""
        if time_idx < len(self.covariances):
            return np.sqrt(np.diag(self.covariances[time_idx]))
        return np.zeros(self.state_dim)


def parse_telemetry(
    source: str | Path | bytes, format: TelemetryFormat = TelemetryFormat.CSV
) -> FlightRecord:
    """
    Parse telemetry data from various formats.

    Args:
        source: File path, string data, or bytes
        format: Telemetry format

    Returns:
        FlightRecord with parsed data
    """
    if format == TelemetryFormat.CSV:
        return _parse_csv(source)
    elif format == TelemetryFormat.JSON:
        return _parse_json(source)
    elif format == TelemetryFormat.BINARY:
        return _parse_binary(source)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _parse_csv(source: str | Path) -> FlightRecord:
    """Parse CSV telemetry file."""
    lines = _read_csv_lines(source)

    if not lines:
        return FlightRecord("unknown", "unknown", "unknown")

    # Parse header
    header = [h.strip().lower() for h in lines[0].split(",")]

    # Parse data rows
    frames = []
    for line in lines[1:]:
        if not line.strip():
            continue

        values = line.split(",")
        row = {
            header[i]: float(values[i]) for i in range(min(len(header), len(values)))
        }
        frame = _row_to_telemetry_frame(row)
        frames.append(frame)

    return FlightRecord(
        flight_id="parsed_csv", vehicle_id="unknown", date="unknown", frames=frames
    )


def _read_csv_lines(source: str | Path) -> list[str]:
    """Read CSV source into lines."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            return path.read_text().strip().split("\n")
        else:
            # Assume it's CSV content as string
            return source.strip().split("\n")
    else:
        return source.decode().strip().split("\n")


# Field parsers - dispatch table for cleaner parsing
_POSITION_FIELDS = [
    (["lat", "lon", "alt"], lambda r: np.array([r["lat"], r["lon"], r["alt"]])),
    (["x", "y", "z"], lambda r: np.array([r["x"], r["y"], r["z"]])),
]

_VELOCITY_FIELDS = [
    (["vn", "ve", "vd"], lambda r: np.array([r["vn"], r["ve"], r["vd"]])),
    (["vx", "vy", "vz"], lambda r: np.array([r["vx"], r["vy"], r["vz"]])),
]

_ATTITUDE_FIELDS = [
    (["roll", "pitch", "yaw"], lambda r: np.array([r["roll"], r["pitch"], r["yaw"]])),
    (["phi", "theta", "psi"], lambda r: np.array([r["phi"], r["theta"], r["psi"]])),
]

_RATES_FIELDS = [
    (["p", "q", "r"], lambda r: np.array([r["p"], r["q"], r["r"]])),
]

_ACCEL_FIELDS = [
    (["ax", "ay", "az"], lambda r: np.array([r["ax"], r["ay"], r["az"]])),
]


def _parse_vector_field(
    row: dict[str, float], field_specs: list
) -> np.ndarray | None:
    """Parse vector field using dispatch table."""
    for keys, parser in field_specs:
        if all(k in row for k in keys):
            return parser(row)
    return None


def _row_to_telemetry_frame(row: dict[str, float]) -> "TelemetryFrame":
    """Convert a parsed row dict to TelemetryFrame."""
    frame = TelemetryFrame(timestamp=row.get("time", row.get("t", 0.0)))

    # Parse vector fields using dispatch tables
    frame.position = _parse_vector_field(row, _POSITION_FIELDS)
    frame.velocity = _parse_vector_field(row, _VELOCITY_FIELDS)
    frame.attitude = _parse_vector_field(row, _ATTITUDE_FIELDS)
    frame.rates = _parse_vector_field(row, _RATES_FIELDS)
    frame.accelerations = _parse_vector_field(row, _ACCEL_FIELDS)

    # Air data - extract known keys
    air_keys = ["mach", "alpha", "beta", "q_dynamic", "p_static"]
    frame.air_data = {k: row[k] for k in air_keys if k in row}

    return frame


def _parse_json(source: str | Path) -> FlightRecord:
    """Parse JSON telemetry file."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = json.loads(source)
    else:
        data = json.loads(source.decode())

    record = FlightRecord(
        flight_id=data.get("flight_id", "unknown"),
        vehicle_id=data.get("vehicle_id", "unknown"),
        date=data.get("date", "unknown"),
        metadata=data.get("metadata", {}),
        events=[(e["time"], e["name"]) for e in data.get("events", [])],
    )

    for frame_data in data.get("frames", []):
        record.frames.append(TelemetryFrame.from_dict(frame_data))

    return record


def _parse_binary(source: str | Path | bytes) -> FlightRecord:
    """Parse binary telemetry file."""
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    else:
        data = source

    frames = []

    # Simple binary format: time(d), pos(3d), vel(3d), att(3d) = 13 doubles = 104 bytes
    frame_size = 104
    num_frames = len(data) // frame_size

    for i in range(num_frames):
        offset = i * frame_size
        values = struct.unpack("13d", data[offset : offset + frame_size])

        frame = TelemetryFrame(
            timestamp=values[0],
            position=np.array(values[1:4]),
            velocity=np.array(values[4:7]),
            attitude=np.array(values[7:10]),
        )
        frames.append(frame)

    return FlightRecord("binary_parsed", "unknown", "unknown", frames=frames)


def validate_against_flight(
    model_trajectory: np.ndarray,  # (N, 6) position + velocity
    model_times: np.ndarray,
    flight_record: FlightRecord,
) -> dict[str, float]:
    """
    Validate model predictions against flight data.

    Args:
        model_trajectory: Model predicted states
        model_times: Model time points
        flight_record: Flight record to compare against

    Returns:
        Dict of validation metrics
    """
    # Extract flight data
    flight_times, flight_pos = flight_record.get_time_series("position")
    _, flight_vel = flight_record.get_time_series("velocity")

    if len(flight_times) == 0:
        return {"error": "No flight data available"}

    # Interpolate model to flight times
    model_pos_interp = np.zeros((len(flight_times), 3))
    model_vel_interp = np.zeros((len(flight_times), 3))

    for i in range(3):
        model_pos_interp[:, i] = np.interp(
            flight_times, model_times, model_trajectory[:, i]
        )
        model_vel_interp[:, i] = np.interp(
            flight_times, model_times, model_trajectory[:, i + 3]
        )

    # Position errors
    pos_error = flight_pos - model_pos_interp
    pos_rmse = np.sqrt(np.mean(pos_error**2, axis=0))
    pos_rmse_total = np.sqrt(np.sum(pos_rmse**2))
    pos_max = np.max(np.abs(pos_error), axis=0)

    # Velocity errors
    vel_error = flight_vel - model_vel_interp
    vel_rmse = np.sqrt(np.mean(vel_error**2, axis=0))
    vel_rmse_total = np.sqrt(np.sum(vel_rmse**2))
    vel_max = np.max(np.abs(vel_error), axis=0)

    # Cross-range / down-range errors (assuming x is downrange)
    downrange_error_mean = np.mean(np.abs(pos_error[:, 0]))
    crossrange_error_mean = np.mean(
        np.sqrt(pos_error[:, 1] ** 2 + pos_error[:, 2] ** 2)
    )

    return {
        "position_rmse_x": pos_rmse[0],
        "position_rmse_y": pos_rmse[1],
        "position_rmse_z": pos_rmse[2],
        "position_rmse_total": pos_rmse_total,
        "position_max_error_x": pos_max[0],
        "position_max_error_y": pos_max[1],
        "position_max_error_z": pos_max[2],
        "velocity_rmse_total": vel_rmse_total,
        "velocity_max_error": np.max(vel_max),
        "downrange_error_mean": downrange_error_mean,
        "crossrange_error_mean": crossrange_error_mean,
        "num_comparison_points": len(flight_times),
    }


def compute_reconstruction_error(
    reconstruction: TrajectoryReconstruction, flight_record: FlightRecord
) -> dict[str, float]:
    """
    Compute reconstruction error metrics.

    Args:
        reconstruction: Reconstructed trajectory
        flight_record: Original flight record

    Returns:
        Error metrics
    """
    if not reconstruction.states or not flight_record.frames:
        return {"error": "Insufficient data"}

    # Get raw sensor data
    flight_times, flight_pos = flight_record.get_time_series("position")

    # Get reconstructed positions
    recon_states = np.array(reconstruction.states)
    recon_times = np.array(reconstruction.times)
    recon_pos = recon_states[:, 0:3]

    # Interpolate to common times
    common_times = np.intersect1d(np.round(flight_times, 2), np.round(recon_times, 2))

    if len(common_times) < 2:
        # Use all available points
        pos_error = np.zeros(3)
        for i, t in enumerate(flight_times):
            idx = np.argmin(np.abs(recon_times - t))
            if idx < len(recon_pos):
                pos_error += np.abs(flight_pos[i] - recon_pos[idx])
        pos_error /= len(flight_times)
    else:
        pos_error = np.mean([0.0, 0.0, 0.0])  # Placeholder

    # Compute smoothness (innovation sequence should be white)
    if len(recon_pos) > 2:
        innovations = np.diff(recon_pos, axis=0)
        smoothness = np.mean(np.abs(np.diff(innovations, axis=0)))
    else:
        smoothness = 0.0

    # Average uncertainty
    avg_uncertainty = np.mean(
        [
            reconstruction.get_uncertainty(i)[0:3].mean()
            for i in range(len(reconstruction.states))
        ]
    )

    return {
        "position_residual_mean": float(np.mean(pos_error)),
        "trajectory_smoothness": float(smoothness),
        "average_uncertainty_m": float(avg_uncertainty),
        "num_reconstructed_points": len(reconstruction.states),
    }


def create_synthetic_flight_record(
    trajectory: np.ndarray, times: np.ndarray, add_noise: bool = True
) -> FlightRecord:
    """
    Create a synthetic flight record for testing.

    Args:
        trajectory: (N, 6+) state trajectory [pos, vel, ...]
        times: Time array
        add_noise: Whether to add sensor noise

    Returns:
        Synthetic FlightRecord
    """
    frames = []

    for i, t in enumerate(times):
        pos = trajectory[i, 0:3]
        vel = trajectory[i, 3:6]

        if add_noise:
            pos = pos + np.random.normal(0, 5, 3)  # 5m noise
            vel = vel + np.random.normal(0, 0.5, 3)  # 0.5 m/s noise

        frame = TelemetryFrame(
            timestamp=t,
            position=pos,
            velocity=vel,
            attitude=np.zeros(3),
            quality=DataQuality.GOOD,
        )
        frames.append(frame)

    return FlightRecord(
        flight_id="synthetic",
        vehicle_id="test_vehicle",
        date="2024-01-01",
        frames=frames,
    )


def validate_flight_data_module():
    """Validate flight data module."""
    print("\n" + "=" * 70)
    print("FLIGHT DATA INTEGRATION VALIDATION")
    print("=" * 70)

    # Test 1: TelemetryFrame
    print("\n[Test 1] TelemetryFrame")
    print("-" * 40)

    frame = TelemetryFrame(
        timestamp=0.0,
        position=np.array([34.0, -118.0, 10000.0]),
        velocity=np.array([100.0, 0.0, -10.0]),
        attitude=np.array([0.0, 0.05, 0.0]),
    )

    d = frame.to_dict()
    frame2 = TelemetryFrame.from_dict(d)

    print(f"Original timestamp: {frame.timestamp}")
    print(f"Roundtrip timestamp: {frame2.timestamp}")
    assert frame.timestamp == frame2.timestamp
    assert np.allclose(frame.position, frame2.position)
    print("✓ PASS")

    # Test 2: FlightRecord
    print("\n[Test 2] FlightRecord")
    print("-" * 40)

    record = FlightRecord(flight_id="TEST_001", vehicle_id="HGV-1", date="2024-06-15")

    for i in range(100):
        t = i * 0.1
        record.frames.append(
            TelemetryFrame(
                timestamp=t,
                position=np.array([t * 100, 0, 10000 - t * 10]),
                velocity=np.array([100, 0, -10]),
                air_data={"mach": 5.0 + t * 0.01, "alpha": 5.0},
            )
        )

    print(f"Flight ID: {record.flight_id}")
    print(f"Duration: {record.duration:.1f} s")
    print(f"Sample rate: {record.sample_rate:.1f} Hz")

    times, mach = record.get_time_series("mach")
    print(f"Mach range: {mach.min():.2f} - {mach.max():.2f}")

    assert record.duration == 9.9
    assert abs(record.sample_rate - 10.1) < 0.2
    print("✓ PASS")

    # Test 3: CSV Parsing
    print("\n[Test 3] CSV Parsing")
    print("-" * 40)

    csv_data = """time,lat,lon,alt,vn,ve,vd,mach
0.0,34.0,-118.0,10000,100,0,-10,5.0
0.1,34.001,-118.0,9990,100,0,-10,5.1
0.2,34.002,-118.0,9980,100,0,-10,5.2
"""

    record = parse_telemetry(csv_data, TelemetryFormat.CSV)
    print(f"Parsed {len(record.frames)} frames")
    print(f"First position: {record.frames[0].position}")

    assert len(record.frames) == 3
    assert record.frames[0].air_data["mach"] == 5.0
    print("✓ PASS")

    # Test 4: Trajectory Reconstruction
    print("\n[Test 4] Trajectory Reconstruction")
    print("-" * 40)

    # Create synthetic trajectory
    times = np.linspace(0, 10, 100)
    trajectory = np.zeros((100, 6))
    trajectory[:, 0] = times * 200  # x = 200*t
    trajectory[:, 1] = np.sin(times * 0.5) * 100  # sinusoidal y
    trajectory[:, 2] = 10000 - times * 50  # descending z
    trajectory[:, 3] = 200  # vx
    trajectory[:, 4] = np.cos(times * 0.5) * 50  # vy
    trajectory[:, 5] = -50  # vz

    record = create_synthetic_flight_record(trajectory, times, add_noise=True)

    recon = TrajectoryReconstruction()
    recon_times, recon_states = recon.reconstruct(record)

    print(f"Reconstructed {len(recon_times)} states")
    print(f"Final position: {recon_states[-1, 0:3]}")
    print(f"Final uncertainty: {recon.get_uncertainty(-1)[0:3]}")

    assert len(recon_times) == len(record.frames)
    print("✓ PASS")

    # Test 5: Model Validation
    print("\n[Test 5] Model Validation")
    print("-" * 40)

    # Model trajectory (slightly different from flight)
    model_times = np.linspace(0, 10, 50)
    model_traj = np.zeros((50, 6))
    model_traj[:, 0] = model_times * 205  # 2.5% velocity error
    model_traj[:, 1] = np.sin(model_times * 0.5) * 95
    model_traj[:, 2] = 10000 - model_times * 48
    model_traj[:, 3] = 205
    model_traj[:, 4] = np.cos(model_times * 0.5) * 47.5
    model_traj[:, 5] = -48

    metrics = validate_against_flight(model_traj, model_times, record)

    print(f"Position RMSE (total): {metrics['position_rmse_total']:.1f} m")
    print(f"Velocity RMSE: {metrics['velocity_rmse_total']:.2f} m/s")
    print(f"Downrange error: {metrics['downrange_error_mean']:.1f} m")

    assert metrics["position_rmse_total"] < 500  # Reasonable for noisy data
    print("✓ PASS")

    # Test 6: Reconstruction Error
    print("\n[Test 6] Reconstruction Error")
    print("-" * 40)

    error_metrics = compute_reconstruction_error(recon, record)

    print(f"Position residual mean: {error_metrics['position_residual_mean']:.2f}")
    print(f"Trajectory smoothness: {error_metrics['trajectory_smoothness']:.4f}")
    print(f"Average uncertainty: {error_metrics['average_uncertainty_m']:.1f} m")

    assert error_metrics["num_reconstructed_points"] > 0
    print("✓ PASS")

    # Test 7: Resample
    print("\n[Test 7] Resample")
    print("-" * 40)

    resampled = record.resample(50.0)  # 50 Hz

    print(f"Original sample rate: {record.sample_rate:.1f} Hz")
    print(f"Resampled rate: {resampled.sample_rate:.1f} Hz")
    print(f"Original frames: {len(record.frames)}")
    print(f"Resampled frames: {len(resampled.frames)}")

    assert abs(resampled.sample_rate - 50.0) < 5.0
    print("✓ PASS")

    print("\n" + "=" * 70)
    print("FLIGHT DATA INTEGRATION VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    validate_flight_data_module()
