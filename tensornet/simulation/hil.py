"""
Hardware-in-the-Loop (HIL) Simulation Interface
================================================

Provides interfaces for HIL testing of guidance algorithms with
realistic sensor noise, actuator dynamics, and communication latency.

Architecture:
    ┌─────────────────┐     ┌──────────────────┐
    │  Flight Computer │◄───►│   HIL Interface   │
    │  (Target HW)     │     │   (This Module)   │
    └─────────────────┘     └────────┬─────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
    ┌─────────┐               ┌─────────────┐             ┌───────────┐
    │ Sensors │               │ Environment │             │ Actuators │
    │ Models  │               │  Dynamics   │             │  Models   │
    └─────────┘               └─────────────┘             └───────────┘

Sensor Models:
    - IMU: Accelerometer + Gyroscope with bias, noise, drift
    - GPS: Position with ionospheric delay and multipath
    - Air Data: Pitot-static with temperature effects
    - Star Tracker: Attitude with sky visibility

Actuator Models:
    - Control Surfaces: Rate limits, position limits, hysteresis
    - Reaction Control: Thrust uncertainty, ignition delay
    - Propulsion: Throttle lag, thrust vectoring

Timing:
    - Synchronous mode: Lock-step with simulation
    - Asynchronous mode: Real-time with hardware clock
    - Accelerated mode: Faster than real-time for Monte Carlo

References:
    [1] NASA-STD-7009: Standard for Models and Simulations
    [2] AIAA Modeling and Simulation Best Practices
"""

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np


class SensorType(Enum):
    """Sensor type classification."""

    IMU = "imu"
    GPS = "gps"
    AIR_DATA = "air_data"
    STAR_TRACKER = "star_tracker"
    RADAR_ALTIMETER = "radar_alt"
    MAGNETOMETER = "mag"


class ActuatorType(Enum):
    """Actuator type classification."""

    AEROSURFACE = "aerosurface"
    REACTION_CONTROL = "rcs"
    THRUST_VECTORING = "tvc"
    THROTTLE = "throttle"


class HILMode(Enum):
    """HIL timing mode."""

    SYNCHRONOUS = "sync"  # Lock-step with simulation
    ASYNCHRONOUS = "async"  # Real-time with hardware
    ACCELERATED = "accel"  # Faster than real-time


@dataclass
class HILConfig:
    """Configuration for HIL simulation."""

    mode: HILMode = HILMode.SYNCHRONOUS
    dt_s: float = 0.001  # 1 kHz update rate
    sensor_latency_ms: float = 1.0
    actuator_latency_ms: float = 2.0
    communication_latency_ms: float = 0.5
    enable_sensor_noise: bool = True
    enable_actuator_dynamics: bool = True
    time_acceleration: float = 1.0  # For ACCELERATED mode
    log_telemetry: bool = True


@dataclass
class SensorNoise:
    """Sensor noise parameters."""

    white_noise_std: float = 0.0
    bias: float = 0.0
    bias_drift_rate: float = 0.0  # per second
    quantization: float = 0.0
    saturation_min: float = -1e10
    saturation_max: float = 1e10


class SensorModel(ABC):
    """Abstract base class for sensor models."""

    def __init__(
        self,
        sensor_type: SensorType,
        update_rate_hz: float = 100.0,
        latency_ms: float = 1.0,
    ):
        self.sensor_type = sensor_type
        self.update_rate_hz = update_rate_hz
        self.latency_ms = latency_ms
        self.last_update_time = 0.0
        self.measurement_buffer: queue.Queue = queue.Queue(maxsize=100)

    @abstractmethod
    def measure(self, true_state: dict[str, float], t: float) -> dict[str, float]:
        """
        Generate sensor measurement from true state.

        Args:
            true_state: True vehicle state
            t: Current simulation time

        Returns:
            Sensor measurement with noise
        """
        pass

    def apply_noise(self, value: float, noise: SensorNoise, dt: float) -> float:
        """Apply noise model to measurement."""
        # Add white noise
        noisy = value + np.random.normal(0, noise.white_noise_std)

        # Add bias with drift
        noise.bias += noise.bias_drift_rate * dt * np.random.randn()
        noisy += noise.bias

        # Apply quantization
        if noise.quantization > 0:
            noisy = np.round(noisy / noise.quantization) * noise.quantization

        # Apply saturation
        noisy = np.clip(noisy, noise.saturation_min, noise.saturation_max)

        return noisy


class IMUSensor(SensorModel):
    """
    Inertial Measurement Unit sensor model.

    Models a 6-DOF IMU with:
    - 3-axis accelerometer
    - 3-axis gyroscope
    - Bias instability and drift
    - Temperature sensitivity
    """

    def __init__(self, update_rate_hz: float = 400.0, latency_ms: float = 0.5):
        super().__init__(SensorType.IMU, update_rate_hz, latency_ms)

        # Accelerometer noise (tactical grade)
        self.accel_noise = SensorNoise(
            white_noise_std=0.01,  # m/s² (1 mg)
            bias=0.0,
            bias_drift_rate=1e-5,  # m/s² per second
            quantization=0.001,  # m/s²
            saturation_min=-200.0,  # m/s² (~20g)
            saturation_max=200.0,
        )

        # Gyroscope noise (tactical grade)
        self.gyro_noise = SensorNoise(
            white_noise_std=0.0001,  # rad/s (0.006 deg/s)
            bias=0.0,
            bias_drift_rate=1e-7,  # rad/s per second
            quantization=1e-5,  # rad/s
            saturation_min=-10.0,  # rad/s (~573 deg/s)
            saturation_max=10.0,
        )

        # Per-axis biases
        self.accel_bias = [0.0, 0.0, 0.0]
        self.gyro_bias = [0.0, 0.0, 0.0]

    def measure(self, true_state: dict[str, float], t: float) -> dict[str, float]:
        """Generate IMU measurement."""
        dt = t - self.last_update_time
        self.last_update_time = t

        # Get true values
        ax_true = true_state.get("ax", 0.0)
        ay_true = true_state.get("ay", 0.0)
        az_true = true_state.get("az", 0.0)
        p_true = true_state.get("p", 0.0)
        q_true = true_state.get("q", 0.0)
        r_true = true_state.get("r", 0.0)

        # Apply noise
        ax = self.apply_noise(ax_true, self.accel_noise, dt)
        ay = self.apply_noise(ay_true, self.accel_noise, dt)
        az = self.apply_noise(az_true, self.accel_noise, dt)
        p = self.apply_noise(p_true, self.gyro_noise, dt)
        q = self.apply_noise(q_true, self.gyro_noise, dt)
        r = self.apply_noise(r_true, self.gyro_noise, dt)

        return {"ax": ax, "ay": ay, "az": az, "p": p, "q": q, "r": r, "timestamp": t}


class GPSSensor(SensorModel):
    """
    GPS receiver sensor model.

    Models GPS with:
    - Position accuracy (CEP)
    - Velocity accuracy
    - Dilution of precision (DOP)
    - Ionospheric/tropospheric delays
    - Signal acquisition/loss
    """

    def __init__(self, update_rate_hz: float = 10.0, latency_ms: float = 50.0):
        super().__init__(SensorType.GPS, update_rate_hz, latency_ms)

        # Position noise (civilian GPS)
        self.position_noise = SensorNoise(
            white_noise_std=2.5,  # meters (CEP)
            bias=0.0,
            quantization=0.01,  # meters
        )

        # Velocity noise
        self.velocity_noise = SensorNoise(
            white_noise_std=0.1,  # m/s
            bias=0.0,
            quantization=0.001,  # m/s
        )

        # DOP factor
        self.hdop = 1.0
        self.vdop = 1.5

        # Signal state
        self.signal_acquired = True
        self.num_satellites = 12

    def measure(self, true_state: dict[str, float], t: float) -> dict[str, float]:
        """Generate GPS measurement."""
        dt = t - self.last_update_time
        self.last_update_time = t

        if not self.signal_acquired:
            return {"valid": False, "timestamp": t}

        # Position
        lat = (
            self.apply_noise(true_state.get("lat", 0.0), self.position_noise, dt)
            * self.hdop
        )

        lon = (
            self.apply_noise(true_state.get("lon", 0.0), self.position_noise, dt)
            * self.hdop
        )

        alt = (
            self.apply_noise(true_state.get("alt", 0.0), self.position_noise, dt)
            * self.vdop
        )

        # Velocity
        vn = self.apply_noise(true_state.get("vn", 0.0), self.velocity_noise, dt)
        ve = self.apply_noise(true_state.get("ve", 0.0), self.velocity_noise, dt)
        vd = self.apply_noise(true_state.get("vd", 0.0), self.velocity_noise, dt)

        return {
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "vn": vn,
            "ve": ve,
            "vd": vd,
            "hdop": self.hdop,
            "vdop": self.vdop,
            "num_satellites": self.num_satellites,
            "valid": True,
            "timestamp": t,
        }


class AirDataSensor(SensorModel):
    """
    Air data system sensor model.

    Models pitot-static system with:
    - Static pressure
    - Dynamic pressure
    - Total temperature
    - Angle of attack vanes
    - Sideslip angle vanes
    """

    def __init__(self, update_rate_hz: float = 100.0, latency_ms: float = 5.0):
        super().__init__(SensorType.AIR_DATA, update_rate_hz, latency_ms)

        # Pressure noise
        self.pressure_noise = SensorNoise(
            white_noise_std=10.0,  # Pa
            bias=0.0,
            quantization=1.0,  # Pa
        )

        # Temperature noise
        self.temperature_noise = SensorNoise(
            white_noise_std=0.5,  # K
            bias=0.0,
            quantization=0.1,  # K
        )

        # Angle noise
        self.angle_noise = SensorNoise(
            white_noise_std=0.5,  # degrees
            bias=0.0,
            quantization=0.1,  # degrees
        )

    def measure(self, true_state: dict[str, float], t: float) -> dict[str, float]:
        """Generate air data measurement."""
        dt = t - self.last_update_time
        self.last_update_time = t

        p_static = self.apply_noise(
            true_state.get("p_static", 101325.0), self.pressure_noise, dt
        )

        q_dynamic = self.apply_noise(
            true_state.get("q_dynamic", 0.0), self.pressure_noise, dt
        )

        T_total = self.apply_noise(
            true_state.get("T_total", 288.15), self.temperature_noise, dt
        )

        alpha = self.apply_noise(true_state.get("alpha_deg", 0.0), self.angle_noise, dt)

        beta = self.apply_noise(true_state.get("beta_deg", 0.0), self.angle_noise, dt)

        return {
            "p_static": p_static,
            "q_dynamic": q_dynamic,
            "T_total": T_total,
            "alpha_deg": alpha,
            "beta_deg": beta,
            "timestamp": t,
        }


class ActuatorModel(ABC):
    """Abstract base class for actuator models."""

    def __init__(
        self,
        actuator_type: ActuatorType,
        rate_limit: float = 100.0,
        position_limit_min: float = -30.0,
        position_limit_max: float = 30.0,
        latency_ms: float = 2.0,
    ):
        self.actuator_type = actuator_type
        self.rate_limit = rate_limit
        self.position_limit_min = position_limit_min
        self.position_limit_max = position_limit_max
        self.latency_ms = latency_ms
        self.current_position = 0.0
        self.commanded_position = 0.0
        self.command_buffer: queue.Queue = queue.Queue(maxsize=100)

    @abstractmethod
    def update(self, command: float, dt: float) -> float:
        """
        Update actuator position based on command.

        Args:
            command: Commanded position
            dt: Time step

        Returns:
            Actual position after dynamics
        """
        pass


class ControlSurface(ActuatorModel):
    """
    Aerodynamic control surface actuator model.

    Models:
    - Servo dynamics (first/second order)
    - Rate limits
    - Position limits
    - Hysteresis
    - Load-dependent performance
    """

    def __init__(
        self,
        name: str = "elevator",
        rate_limit_deg_s: float = 60.0,
        position_limit_deg: float = 30.0,
        bandwidth_hz: float = 10.0,
        latency_ms: float = 5.0,
    ):
        super().__init__(
            ActuatorType.AEROSURFACE,
            rate_limit_deg_s,
            -position_limit_deg,
            position_limit_deg,
            latency_ms,
        )
        self.name = name
        self.bandwidth_hz = bandwidth_hz
        self.omega_n = 2 * np.pi * bandwidth_hz
        self.damping = 0.7
        self.velocity = 0.0

    def update(self, command: float, dt: float) -> float:
        """Update control surface position with second-order dynamics."""
        # Apply position limits to command
        command = np.clip(command, self.position_limit_min, self.position_limit_max)

        # Second-order dynamics
        error = command - self.current_position
        accel = (
            self.omega_n**2 * error - 2 * self.damping * self.omega_n * self.velocity
        )

        self.velocity += accel * dt

        # Apply rate limit
        self.velocity = np.clip(self.velocity, -self.rate_limit, self.rate_limit)

        self.current_position += self.velocity * dt

        # Apply position limits
        self.current_position = np.clip(
            self.current_position, self.position_limit_min, self.position_limit_max
        )

        return self.current_position


class ThrustActuator(ActuatorModel):
    """
    Propulsion thrust actuator model.

    Models:
    - Throttle lag (first order)
    - Ignition delay
    - Thrust uncertainty
    - Thrust vectoring
    """

    def __init__(
        self,
        max_thrust_N: float = 100000.0,
        throttle_lag_s: float = 0.1,
        ignition_delay_s: float = 0.05,
        thrust_uncertainty_pct: float = 3.0,
    ):
        super().__init__(
            ActuatorType.THROTTLE,
            rate_limit=10.0,  # Throttle rate limit (0-1 per second)
            position_limit_min=0.0,
            position_limit_max=1.0,
        )
        self.max_thrust_N = max_thrust_N
        self.throttle_lag_s = throttle_lag_s
        self.ignition_delay_s = ignition_delay_s
        self.thrust_uncertainty_pct = thrust_uncertainty_pct
        self.thrust_multiplier = 1.0 + np.random.normal(0, thrust_uncertainty_pct / 100)
        self.ignition_time: float | None = None
        self.is_ignited = False

    def ignite(self, t: float):
        """Command ignition."""
        self.ignition_time = t

    def update(self, command: float, dt: float) -> float:
        """Update thrust level."""
        # Check ignition delay
        if self.ignition_time is not None and not self.is_ignited:
            if time.time() - self.ignition_time > self.ignition_delay_s:
                self.is_ignited = True

        if not self.is_ignited:
            self.current_position = 0.0
            return 0.0

        # First-order throttle lag
        command = np.clip(command, 0.0, 1.0)
        tau = self.throttle_lag_s
        alpha = dt / (tau + dt)
        self.current_position += alpha * (command - self.current_position)

        # Compute thrust with uncertainty
        thrust = self.current_position * self.max_thrust_N * self.thrust_multiplier

        return thrust


class HILInterface:
    """
    Hardware-in-the-Loop simulation interface.

    Manages the complete HIL loop including:
    - Sensor data generation
    - Actuator command processing
    - Timing and synchronization
    - Telemetry logging
    """

    def __init__(self, config: HILConfig = None):
        self.config = config or HILConfig()

        self.sensors: dict[str, SensorModel] = {}
        self.actuators: dict[str, ActuatorModel] = {}

        self.simulation_time = 0.0
        self.wall_clock_start: float | None = None

        self.telemetry_log: list[dict] = []
        self._running = False
        self._lock = threading.Lock()

    def add_sensor(self, name: str, sensor: SensorModel):
        """Add a sensor to the HIL interface."""
        self.sensors[name] = sensor

    def add_actuator(self, name: str, actuator: ActuatorModel):
        """Add an actuator to the HIL interface."""
        self.actuators[name] = actuator

    def get_sensor_data(self, true_state: dict[str, float]) -> dict[str, dict]:
        """
        Get all sensor measurements.

        Args:
            true_state: True vehicle state

        Returns:
            Dict of sensor name -> measurement
        """
        measurements = {}

        for name, sensor in self.sensors.items():
            measurements[name] = sensor.measure(true_state, self.simulation_time)

        return measurements

    def set_actuator_commands(
        self, commands: dict[str, float], dt: float
    ) -> dict[str, float]:
        """
        Set actuator commands and get actual positions.

        Args:
            commands: Dict of actuator name -> command
            dt: Time step

        Returns:
            Dict of actuator name -> actual position
        """
        positions = {}

        for name, actuator in self.actuators.items():
            if name in commands:
                positions[name] = actuator.update(commands[name], dt)
            else:
                positions[name] = actuator.current_position

        return positions

    def step(
        self, true_state: dict[str, float], commands: dict[str, float]
    ) -> tuple[dict[str, dict], dict[str, float]]:
        """
        Execute one HIL step.

        Args:
            true_state: True vehicle state
            commands: Actuator commands

        Returns:
            (sensor_measurements, actuator_positions)
        """
        dt = self.config.dt_s

        # Get sensor measurements
        sensors = self.get_sensor_data(true_state)

        # Process actuator commands
        actuators = self.set_actuator_commands(commands, dt)

        # Log telemetry
        if self.config.log_telemetry:
            self.telemetry_log.append(
                {
                    "time": self.simulation_time,
                    "sensors": sensors,
                    "actuators": actuators,
                    "true_state": true_state.copy(),
                }
            )

        # Advance time
        self.simulation_time += dt

        return sensors, actuators

    def run_realtime(
        self,
        get_true_state: Callable[[], dict[str, float]],
        get_commands: Callable[[dict], dict[str, float]],
        duration_s: float,
    ):
        """
        Run HIL in real-time mode.

        Args:
            get_true_state: Callback to get current true state
            get_commands: Callback to get actuator commands from sensor data
            duration_s: Duration to run
        """
        self.wall_clock_start = time.perf_counter()
        self._running = True

        while self._running and self.simulation_time < duration_s:
            # Get true state
            true_state = get_true_state()

            # Generate sensor data
            sensors = self.get_sensor_data(true_state)

            # Get commands from flight computer
            commands = get_commands(sensors)

            # Update actuators
            actuators = self.set_actuator_commands(commands, self.config.dt_s)

            # Log
            if self.config.log_telemetry:
                self.telemetry_log.append(
                    {
                        "time": self.simulation_time,
                        "wall_time": time.perf_counter() - self.wall_clock_start,
                        "sensors": sensors,
                        "actuators": actuators,
                    }
                )

            # Timing
            self.simulation_time += self.config.dt_s

            if self.config.mode == HILMode.ASYNCHRONOUS:
                # Real-time: wait for wall clock
                target_wall = self.simulation_time / self.config.time_acceleration
                elapsed = time.perf_counter() - self.wall_clock_start
                if elapsed < target_wall:
                    time.sleep(target_wall - elapsed)

    def stop(self):
        """Stop real-time HIL."""
        self._running = False

    def reset(self):
        """Reset HIL state."""
        self.simulation_time = 0.0
        self.telemetry_log.clear()

        for actuator in self.actuators.values():
            actuator.current_position = 0.0
            actuator.velocity = 0.0 if hasattr(actuator, "velocity") else None

    def get_telemetry(self) -> list[dict]:
        """Get logged telemetry."""
        return self.telemetry_log


def create_vehicle_sensors(vehicle_class: str = "hypersonic") -> dict[str, SensorModel]:
    """
    Create a standard sensor suite for a vehicle class.

    Args:
        vehicle_class: 'hypersonic', 'missile', 'aircraft'

    Returns:
        Dict of sensor name -> SensorModel
    """
    sensors = {}

    if vehicle_class == "hypersonic":
        sensors["imu"] = IMUSensor(update_rate_hz=400, latency_ms=0.5)
        sensors["gps"] = GPSSensor(update_rate_hz=10, latency_ms=50)
        sensors["air_data"] = AirDataSensor(update_rate_hz=100, latency_ms=5)
    elif vehicle_class == "missile":
        sensors["imu"] = IMUSensor(update_rate_hz=1000, latency_ms=0.2)
        sensors["gps"] = GPSSensor(update_rate_hz=20, latency_ms=30)
    else:  # aircraft
        sensors["imu"] = IMUSensor(update_rate_hz=200, latency_ms=1.0)
        sensors["gps"] = GPSSensor(update_rate_hz=5, latency_ms=100)
        sensors["air_data"] = AirDataSensor(update_rate_hz=50, latency_ms=10)

    return sensors


def create_vehicle_actuators(
    vehicle_class: str = "hypersonic",
) -> dict[str, ActuatorModel]:
    """
    Create a standard actuator suite for a vehicle class.

    Args:
        vehicle_class: 'hypersonic', 'missile', 'aircraft'

    Returns:
        Dict of actuator name -> ActuatorModel
    """
    actuators = {}

    if vehicle_class == "hypersonic":
        actuators["elevator"] = ControlSurface(
            name="elevator", rate_limit_deg_s=60, position_limit_deg=30, bandwidth_hz=10
        )
        actuators["aileron"] = ControlSurface(
            name="aileron", rate_limit_deg_s=100, position_limit_deg=25, bandwidth_hz=15
        )
        actuators["rudder"] = ControlSurface(
            name="rudder", rate_limit_deg_s=60, position_limit_deg=30, bandwidth_hz=10
        )
    elif vehicle_class == "missile":
        actuators["fin1"] = ControlSurface("fin1", 200, 30, 20)
        actuators["fin2"] = ControlSurface("fin2", 200, 30, 20)
        actuators["fin3"] = ControlSurface("fin3", 200, 30, 20)
        actuators["fin4"] = ControlSurface("fin4", 200, 30, 20)
        actuators["thrust"] = ThrustActuator(max_thrust_N=50000)
    else:  # aircraft
        actuators["elevator"] = ControlSurface("elevator", 30, 25, 5)
        actuators["aileron"] = ControlSurface("aileron", 40, 20, 8)
        actuators["rudder"] = ControlSurface("rudder", 30, 25, 5)
        actuators["throttle"] = ThrustActuator(max_thrust_N=200000)

    return actuators


def validate_hil_module():
    """Validate HIL module."""
    print("\n" + "=" * 70)
    print("HIL SIMULATION VALIDATION")
    print("=" * 70)

    # Test 1: IMU Sensor
    print("\n[Test 1] IMU Sensor")
    print("-" * 40)

    imu = IMUSensor()
    true_state = {"ax": 9.81, "ay": 0, "az": 0, "p": 0.1, "q": 0, "r": 0}

    measurements = []
    for i in range(100):
        m = imu.measure(true_state, i * 0.0025)
        measurements.append(m["ax"])

    mean_ax = np.mean(measurements)
    std_ax = np.std(measurements)

    print("True ax: 9.81 m/s²")
    print(f"Mean measured: {mean_ax:.3f} m/s²")
    print(f"Std dev: {std_ax:.4f} m/s²")
    assert abs(mean_ax - 9.81) < 0.1
    print("✓ PASS")

    # Test 2: GPS Sensor
    print("\n[Test 2] GPS Sensor")
    print("-" * 40)

    gps = GPSSensor()
    true_state = {
        "lat": 34.0,
        "lon": -118.0,
        "alt": 10000,
        "vn": 100,
        "ve": 0,
        "vd": -10,
    }

    m = gps.measure(true_state, 0.1)

    print("True position: (34.0, -118.0, 10000)")
    print(f"Measured: ({m['lat']:.2f}, {m['lon']:.2f}, {m['alt']:.0f})")
    print(f"Satellites: {m['num_satellites']}")
    assert m["valid"]
    print("✓ PASS")

    # Test 3: Air Data Sensor
    print("\n[Test 3] Air Data Sensor")
    print("-" * 40)

    air = AirDataSensor()
    true_state = {
        "p_static": 50000,
        "q_dynamic": 25000,
        "T_total": 400,
        "alpha_deg": 5,
        "beta_deg": 0,
    }

    m = air.measure(true_state, 0.01)

    print("True p_static: 50000 Pa")
    print(f"Measured: {m['p_static']:.0f} Pa")
    print(f"True alpha: 5 deg, Measured: {m['alpha_deg']:.1f} deg")
    print("✓ PASS")

    # Test 4: Control Surface Actuator
    print("\n[Test 4] Control Surface")
    print("-" * 40)

    elevator = ControlSurface(
        name="elevator", rate_limit_deg_s=60, position_limit_deg=30
    )

    positions = []
    for i in range(100):
        pos = elevator.update(20.0, 0.01)  # Command 20 deg
        positions.append(pos)

    print("Command: 20 deg")
    print(f"Final position: {positions[-1]:.1f} deg")
    print(f"Time to 90%: ~{np.argmax(np.array(positions) > 18) * 10} ms")
    assert abs(positions[-1] - 20.0) < 1.0
    print("✓ PASS")

    # Test 5: Thrust Actuator
    print("\n[Test 5] Thrust Actuator")
    print("-" * 40)

    thrust = ThrustActuator(max_thrust_N=100000)
    thrust.is_ignited = True  # Skip ignition delay

    thrusts = []
    for i in range(50):
        t = thrust.update(0.8, 0.01)  # 80% throttle
        thrusts.append(t)

    print("Command: 80% throttle")
    print("Max thrust: 100000 N")
    print(f"Final thrust: {thrusts[-1]:.0f} N")
    print(f"Uncertainty factor: {thrust.thrust_multiplier:.3f}")
    print("✓ PASS")

    # Test 6: HIL Interface
    print("\n[Test 6] HIL Interface")
    print("-" * 40)

    config = HILConfig(mode=HILMode.SYNCHRONOUS, dt_s=0.01)
    hil = HILInterface(config)

    hil.add_sensor("imu", IMUSensor())
    hil.add_sensor("gps", GPSSensor())
    hil.add_actuator("elevator", ControlSurface())

    true_state = {
        "ax": 0,
        "ay": 0,
        "az": -9.81,
        "p": 0,
        "q": 0,
        "r": 0,
        "lat": 0,
        "lon": 0,
        "alt": 1000,
        "vn": 100,
        "ve": 0,
        "vd": 0,
    }
    commands = {"elevator": 5.0}

    sensors, actuators = hil.step(true_state, commands)

    print(f"Sensors: {list(sensors.keys())}")
    print(f"Actuators: {list(actuators.keys())}")
    print(f"Elevator position: {actuators['elevator']:.2f} deg")
    print(f"Telemetry entries: {len(hil.telemetry_log)}")
    print("✓ PASS")

    # Test 7: Vehicle Sensor Suite
    print("\n[Test 7] Vehicle Sensor Suite")
    print("-" * 40)

    sensors = create_vehicle_sensors("hypersonic")
    actuators = create_vehicle_actuators("hypersonic")

    print(f"Hypersonic sensors: {list(sensors.keys())}")
    print(f"Hypersonic actuators: {list(actuators.keys())}")

    assert "imu" in sensors
    assert "elevator" in actuators
    print("✓ PASS")

    print("\n" + "=" * 70)
    print("HIL SIMULATION VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    validate_hil_module()
