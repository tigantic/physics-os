#!/usr/bin/env python3
"""DOMINION Validation Framework - Pytest Configuration

The Validation Doctrine for Operation: DOMINION
Type I Civilization Engine Forensic Audits

Author: HyperTensor Physics Laboratory
Copyright (c) 2025 TiganticLabz. All Rights Reserved.
"""

import pytest
import time
import json
import mmap
import struct
import subprocess
import psutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DOMINION_EXE = PROJECT_ROOT / "dominion-gui" / "target" / "release" / "dominion"
BRIDGE_SHM = "/dev/shm/hyperfoam_bridge"
COMMAND_PIPE = "/tmp/dominion_cmd.json"

# Thresholds (The Doctrine)
LATENCY_THRESHOLD_MS = 1.0       # Round-trip < 1ms
LATENCY_SPIKE_MS = 5.0           # Any spike > 5ms = FAIL
FRAME_BUDGET_MS = 16.6           # 60 FPS budget
COLD_START_THRESHOLD_S = 2.0     # < 2 seconds to interactive
KILL_SWITCH_FRAMES = 3           # < 3 frames for response

# Physics validation tolerances
ASHRAE_TOLERANCE = 0.1           # PMV within ±0.1
ALPERT_TOLERANCE = 0.10          # 10% of correlation
INVERSE_DESIGN_ITERATIONS = 50   # Max iterations for convergence


# ============================================================================
# FIXTURES
# ============================================================================

@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    measured_value: Any
    threshold: Any
    unit: str = ""
    details: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} | {self.test_name}: {self.measured_value} {self.unit} (threshold: {self.threshold})"


@dataclass  
class ValidationReport:
    """Complete validation report for a deployment."""
    deployment: str
    results: List[ValidationResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def add(self, result: ValidationResult):
        self.results.append(result)
    
    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)
    
    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)
    
    def finalize(self):
        self.end_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment": self.deployment,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "duration_s": (self.end_time or time.time()) - self.start_time,
            "results": [
                {
                    "test": r.test_name,
                    "passed": r.passed,
                    "value": r.measured_value,
                    "threshold": r.threshold,
                    "unit": r.unit,
                    "details": r.details,
                }
                for r in self.results
            ]
        }
    
    def __str__(self):
        lines = [
            f"╔══════════════════════════════════════════════════════════════╗",
            f"║ VALIDATION REPORT: {self.deployment:^41} ║",
            f"╠══════════════════════════════════════════════════════════════╣",
        ]
        for r in self.results:
            lines.append(f"║ {str(r):<62} ║")
        lines.append(f"╠══════════════════════════════════════════════════════════════╣")
        status = "CERTIFIED" if self.passed else "FAILED"
        lines.append(f"║ STATUS: {status:^53} ║")
        lines.append(f"║ Pass Rate: {self.pass_rate*100:.1f}%{' '*48}║")
        lines.append(f"╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


@pytest.fixture
def validation_report(request):
    """Create a validation report for the current test module."""
    deployment = getattr(request.module, 'DEPLOYMENT_NAME', 'Unknown')
    report = ValidationReport(deployment=deployment)
    yield report
    report.finalize()
    print("\n" + str(report))


@pytest.fixture
def bridge_connection():
    """Establish connection to the physics bridge."""
    from tests.bridge_client import BridgeClient
    client = BridgeClient()
    if client.connect():
        yield client
        client.disconnect()
    else:
        pytest.skip("Bridge not available")


@pytest.fixture
def dominion_process():
    """Launch DOMINION process for testing."""
    if not DOMINION_EXE.exists():
        pytest.skip(f"DOMINION executable not found at {DOMINION_EXE}")
    
    proc = subprocess.Popen(
        [str(DOMINION_EXE)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for startup
    time.sleep(0.5)
    
    yield proc
    
    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def measure_latency_us() -> float:
    """Measure round-trip latency to bridge in microseconds."""
    try:
        # Open shared memory
        with open(BRIDGE_SHM, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            
            # Send ping timestamp
            t0 = time.perf_counter_ns()
            
            # Write ping command
            mm.seek(0)
            mm.write(struct.pack('<Q', t0))  # 8-byte timestamp
            
            # Wait for response (poll)
            timeout = 0.01  # 10ms timeout
            start = time.perf_counter()
            while time.perf_counter() - start < timeout:
                mm.seek(8)
                response = struct.unpack('<Q', mm.read(8))[0]
                if response == t0:  # Echo received
                    t1 = time.perf_counter_ns()
                    mm.close()
                    return (t1 - t0) / 1000  # Convert to microseconds
            
            mm.close()
            return float('inf')
    except Exception as e:
        return float('inf')


def get_process_memory_mb(pid: int) -> float:
    """Get memory usage of a process in MB."""
    try:
        proc = psutil.Process(pid)
        return proc.memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def compute_pmv(air_temp_c: float, mean_radiant_temp_c: float, 
                relative_humidity: float, air_velocity_mps: float,
                metabolic_rate_met: float, clothing_clo: float) -> float:
    """Compute PMV using Fanger's equation (simplified).
    
    Args:
        air_temp_c: Air temperature (°C)
        mean_radiant_temp_c: Mean radiant temperature (°C)
        relative_humidity: Relative humidity (0-1 scale)
        air_velocity_mps: Air velocity (m/s)
        metabolic_rate_met: Metabolic rate (met)
        clothing_clo: Clothing insulation (clo)
    
    Returns:
        PMV value (-3 to +3)
    """
    ta = air_temp_c
    tr = mean_radiant_temp_c
    vel = air_velocity_mps
    met = metabolic_rate_met
    clo = clothing_clo
    rh = relative_humidity * 100  # Convert to percentage
    
    # Metabolic rate in W/m²
    M = met * 58.15
    
    # Clothing insulation in m²K/W
    Icl = clo * 0.155
    
    # Clothing area factor
    if Icl <= 0.078:
        fcl = 1.0 + 1.29 * Icl
    else:
        fcl = 1.05 + 0.645 * Icl
    
    # Operative temperature (simplified)
    to = (ta + tr) / 2.0
    
    # Heat transfer coefficient
    hc = 12.1 * (vel ** 0.5) if vel > 0.1 else 2.38 * abs(ta - to) ** 0.25
    hc = max(hc, 2.38 * abs(ta - to) ** 0.25)
    
    # Clothing surface temperature (iterative, simplified)
    tcl = to
    for _ in range(10):
        tcl_new = 35.7 - 0.028 * M - Icl * (
            3.96e-8 * fcl * ((tcl + 273.0) ** 4 - (tr + 273.0) ** 4)
            + fcl * hc * (tcl - ta)
        )
        if abs(tcl_new - tcl) < 0.001:
            break
        tcl = tcl_new
    
    # Heat loss components
    HL1 = 3.05e-3 * (5733.0 - 6.99 * M - ta * 133.322)  # Skin diffusion
    HL2 = 0.42 * (M - 58.15)  # Sweating
    HL3 = 1.7e-5 * M * (5867.0 - ta * 133.322)  # Respiration latent
    HL4 = 0.0014 * M * (34.0 - ta)  # Respiration sensible
    HL5 = 3.96e-8 * fcl * ((tcl + 273.0) ** 4 - (tr + 273.0) ** 4)  # Radiation
    HL6 = fcl * hc * (tcl - ta)  # Convection
    
    # Thermal sensation
    TS = 0.303 * ((-0.036 * M) ** 0.5) + 0.028
    PMV = TS * (M - HL1 - HL2 - HL3 - HL4 - HL5 - HL6)
    
    return max(-3.0, min(3.0, PMV))


def alpert_ceiling_jet_velocity(Q: float, H: float, r: float) -> float:
    """Alpert's ceiling jet correlation for velocity.
    
    Args:
        Q: Heat release rate (kW)
        H: Ceiling height (m)
        r: Radial distance from fire axis (m)
    
    Returns:
        Ceiling jet velocity (m/s)
    """
    if r / H <= 0.15:
        # Near the plume axis
        return 0.96 * (Q / H) ** (1/3)
    else:
        # In the ceiling jet region
        return 0.195 * (Q ** (1/3)) * (H ** (1/2)) / (r ** (5/6))


def alpert_ceiling_jet_temp(Q: float, H: float, r: float, T_amb: float = 20.0) -> float:
    """Alpert's ceiling jet correlation for temperature rise.
    
    Args:
        Q: Heat release rate (kW)
        H: Ceiling height (m)
        r: Radial distance from fire axis (m)
        T_amb: Ambient temperature (°C)
    
    Returns:
        Ceiling jet temperature (°C)
    """
    if r / H <= 0.18:
        delta_T = 16.9 * (Q ** (2/3)) / (H ** (5/3))
    else:
        delta_T = 5.38 * (Q / r) ** (2/3) / H
    
    return T_amb + delta_T
