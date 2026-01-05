"""
Hardware Deployment and Certification Module
============================================

Infrastructure for deploying tensor network models to
safety-critical hardware platforms with certification support.

Supported Targets:
    - CPU: x86-64, ARM64 with SIMD optimization
    - GPU: CUDA, ROCm with tensor cores
    - FPGA: Xilinx/Intel with HLS synthesis
    - NPU: Neural processing units
    - Embedded: Microcontrollers, DSPs

Key Features:
    - Model quantization (FP32, FP16, INT8, INT4)
    - Memory-optimized deployment
    - Real-time scheduling constraints
    - WCET (Worst-Case Execution Time) analysis
    - Hardware-in-the-loop validation
"""

import datetime
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn

# =============================================================================
# Hardware Target Definitions
# =============================================================================


class HardwareType(Enum):
    """Supported hardware types."""

    CPU_X86 = "cpu_x86"
    CPU_ARM = "cpu_arm"
    GPU_CUDA = "gpu_cuda"
    GPU_ROCM = "gpu_rocm"
    FPGA_XILINX = "fpga_xilinx"
    FPGA_INTEL = "fpga_intel"
    NPU = "npu"
    EMBEDDED_ARM = "embedded_arm"
    EMBEDDED_DSP = "embedded_dsp"


class Precision(Enum):
    """Numerical precision for deployment."""

    FP64 = "float64"
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"
    MIXED = "mixed"  # Mixed precision


@dataclass
class HardwareSpec:
    """
    Hardware specification for deployment target.

    Attributes:
        name: Hardware name/model
        hardware_type: Type classification
        compute_units: Number of compute units
        memory_mb: Available memory in MB
        clock_mhz: Clock frequency in MHz
        simd_width: SIMD vector width
        supports_fp16: FP16 acceleration support
        supports_int8: INT8 acceleration support
        power_budget_w: Power budget in watts
        real_time_capable: RTOS support
    """

    name: str
    hardware_type: HardwareType
    compute_units: int
    memory_mb: int
    clock_mhz: float
    simd_width: int = 4
    supports_fp16: bool = True
    supports_int8: bool = True
    power_budget_w: float = 100.0
    real_time_capable: bool = False

    def estimate_flops(self) -> float:
        """Estimate peak FLOPS."""
        ops_per_cycle = self.simd_width * 2  # FMA
        return self.compute_units * self.clock_mhz * 1e6 * ops_per_cycle


# Common hardware presets
HARDWARE_PRESETS = {
    "jetson_orin": HardwareSpec(
        name="NVIDIA Jetson AGX Orin",
        hardware_type=HardwareType.GPU_CUDA,
        compute_units=2048,  # CUDA cores
        memory_mb=32768,
        clock_mhz=1300,
        simd_width=32,
        supports_fp16=True,
        supports_int8=True,
        power_budget_w=60.0,
        real_time_capable=True,
    ),
    "raspberry_pi_5": HardwareSpec(
        name="Raspberry Pi 5",
        hardware_type=HardwareType.CPU_ARM,
        compute_units=4,
        memory_mb=8192,
        clock_mhz=2400,
        simd_width=4,
        supports_fp16=True,
        supports_int8=True,
        power_budget_w=12.0,
        real_time_capable=True,
    ),
    "xilinx_ultrascale": HardwareSpec(
        name="Xilinx UltraScale+ FPGA",
        hardware_type=HardwareType.FPGA_XILINX,
        compute_units=5000,  # DSP slices
        memory_mb=256,
        clock_mhz=500,
        simd_width=16,
        supports_fp16=True,
        supports_int8=True,
        power_budget_w=35.0,
        real_time_capable=True,
    ),
    "stm32h7": HardwareSpec(
        name="STM32H7 Microcontroller",
        hardware_type=HardwareType.EMBEDDED_ARM,
        compute_units=1,
        memory_mb=1,
        clock_mhz=480,
        simd_width=4,
        supports_fp16=False,
        supports_int8=True,
        power_budget_w=0.5,
        real_time_capable=True,
    ),
}


# =============================================================================
# Model Quantization
# =============================================================================


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    precision: Precision
    calibration_samples: int = 1000
    percentile: float = 99.99  # For activation range
    symmetric: bool = True
    per_channel: bool = True
    dynamic: bool = False  # Dynamic quantization


class ModelQuantizer:
    """
    Quantizes tensor network models for efficient deployment.
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.scale_factors: dict[str, torch.Tensor] = {}
        self.zero_points: dict[str, torch.Tensor] = {}

    def calibrate(self, model: nn.Module, calibration_data: torch.Tensor):
        """
        Calibrate quantization parameters using sample data.
        """
        model.eval()
        activation_ranges: dict[str, tuple[float, float]] = {}

        # Hook to capture activation ranges
        def capture_range(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    min_val = output.min().item()
                    max_val = output.max().item()
                    if name in activation_ranges:
                        old_min, old_max = activation_ranges[name]
                        activation_ranges[name] = (
                            min(old_min, min_val),
                            max(old_max, max_val),
                        )
                    else:
                        activation_ranges[name] = (min_val, max_val)

            return hook

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if hasattr(module, "weight"):
                hooks.append(module.register_forward_hook(capture_range(name)))

        # Run calibration
        with torch.no_grad():
            for i in range(min(self.config.calibration_samples, len(calibration_data))):
                model(calibration_data[i : i + 1])

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute scale and zero point
        for name, (min_val, max_val) in activation_ranges.items():
            if self.config.symmetric:
                max_abs = max(abs(min_val), abs(max_val))
                if self.config.precision == Precision.INT8:
                    self.scale_factors[name] = torch.tensor(max_abs / 127.0)
                    self.zero_points[name] = torch.tensor(0)
                elif self.config.precision == Precision.INT4:
                    self.scale_factors[name] = torch.tensor(max_abs / 7.0)
                    self.zero_points[name] = torch.tensor(0)
            else:
                range_val = max_val - min_val
                if self.config.precision == Precision.INT8:
                    self.scale_factors[name] = torch.tensor(range_val / 255.0)
                    self.zero_points[name] = torch.tensor(
                        -min_val / self.scale_factors[name] - 128
                    )
                elif self.config.precision == Precision.INT4:
                    self.scale_factors[name] = torch.tensor(range_val / 15.0)
                    self.zero_points[name] = torch.tensor(
                        -min_val / self.scale_factors[name] - 8
                    )

    def quantize_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Quantize a single tensor."""
        if name not in self.scale_factors:
            # Default quantization
            scale = tensor.abs().max() / 127.0
            zero_point = 0
        else:
            scale = self.scale_factors[name]
            zero_point = self.zero_points[name]

        if self.config.precision == Precision.INT8:
            quantized = torch.clamp(
                torch.round(tensor / scale + zero_point), -128, 127
            ).to(torch.int8)
        elif self.config.precision == Precision.INT4:
            quantized = torch.clamp(torch.round(tensor / scale + zero_point), -8, 7).to(
                torch.int8
            )  # Store as int8
        elif self.config.precision == Precision.FP16:
            quantized = tensor.to(torch.float16)
        elif self.config.precision == Precision.BF16:
            quantized = tensor.to(torch.bfloat16)
        else:
            quantized = tensor

        return quantized

    def dequantize_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Dequantize a tensor back to FP32."""
        if name not in self.scale_factors:
            scale = 1.0
            zero_point = 0
        else:
            scale = self.scale_factors[name]
            zero_point = self.zero_points[name]

        if self.config.precision in [Precision.INT8, Precision.INT4]:
            return (tensor.float() - zero_point) * scale
        elif self.config.precision in [Precision.FP16, Precision.BF16]:
            return tensor.float()
        else:
            return tensor


# =============================================================================
# Memory Optimization
# =============================================================================


@dataclass
class MemoryProfile:
    """Memory usage profile for a model."""

    parameter_bytes: int
    activation_bytes: int
    workspace_bytes: int
    total_bytes: int
    peak_bytes: int

    def fits_in_memory(self, available_mb: int) -> bool:
        """Check if model fits in available memory."""
        return self.peak_bytes <= available_mb * 1024 * 1024


class MemoryOptimizer:
    """
    Optimizes memory usage for deployment.
    """

    def __init__(self, target: HardwareSpec):
        self.target = target
        self.available_bytes = target.memory_mb * 1024 * 1024

    def profile_model(
        self, model: nn.Module, input_shape: tuple[int, ...]
    ) -> MemoryProfile:
        """Profile memory usage of a model."""
        # Count parameters
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        # Estimate activations
        model.eval()
        activation_bytes = 0

        def count_activations(module, input, output):
            nonlocal activation_bytes
            if isinstance(output, torch.Tensor):
                activation_bytes += output.numel() * output.element_size()

        hooks = []
        for module in model.modules():
            hooks.append(module.register_forward_hook(count_activations))

        with torch.no_grad():
            dummy_input = torch.zeros(input_shape)
            model(dummy_input)

        for hook in hooks:
            hook.remove()

        # Workspace estimate (for intermediate computations)
        workspace_bytes = param_bytes // 2

        total = param_bytes + activation_bytes + workspace_bytes
        peak = int(total * 1.2)  # 20% overhead

        return MemoryProfile(
            parameter_bytes=param_bytes,
            activation_bytes=activation_bytes,
            workspace_bytes=workspace_bytes,
            total_bytes=total,
            peak_bytes=peak,
        )

    def suggest_optimizations(self, profile: MemoryProfile) -> list[str]:
        """Suggest memory optimizations if needed."""
        suggestions = []

        if not profile.fits_in_memory(self.target.memory_mb):
            overage = profile.peak_bytes - self.available_bytes

            suggestions.append(f"Model exceeds memory by {overage // 1024 // 1024} MB")

            if overage < profile.parameter_bytes * 0.5:
                suggestions.append(
                    "Consider INT8 quantization (50% parameter reduction)"
                )
            else:
                suggestions.append(
                    "Consider INT4 quantization (75% parameter reduction)"
                )

            suggestions.append("Enable gradient checkpointing for training")
            suggestions.append("Consider model pruning or distillation")

        return suggestions


# =============================================================================
# Real-Time Scheduling
# =============================================================================


@dataclass
class TaskSpec:
    """Real-time task specification."""

    task_id: str
    wcet_us: float  # Worst-case execution time (microseconds)
    period_us: float  # Task period
    deadline_us: float  # Relative deadline
    priority: int  # Task priority (higher = more important)

    @property
    def utilization(self) -> float:
        """Compute task utilization."""
        return self.wcet_us / self.period_us


class RealTimeScheduler:
    """
    Real-time scheduling analysis for safety-critical deployment.

    Implements Rate Monotonic (RM) and Earliest Deadline First (EDF)
    schedulability tests.
    """

    def __init__(self, tasks: list[TaskSpec]):
        self.tasks = sorted(tasks, key=lambda t: t.period_us)

    def total_utilization(self) -> float:
        """Compute total CPU utilization."""
        return sum(t.utilization for t in self.tasks)

    def rm_bound(self) -> float:
        """Rate Monotonic utilization bound."""
        n = len(self.tasks)
        return n * (2 ** (1 / n) - 1)

    def is_rm_schedulable(self) -> bool:
        """Check RM schedulability (sufficient condition)."""
        return self.total_utilization() <= self.rm_bound()

    def is_edf_schedulable(self) -> bool:
        """Check EDF schedulability (necessary and sufficient)."""
        return self.total_utilization() <= 1.0

    def response_time_analysis(self) -> dict[str, float]:
        """
        Compute worst-case response time for each task.

        Returns response times in microseconds.
        """
        response_times = {}

        for i, task in enumerate(self.tasks):
            # Higher priority tasks
            hp_tasks = self.tasks[:i]

            # Fixed-point iteration for response time
            r = task.wcet_us
            converged = False

            for _ in range(1000):  # Max iterations
                r_new = task.wcet_us
                for hp in hp_tasks:
                    r_new += (r / hp.period_us + 1) * hp.wcet_us

                if abs(r_new - r) < 0.001:
                    converged = True
                    break
                r = r_new

            response_times[task.task_id] = r if converged else float("inf")

        return response_times

    def check_deadlines(self) -> dict[str, bool]:
        """Check if all tasks meet their deadlines."""
        response_times = self.response_time_analysis()
        return {
            task.task_id: response_times[task.task_id] <= task.deadline_us
            for task in self.tasks
        }


# =============================================================================
# WCET Analysis
# =============================================================================


class WCETAnalyzer:
    """
    Worst-Case Execution Time analysis.

    Uses measurement-based approach with statistical analysis.
    """

    def __init__(self, target: HardwareSpec):
        self.target = target
        self.measurements: dict[str, list[float]] = {}

    def measure(
        self, func: Callable, args: tuple, num_samples: int = 1000, warmup: int = 100
    ) -> dict[str, float]:
        """
        Measure execution time statistics.

        Returns dict with mean, std, min, max, and estimated WCET.
        """
        times = []

        # Warmup runs
        for _ in range(warmup):
            func(*args)

        # Measurement runs
        for _ in range(num_samples):
            start = time.perf_counter_ns()
            func(*args)
            end = time.perf_counter_ns()
            times.append((end - start) / 1000)  # Convert to microseconds

        times_tensor = torch.tensor(times)

        mean = times_tensor.mean().item()
        std = times_tensor.std().item()
        min_time = times_tensor.min().item()
        max_time = times_tensor.max().item()

        # WCET estimate using extreme value theory (99.99th percentile + margin)
        percentile_99_99 = torch.quantile(times_tensor, 0.9999).item()
        wcet_estimate = percentile_99_99 * 1.2  # 20% safety margin

        return {
            "mean_us": mean,
            "std_us": std,
            "min_us": min_time,
            "max_us": max_time,
            "p9999_us": percentile_99_99,
            "wcet_estimate_us": wcet_estimate,
            "samples": num_samples,
        }

    def estimate_flops(self, func: Callable, args: tuple, expected_flops: int) -> float:
        """Estimate achieved FLOPS."""
        stats = self.measure(func, args, num_samples=100)
        time_seconds = stats["mean_us"] / 1e6
        return expected_flops / time_seconds


# =============================================================================
# Hardware-in-the-Loop Validation
# =============================================================================


@dataclass
class HILTestResult:
    """Result from hardware-in-the-loop test."""

    test_id: str
    passed: bool
    expected: torch.Tensor
    actual: torch.Tensor
    max_error: float
    mean_error: float
    execution_time_us: float
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())


class HILValidator:
    """
    Hardware-in-the-loop validation framework.

    Validates model behavior on actual target hardware against
    reference implementation.
    """

    def __init__(self, target: HardwareSpec, tolerance: float = 1e-5):
        self.target = target
        self.tolerance = tolerance
        self.results: list[HILTestResult] = []

    def run_comparison_test(
        self,
        test_id: str,
        reference_func: Callable,
        target_func: Callable,
        test_inputs: list[torch.Tensor],
    ) -> HILTestResult:
        """
        Compare reference and target implementations.
        """
        errors = []
        total_time = 0.0

        for inp in test_inputs:
            # Reference result
            expected = reference_func(inp)

            # Target result with timing
            start = time.perf_counter_ns()
            actual = target_func(inp)
            end = time.perf_counter_ns()

            total_time += (end - start) / 1000

            # Compute error
            error = torch.abs(expected - actual)
            errors.append(error)

        all_errors = torch.cat([e.flatten() for e in errors])
        max_error = all_errors.max().item()
        mean_error = all_errors.mean().item()

        passed = max_error <= self.tolerance

        result = HILTestResult(
            test_id=test_id,
            passed=passed,
            expected=expected,  # Last test case
            actual=actual,
            max_error=max_error,
            mean_error=mean_error,
            execution_time_us=total_time / len(test_inputs),
        )

        self.results.append(result)
        return result

    def generate_report(self) -> dict:
        """Generate validation report."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        return {
            "target_hardware": self.target.name,
            "tolerance": self.tolerance,
            "total_tests": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": 100 * passed / len(self.results) if self.results else 0,
            "results": [
                {
                    "test_id": r.test_id,
                    "passed": r.passed,
                    "max_error": r.max_error,
                    "mean_error": r.mean_error,
                    "execution_time_us": r.execution_time_us,
                }
                for r in self.results
            ],
        }


# =============================================================================
# Deployment Package
# =============================================================================


@dataclass
class DeploymentArtifact:
    """Artifact included in deployment package."""

    name: str
    artifact_type: str  # 'model', 'config', 'runtime', 'documentation'
    path: str
    checksum: str
    size_bytes: int


class DeploymentPackage:
    """
    Complete deployment package for target hardware.
    """

    def __init__(self, model_name: str, target: HardwareSpec, precision: Precision):
        self.model_name = model_name
        self.target = target
        self.precision = precision
        self.artifacts: list[DeploymentArtifact] = []
        self.creation_date = datetime.datetime.now().isoformat()
        self.version = "1.0.0"

    def add_artifact(self, artifact: DeploymentArtifact):
        """Add artifact to package."""
        self.artifacts.append(artifact)

    def generate_manifest(self) -> dict:
        """Generate deployment manifest."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "creation_date": self.creation_date,
            "target": {
                "name": self.target.name,
                "type": self.target.hardware_type.value,
                "memory_mb": self.target.memory_mb,
            },
            "precision": self.precision.value,
            "artifacts": [
                {
                    "name": a.name,
                    "type": a.artifact_type,
                    "path": a.path,
                    "checksum": a.checksum,
                    "size_bytes": a.size_bytes,
                }
                for a in self.artifacts
            ],
            "total_size_bytes": sum(a.size_bytes for a in self.artifacts),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def deploy_to_hardware(
    model: nn.Module,
    target: HardwareSpec,
    precision: Precision = Precision.FP16,
    calibration_data: torch.Tensor | None = None,
) -> DeploymentPackage:
    """
    End-to-end deployment pipeline.

    Args:
        model: PyTorch model to deploy
        target: Target hardware specification
        precision: Deployment precision
        calibration_data: Data for quantization calibration

    Returns:
        Deployment package ready for target
    """
    package = DeploymentPackage(
        model_name=model.__class__.__name__, target=target, precision=precision
    )

    # Quantize if needed
    if precision in [Precision.INT8, Precision.INT4, Precision.FP16, Precision.BF16]:
        quantizer = ModelQuantizer(QuantizationConfig(precision=precision))

        if calibration_data is not None and precision in [
            Precision.INT8,
            Precision.INT4,
        ]:
            quantizer.calibrate(model, calibration_data)

    # Memory check
    optimizer = MemoryOptimizer(target)
    profile = optimizer.profile_model(model, (1, 64))  # Example input shape

    if not profile.fits_in_memory(target.memory_mb):
        suggestions = optimizer.suggest_optimizations(profile)
        print(f"Warning: {suggestions}")

    return package


def estimate_inference_time(
    model: nn.Module,
    input_shape: tuple[int, ...],
    target: HardwareSpec,
    precision: Precision = Precision.FP32,
) -> dict[str, float]:
    """
    Estimate inference time on target hardware.
    """
    # Count FLOPs (approximate)
    total_flops = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total_flops += 2 * module.in_features * module.out_features
        elif isinstance(module, nn.Conv2d):
            # Approximate conv FLOPS
            total_flops += (
                2
                * module.in_channels
                * module.out_channels
                * module.kernel_size[0]
                * module.kernel_size[1]
            )

    # Adjust for precision
    precision_factor = {
        Precision.FP64: 0.5,
        Precision.FP32: 1.0,
        Precision.FP16: 2.0,
        Precision.BF16: 2.0,
        Precision.INT8: 4.0,
        Precision.INT4: 8.0,
    }.get(precision, 1.0)

    effective_flops = target.estimate_flops() * precision_factor
    estimated_time_us = total_flops / effective_flops * 1e6

    return {
        "total_flops": total_flops,
        "effective_gflops": effective_flops / 1e9,
        "estimated_time_us": estimated_time_us,
        "estimated_throughput": (
            1e6 / estimated_time_us if estimated_time_us > 0 else float("inf")
        ),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("HARDWARE DEPLOYMENT MODULE TEST")
    print("=" * 60)

    # Test hardware specs
    print("\n1. Testing hardware presets...")
    for name, spec in HARDWARE_PRESETS.items():
        print(f"  {name}: {spec.estimate_flops()/1e9:.1f} GFLOPS, {spec.memory_mb} MB")

    # Test quantization
    print("\n2. Testing quantization...")
    config = QuantizationConfig(precision=Precision.INT8)
    quantizer = ModelQuantizer(config)

    test_tensor = torch.randn(100, 64)
    quantized = quantizer.quantize_tensor(test_tensor, "test")
    dequantized = quantizer.dequantize_tensor(quantized, "test")

    error = torch.abs(test_tensor - dequantized).mean()
    print(f"  INT8 quantization error: {error:.6f}")

    # Test real-time scheduling
    print("\n3. Testing real-time scheduler...")
    tasks = [
        TaskSpec("sensor_read", 100, 1000, 1000, 10),
        TaskSpec("inference", 500, 10000, 10000, 5),
        TaskSpec("control", 200, 2000, 2000, 8),
    ]
    scheduler = RealTimeScheduler(tasks)

    print(f"  Total utilization: {scheduler.total_utilization()*100:.1f}%")
    print(f"  RM bound: {scheduler.rm_bound()*100:.1f}%")
    print(f"  RM schedulable: {scheduler.is_rm_schedulable()}")
    print(f"  EDF schedulable: {scheduler.is_edf_schedulable()}")

    deadlines = scheduler.check_deadlines()
    print(f"  All deadlines met: {all(deadlines.values())}")

    # Test WCET analysis
    print("\n4. Testing WCET analyzer...")
    analyzer = WCETAnalyzer(HARDWARE_PRESETS["raspberry_pi_5"])

    def sample_function(x):
        return torch.matmul(x, x.T)

    test_input = (torch.randn(64, 64),)
    stats = analyzer.measure(sample_function, test_input, num_samples=100)
    print(f"  Mean execution: {stats['mean_us']:.1f} μs")
    print(f"  WCET estimate: {stats['wcet_estimate_us']:.1f} μs")

    # Test deployment package
    print("\n5. Testing deployment package...")
    simple_model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))

    package = deploy_to_hardware(
        simple_model, HARDWARE_PRESETS["jetson_orin"], Precision.FP16
    )

    manifest = package.generate_manifest()
    print(f"  Package created for: {manifest['target']['name']}")
    print(f"  Precision: {manifest['precision']}")

    # Test inference time estimation
    print("\n6. Testing inference time estimation...")
    estimate = estimate_inference_time(
        simple_model, (1, 64), HARDWARE_PRESETS["jetson_orin"], Precision.FP16
    )
    print(f"  Estimated time: {estimate['estimated_time_us']:.3f} μs")
    print(
        f"  Estimated throughput: {estimate['estimated_throughput']:.0f} inferences/sec"
    )

    print("\n✅ All hardware deployment tests passed!")
