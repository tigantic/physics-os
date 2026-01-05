"""
Performance analysis and optimization recommendations.

This module provides analysis tools for identifying performance
bottlenecks and generating optimization recommendations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class OptimizationCategory(Enum):
    """Categories of optimization recommendations."""

    PRECISION = auto()
    BATCH_SIZE = auto()
    LAYER_FUSION = auto()
    MEMORY = auto()
    KERNEL = auto()
    ARCHITECTURE = auto()
    HARDWARE = auto()


class ImpactLevel(Enum):
    """Impact level of optimization."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class EffortLevel(Enum):
    """Implementation effort level."""

    TRIVIAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    SIGNIFICANT = 5


@dataclass
class OptimizationRecommendation:
    """Single optimization recommendation."""

    title: str
    description: str
    category: OptimizationCategory
    impact: ImpactLevel
    effort: EffortLevel

    # Expected improvement
    expected_speedup: float | None = None
    expected_memory_reduction: float | None = None

    # Implementation details
    implementation_steps: list[str] = field(default_factory=list)
    code_example: str = ""
    references: list[str] = field(default_factory=list)

    # Priority score (computed)
    priority_score: float = 0.0

    def __post_init__(self):
        """Calculate priority score."""
        # Higher impact, lower effort = higher priority
        self.priority_score = self.impact.value / max(self.effort.value, 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "category": self.category.name,
            "impact": self.impact.name,
            "effort": self.effort.name,
            "expected_speedup": self.expected_speedup,
            "expected_memory_reduction": self.expected_memory_reduction,
            "implementation_steps": self.implementation_steps,
            "priority_score": round(self.priority_score, 2),
        }


@dataclass
class BottleneckAnalysis:
    """Analysis of performance bottleneck."""

    name: str
    bottleneck_type: str
    severity: ImpactLevel

    # Metrics
    time_percentage: float = 0.0
    memory_percentage: float = 0.0

    # Location
    layers: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)

    # Root cause
    root_cause: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    # Recommendations
    recommendations: list[OptimizationRecommendation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.bottleneck_type,
            "severity": self.severity.name,
            "time_percentage": round(self.time_percentage, 2),
            "memory_percentage": round(self.memory_percentage, 2),
            "layers": self.layers,
            "root_cause": self.root_cause,
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


@dataclass
class PerformanceAnalysis:
    """Complete performance analysis."""

    model_name: str
    total_time_ms: float

    # Analysis results
    bottlenecks: list[BottleneckAnalysis] = field(default_factory=list)
    recommendations: list[OptimizationRecommendation] = field(default_factory=list)

    # Performance characteristics
    compute_bound: bool = False
    memory_bound: bool = False

    # Metrics
    compute_utilization: float = 0.0
    memory_bandwidth_utilization: float = 0.0

    # Summary
    primary_bottleneck: str = ""
    optimization_potential: float = 0.0

    def get_top_recommendations(self, n: int = 5) -> list[OptimizationRecommendation]:
        """Get top N recommendations by priority."""
        sorted_recs = sorted(
            self.recommendations, key=lambda r: r.priority_score, reverse=True
        )
        return sorted_recs[:n]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "total_time_ms": round(self.total_time_ms, 4),
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "compute_bound": self.compute_bound,
            "memory_bound": self.memory_bound,
            "compute_utilization": round(self.compute_utilization, 2),
            "memory_bandwidth_utilization": round(self.memory_bandwidth_utilization, 2),
            "primary_bottleneck": self.primary_bottleneck,
            "optimization_potential": round(self.optimization_potential, 2),
        }

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"Performance Analysis: {self.model_name}",
            "=" * 50,
            f"Total inference time: {self.total_time_ms:.4f} ms",
            "",
            f"Compute bound: {'Yes' if self.compute_bound else 'No'}",
            f"Memory bound: {'Yes' if self.memory_bound else 'No'}",
            f"Primary bottleneck: {self.primary_bottleneck}",
            f"Optimization potential: {self.optimization_potential:.1f}x",
            "",
            "Top Recommendations:",
        ]

        for i, rec in enumerate(self.get_top_recommendations(3), 1):
            lines.append(f"  {i}. {rec.title} (Impact: {rec.impact.name})")

        return "\n".join(lines)


class PerformanceAnalyzer:
    """
    Analyzes performance profiles and generates recommendations.
    """

    def __init__(self):
        """Initialize analyzer."""
        self._thresholds = {
            "high_latency_layer_pct": 10.0,  # Layer taking >10% of time
            "memory_bound_threshold": 0.7,  # Memory BW >70% = memory bound
            "compute_bound_threshold": 0.7,  # Compute util >70% = compute bound
            "fusion_candidate_pct": 5.0,  # Small layers to consider fusing
        }

    def analyze(
        self,
        profile_result: Any,
    ) -> PerformanceAnalysis:
        """
        Analyze profile results.

        Args:
            profile_result: ProfileResult from profiler

        Returns:
            PerformanceAnalysis with bottlenecks and recommendations
        """
        analysis = PerformanceAnalysis(
            model_name=profile_result.model_name,
            total_time_ms=profile_result.total_time_ms,
        )

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(profile_result)
        analysis.bottlenecks = bottlenecks

        # Generate recommendations
        recommendations = self._generate_recommendations(profile_result, bottlenecks)
        analysis.recommendations = recommendations

        # Determine if compute or memory bound
        analysis.compute_bound, analysis.memory_bound = self._classify_bound(
            profile_result
        )

        # Set primary bottleneck
        if bottlenecks:
            analysis.primary_bottleneck = bottlenecks[0].name

        # Estimate optimization potential
        analysis.optimization_potential = self._estimate_optimization_potential(
            profile_result, recommendations
        )

        return analysis

    def _identify_bottlenecks(
        self,
        profile_result: Any,
    ) -> list[BottleneckAnalysis]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        # High-latency layers
        for layer in profile_result.layer_profiles:
            if layer.percentage > self._thresholds["high_latency_layer_pct"]:
                bottleneck = BottleneckAnalysis(
                    name=f"High latency: {layer.name}",
                    bottleneck_type="latency",
                    severity=self._severity_from_percentage(layer.percentage),
                    time_percentage=layer.percentage,
                    layers=[layer.name],
                    operations=[layer.operation_type.name],
                    root_cause=f"{layer.layer_type} operation consuming {layer.percentage:.1f}% of time",
                )
                bottlenecks.append(bottleneck)

        # Operation-level bottlenecks
        for op_type, op_profile in profile_result.operation_profiles.items():
            if op_profile.percentage > 30:  # Single op type >30%
                bottleneck = BottleneckAnalysis(
                    name=f"Dominant operation: {op_type.name}",
                    bottleneck_type="operation",
                    severity=(
                        ImpactLevel.HIGH
                        if op_profile.percentage > 50
                        else ImpactLevel.MEDIUM
                    ),
                    time_percentage=op_profile.percentage,
                    operations=[op_type.name],
                    root_cause=f"{op_type.name} operations consuming {op_profile.percentage:.1f}% of time",
                )
                bottlenecks.append(bottleneck)

        # Sort by severity
        bottlenecks.sort(key=lambda b: b.severity.value, reverse=True)

        return bottlenecks

    def _severity_from_percentage(self, pct: float) -> ImpactLevel:
        """Map percentage to severity level."""
        if pct > 30:
            return ImpactLevel.CRITICAL
        elif pct > 20:
            return ImpactLevel.HIGH
        elif pct > 10:
            return ImpactLevel.MEDIUM
        else:
            return ImpactLevel.LOW

    def _generate_recommendations(
        self,
        profile_result: Any,
        bottlenecks: list[BottleneckAnalysis],
    ) -> list[OptimizationRecommendation]:
        """Generate optimization recommendations."""
        recommendations = []

        # Precision recommendations
        recommendations.append(
            OptimizationRecommendation(
                title="Enable FP16 precision",
                description="Use mixed-precision inference with FP16 to approximately double throughput on Tensor Cores.",
                category=OptimizationCategory.PRECISION,
                impact=ImpactLevel.HIGH,
                effort=EffortLevel.LOW,
                expected_speedup=1.5,
                implementation_steps=[
                    "Export model with FP16 precision mode",
                    "Validate accuracy degradation is acceptable",
                    "Use TensorRT FP16 engine",
                ],
                code_example="""
# Enable FP16 in TensorRT export
config = ExportConfig(precision=Precision.FP16)
exporter.export(model, config)
""",
            )
        )

        # Batch size optimization
        recommendations.append(
            OptimizationRecommendation(
                title="Optimize batch size",
                description="Increase batch size to improve GPU utilization and throughput.",
                category=OptimizationCategory.BATCH_SIZE,
                impact=ImpactLevel.MEDIUM,
                effort=EffortLevel.TRIVIAL,
                expected_speedup=1.3,
                implementation_steps=[
                    "Profile with different batch sizes",
                    "Find optimal batch size for latency/throughput tradeoff",
                    "Configure inference server with optimal batch",
                ],
            )
        )

        # Check for convolution bottlenecks
        conv_ops = profile_result.operation_profiles.get("CONVOLUTION", None)
        if conv_ops and hasattr(conv_ops, "percentage") and conv_ops.percentage > 40:
            recommendations.append(
                OptimizationRecommendation(
                    title="Optimize convolution layers",
                    description="Convolutions dominate runtime. Consider architecture optimizations.",
                    category=OptimizationCategory.ARCHITECTURE,
                    impact=ImpactLevel.HIGH,
                    effort=EffortLevel.HIGH,
                    expected_speedup=1.2,
                    implementation_steps=[
                        "Consider depthwise separable convolutions",
                        "Evaluate smaller kernel sizes",
                        "Try channel pruning",
                    ],
                )
            )

        # Layer fusion recommendations
        small_layers = [
            l
            for l in profile_result.layer_profiles
            if l.percentage < self._thresholds["fusion_candidate_pct"]
        ]
        if len(small_layers) > 10:
            recommendations.append(
                OptimizationRecommendation(
                    title="Enable layer fusion",
                    description=f"Found {len(small_layers)} small layers that may benefit from fusion.",
                    category=OptimizationCategory.LAYER_FUSION,
                    impact=ImpactLevel.MEDIUM,
                    effort=EffortLevel.LOW,
                    expected_speedup=1.15,
                    implementation_steps=[
                        "Enable TensorRT layer fusion optimization",
                        "Use higher optimization level in export",
                    ],
                )
            )

        # Memory optimization
        if (
            hasattr(profile_result, "peak_memory_bytes")
            and profile_result.peak_memory_bytes > 1e9
        ):
            recommendations.append(
                OptimizationRecommendation(
                    title="Reduce memory footprint",
                    description="High memory usage detected. Optimize memory layout and reuse.",
                    category=OptimizationCategory.MEMORY,
                    impact=ImpactLevel.MEDIUM,
                    effort=EffortLevel.MEDIUM,
                    expected_memory_reduction=0.3,
                    implementation_steps=[
                        "Enable TensorRT memory pooling",
                        "Use in-place operations where possible",
                        "Consider gradient checkpointing for training",
                    ],
                )
            )

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)

        return recommendations

    def _classify_bound(
        self,
        profile_result: Any,
    ) -> tuple[bool, bool]:
        """Classify if model is compute or memory bound."""
        # Simplified heuristic based on operation mix
        compute_ops = {"CONVOLUTION", "MATRIX_MULTIPLY", "ATTENTION"}
        memory_ops = {"MEMORY", "RESHAPE", "ELEMENTWISE"}

        compute_time = 0.0
        memory_time = 0.0

        for op_type, profile in profile_result.operation_profiles.items():
            if op_type.name in compute_ops:
                compute_time += profile.percentage
            elif op_type.name in memory_ops:
                memory_time += profile.percentage

        compute_bound = compute_time > 60
        memory_bound = memory_time > 30

        return compute_bound, memory_bound

    def _estimate_optimization_potential(
        self,
        profile_result: Any,
        recommendations: list[OptimizationRecommendation],
    ) -> float:
        """Estimate potential speedup from recommendations."""
        # Conservative estimate: multiply expected speedups
        potential = 1.0

        for rec in recommendations[:3]:  # Top 3 recommendations
            if rec.expected_speedup:
                # Apply diminishing returns
                speedup_contribution = 1 + (rec.expected_speedup - 1) * 0.5
                potential *= speedup_contribution

        return min(potential, 3.0)  # Cap at 3x


def analyze_performance(profile_result: Any) -> PerformanceAnalysis:
    """
    Analyze performance from profile results.

    Args:
        profile_result: ProfileResult from profiler

    Returns:
        PerformanceAnalysis with recommendations
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.analyze(profile_result)


def identify_bottlenecks(profile_result: Any) -> list[BottleneckAnalysis]:
    """
    Identify bottlenecks from profile results.

    Args:
        profile_result: ProfileResult from profiler

    Returns:
        List of identified bottlenecks
    """
    analyzer = PerformanceAnalyzer()
    analysis = analyzer.analyze(profile_result)
    return analysis.bottlenecks


def recommend_optimizations(
    profile_result: Any,
    max_recommendations: int = 5,
) -> list[OptimizationRecommendation]:
    """
    Generate optimization recommendations.

    Args:
        profile_result: ProfileResult from profiler
        max_recommendations: Maximum number of recommendations

    Returns:
        List of optimization recommendations
    """
    analyzer = PerformanceAnalyzer()
    analysis = analyzer.analyze(profile_result)
    return analysis.get_top_recommendations(max_recommendations)
