"""
PIPELINES — Cross-Primitive Discovery Chains

Implements multi-stage discovery pipelines that chain Genesis primitives:
    - OT → SGW → RKHS → PH → GA (full pipeline)
    - Custom combinations for domain-specific analysis

The key insight: Each primitive extracts different information:
    - OT: Distribution geometry
    - SGW: Multi-scale structure
    - RMT: Spectral statistics
    - RKHS: Kernel embeddings
    - PH: Topological features
    - GA: Geometric transformations

Chaining them without going dense preserves the O(r³ log N) complexity.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import torch

from ontic.ml.discovery.findings import (
    Finding,
    FindingCollection,
    FindingType,
    Severity,
)
from ontic.ml.discovery.protocol import (
    GenesisPrimitive,
    PrimitiveChain,
    PrimitiveConfig,
    PrimitiveResult,
    PrimitiveType,
)


@dataclass
class PipelineStage:
    """
    Configuration for a single pipeline stage.
    
    Attributes:
        primitive_class: The primitive class to instantiate
        config: Configuration for the primitive
        name: Optional human-readable name
        enabled: Whether this stage is enabled
    """
    
    primitive_class: Type[GenesisPrimitive]
    config: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    enabled: bool = True
    
    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.primitive_class.__name__
        if self.config is None:
            self.config = {}
    
    def instantiate(self) -> GenesisPrimitive:
        """Create an instance of the primitive."""
        return self.primitive_class(**self.config)


@dataclass
class PipelineResult:
    """
    Result from executing a discovery pipeline.
    
    Attributes:
        stages: List of stage names executed
        results: PrimitiveResult from each stage
        findings: Collected findings from all stages
        total_time: Total execution time
        metadata: Pipeline-level metadata
    """
    
    stages: List[str]
    results: List[PrimitiveResult]
    findings: FindingCollection
    total_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def final_result(self) -> Optional[PrimitiveResult]:
        """Get the result from the last stage."""
        return self.results[-1] if self.results else None
    
    def get_stage_result(self, stage_name: str) -> Optional[PrimitiveResult]:
        """Get result from a specific stage by name."""
        for name, result in zip(self.stages, self.results):
            if name == stage_name:
                return result
        return None
    
    def summary(self) -> Dict[str, Any]:
        """Return pipeline execution summary."""
        return {
            "stages": self.stages,
            "n_stages": len(self.stages),
            "total_time_ms": self.total_time * 1000,
            "findings_summary": self.findings.summary(),
            "stage_times_ms": [r.elapsed_time * 1000 for r in self.results],
        }


class Pipeline:
    """
    Base class for discovery pipelines.
    
    Provides common functionality for building and executing
    multi-stage primitive chains.
    
    Example:
        >>> from ontic.ml.discovery.pipelines import Pipeline
        >>> from ontic.ml.discovery.primitives import (
        ...     OptimalTransportPrimitive,
        ...     TopologyPrimitive,
        ... )
        >>> 
        >>> pipeline = Pipeline([
        ...     OptimalTransportPrimitive(epsilon=0.01),
        ...     TopologyPrimitive(max_dimension=2),
        ... ])
        >>> 
        >>> result = pipeline.execute(input_data)
    """
    
    def __init__(
        self,
        primitives: Sequence[Union[GenesisPrimitive, PipelineStage]],
        name: str = "Pipeline",
    ) -> None:
        """
        Initialize pipeline.
        
        Args:
            primitives: Sequence of primitives or stage configs
            name: Human-readable pipeline name
        """
        self.name = name
        self.stages: List[GenesisPrimitive] = []
        self.stage_names: List[str] = []
        
        for item in primitives:
            if isinstance(item, GenesisPrimitive):
                self.stages.append(item)
                self.stage_names.append(item.name)
            elif isinstance(item, PipelineStage):
                if item.enabled:
                    primitive = item.instantiate()
                    self.stages.append(primitive)
                    self.stage_names.append(item.name)
            else:
                raise ValueError(f"Invalid pipeline item: {type(item)}")
    
    def execute(self, input_data: Any) -> PipelineResult:
        """
        Execute the full pipeline.
        
        Args:
            input_data: Initial input to first stage
            
        Returns:
            PipelineResult with all stage results and findings
        """
        start_time = time.perf_counter()
        
        results: List[PrimitiveResult] = []
        findings = FindingCollection()
        current_input = input_data
        
        for stage in self.stages:
            # Run discovery on this stage
            result = stage.discover(current_input)
            results.append(result)
            
            # Collect findings
            for finding in result.findings:
                findings.add(finding)
            
            # Pass result to next stage
            current_input = result
        
        total_time = time.perf_counter() - start_time
        
        return PipelineResult(
            stages=self.stage_names.copy(),
            results=results,
            findings=findings,
            total_time=total_time,
            metadata={
                "pipeline_name": self.name,
                "n_stages": len(self.stages),
            },
        )
    
    def add_stage(self, primitive: GenesisPrimitive) -> "Pipeline":
        """Add a stage to the pipeline."""
        self.stages.append(primitive)
        self.stage_names.append(primitive.name)
        return self
    
    def reset_all(self) -> None:
        """Reset state of all stages."""
        for stage in self.stages:
            stage.reset_state()
    
    def __len__(self) -> int:
        return len(self.stages)
    
    def __repr__(self) -> str:
        stages_str = " → ".join(self.stage_names)
        return f"Pipeline({self.name}: {stages_str})"


class CrossPrimitivePipeline(Pipeline):
    """
    Full cross-primitive pipeline: OT → SGW → RKHS → PH → GA
    
    This is the flagship pipeline demonstrating the moat:
    chaining all Genesis primitives without densification.
    
    Each stage:
        1. OT: Analyze distribution geometry
        2. SGW: Extract multi-scale features
        3. RKHS: Compute kernel embeddings
        4. PH: Extract topological features
        5. GA: Apply geometric transformations
    
    Example:
        >>> from ontic.ml.discovery.pipelines import CrossPrimitivePipeline
        >>> 
        >>> pipeline = CrossPrimitivePipeline()
        >>> result = pipeline.execute(data)
        >>> 
        >>> # Get findings from all stages
        >>> for finding in result.findings:
        ...     print(f"[{finding.severity}] {finding.summary}")
    """
    
    def __init__(
        self,
        ot_config: Optional[Dict[str, Any]] = None,
        sgw_config: Optional[Dict[str, Any]] = None,
        rkhs_config: Optional[Dict[str, Any]] = None,
        ph_config: Optional[Dict[str, Any]] = None,
        ga_config: Optional[Dict[str, Any]] = None,
        include_rmt: bool = False,
        rmt_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize cross-primitive pipeline.
        
        Args:
            ot_config: Configuration for OT primitive
            sgw_config: Configuration for SGW primitive
            rkhs_config: Configuration for RKHS primitive
            ph_config: Configuration for PH primitive
            ga_config: Configuration for GA primitive
            include_rmt: Whether to include RMT stage
            rmt_config: Configuration for RMT primitive (if included)
        """
        # Import primitives here to avoid circular imports
        from ontic.ml.discovery.primitives import (
            OptimalTransportPrimitive,
            SpectralWaveletPrimitive,
            RandomMatrixPrimitive,
            KernelPrimitive,
            TopologyPrimitive,
            GeometricAlgebraPrimitive,
        )
        
        # Default configs
        ot_config = ot_config or {}
        sgw_config = sgw_config or {}
        rkhs_config = rkhs_config or {}
        ph_config = ph_config or {}
        ga_config = ga_config or {}
        rmt_config = rmt_config or {}
        
        # Build primitive list
        primitives: List[GenesisPrimitive] = [
            OptimalTransportPrimitive(**ot_config),
            SpectralWaveletPrimitive(**sgw_config),
        ]
        
        if include_rmt:
            primitives.append(RandomMatrixPrimitive(**rmt_config))
        
        primitives.extend([
            KernelPrimitive(**rkhs_config),
            TopologyPrimitive(**ph_config),
            GeometricAlgebraPrimitive(**ga_config),
        ])
        
        super().__init__(primitives, name="CrossPrimitivePipeline")
    
    def execute_with_checkpoints(
        self,
        input_data: Any,
        checkpoint_dir: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute pipeline with intermediate checkpoints.
        
        Args:
            input_data: Initial input
            checkpoint_dir: Directory to save checkpoints (optional)
            
        Returns:
            PipelineResult with checkpoints saved
        """
        import json
        import os
        
        start_time = time.perf_counter()
        
        results: List[PrimitiveResult] = []
        findings = FindingCollection()
        current_input = input_data
        
        for i, stage in enumerate(self.stages):
            stage_name = self.stage_names[i]
            
            # Run discovery
            result = stage.discover(current_input)
            results.append(result)
            
            # Collect findings
            for finding in result.findings:
                findings.add(finding)
            
            # Save checkpoint
            if checkpoint_dir is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"stage_{i}_{stage_name}.json"
                )
                
                checkpoint_data = {
                    "stage": i,
                    "stage_name": stage_name,
                    "elapsed_time": result.elapsed_time,
                    "metadata": result.metadata,
                    "n_findings": len(result.findings),
                }
                
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint_data, f, indent=2, default=str)
            
            current_input = result
        
        total_time = time.perf_counter() - start_time
        
        return PipelineResult(
            stages=self.stage_names.copy(),
            results=results,
            findings=findings,
            total_time=total_time,
            metadata={
                "pipeline_name": self.name,
                "n_stages": len(self.stages),
                "checkpointed": checkpoint_dir is not None,
            },
        )


class DeFiPipeline(Pipeline):
    """
    Specialized pipeline for DeFi vulnerability analysis.
    
    Focus: Distribution anomalies, kernel embeddings, topological defects
    in smart contract state.
    """
    
    def __init__(
        self,
        wasserstein_threshold: float = 0.1,
        mmd_threshold: float = 0.05,
    ) -> None:
        from ontic.ml.discovery.primitives import (
            OptimalTransportPrimitive,
            KernelPrimitive,
            TopologyPrimitive,
        )
        
        primitives = [
            OptimalTransportPrimitive(
                epsilon=0.01,
                threshold=wasserstein_threshold,
            ),
            KernelPrimitive(
                kernel_type="rbf",
                mmd_threshold=mmd_threshold,
            ),
            TopologyPrimitive(
                max_dimension=1,  # Focus on loops (cycles)
            ),
        ]
        
        super().__init__(primitives, name="DeFiPipeline")


class PhysicsPipeline(Pipeline):
    """
    Specialized pipeline for physics simulation analysis.
    
    Focus: Spectral analysis, RMT universality, conservation laws.
    """
    
    def __init__(
        self,
        scales: Optional[List[float]] = None,
        universality_class: str = "auto",
    ) -> None:
        from ontic.ml.discovery.primitives import (
            SpectralWaveletPrimitive,
            RandomMatrixPrimitive,
            TopologyPrimitive,
        )
        
        if scales is None:
            scales = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        
        primitives = [
            SpectralWaveletPrimitive(scales=scales),
            RandomMatrixPrimitive(universality_class=universality_class),
            TopologyPrimitive(max_dimension=2),
        ]
        
        super().__init__(primitives, name="PhysicsPipeline")


class GeometryPipeline(Pipeline):
    """
    Specialized pipeline for geometric data analysis.
    
    Focus: Optimal transport, topological features, geometric algebra.
    """
    
    def __init__(
        self,
        signature: Tuple[int, int, int] = (3, 0, 0),
        use_conformal: bool = True,
    ) -> None:
        from ontic.ml.discovery.primitives import (
            OptimalTransportPrimitive,
            TopologyPrimitive,
            GeometricAlgebraPrimitive,
        )
        
        primitives = [
            OptimalTransportPrimitive(epsilon=0.01),
            TopologyPrimitive(max_dimension=2),
            GeometricAlgebraPrimitive(
                signature=signature,
                use_conformal=use_conformal,
            ),
        ]
        
        super().__init__(primitives, name="GeometryPipeline")
