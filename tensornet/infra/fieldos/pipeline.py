"""
Composable Pipeline System
============================

Build processing pipelines from reusable stages.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import numpy as np

from .field import Field

T = TypeVar("T")


# =============================================================================
# STAGE RESULT
# =============================================================================


class StageStatus(Enum):
    """Status of stage execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """
    Result of executing a pipeline stage.
    """

    status: StageStatus
    output: Any | None = None
    error: str | None = None
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == StageStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "error": self.error,
            "duration": self.duration,
            "metadata": self.metadata,
        }


# =============================================================================
# STAGE
# =============================================================================


class Stage(ABC):
    """
    Abstract base for pipeline stages.

    Each stage:
    - Takes input (field, data, or previous stage output)
    - Processes it
    - Returns StageResult
    """

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @abstractmethod
    def process(self, input_data: Any, context: dict[str, Any]) -> StageResult:
        """
        Process input data.

        Args:
            input_data: Input from previous stage or pipeline input
            context: Shared pipeline context

        Returns:
            StageResult with output or error
        """
        pass

    def __call__(self, input_data: Any, context: dict[str, Any]) -> StageResult:
        """Execute stage."""
        if not self._enabled:
            return StageResult(
                status=StageStatus.SKIPPED,
                output=input_data,
            )

        start = time.time()
        try:
            result = self.process(input_data, context)
            result.duration = time.time() - start
            return result
        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration=time.time() - start,
            )


# =============================================================================
# BUILT-IN STAGES
# =============================================================================


class FunctionStage(Stage):
    """Stage wrapping a simple function."""

    def __init__(self, func: Callable, name: str = ""):
        super().__init__(name or func.__name__)
        self.func = func

    def process(self, input_data: Any, context: dict[str, Any]) -> StageResult:
        output = self.func(input_data, **context)
        return StageResult(status=StageStatus.SUCCESS, output=output)


class TransformStage(Stage):
    """Stage that transforms field data."""

    def __init__(self, transform: Callable[[np.ndarray], np.ndarray], name: str = ""):
        super().__init__(name)
        self.transform = transform

    def process(self, input_data: Any, context: dict[str, Any]) -> StageResult:
        if isinstance(input_data, Field):
            new_data = self.transform(input_data.data)
            result_field = input_data.copy()
            result_field.update(new_data, source=self.name)
            return StageResult(status=StageStatus.SUCCESS, output=result_field)
        elif isinstance(input_data, np.ndarray):
            output = self.transform(input_data)
            return StageResult(status=StageStatus.SUCCESS, output=output)
        else:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"Unsupported input type: {type(input_data)}",
            )


class FilterStage(Stage):
    """Stage that filters based on condition."""

    def __init__(self, condition: Callable[[Any], bool], name: str = ""):
        super().__init__(name)
        self.condition = condition

    def process(self, input_data: Any, context: dict[str, Any]) -> StageResult:
        if self.condition(input_data):
            return StageResult(status=StageStatus.SUCCESS, output=input_data)
        else:
            return StageResult(status=StageStatus.SKIPPED, output=None)


class BranchStage(Stage):
    """Stage that branches based on condition."""

    def __init__(
        self,
        condition: Callable[[Any], bool],
        true_stage: Stage,
        false_stage: Stage,
        name: str = "",
    ):
        super().__init__(name)
        self.condition = condition
        self.true_stage = true_stage
        self.false_stage = false_stage

    def process(self, input_data: Any, context: dict[str, Any]) -> StageResult:
        if self.condition(input_data):
            return self.true_stage(input_data, context)
        else:
            return self.false_stage(input_data, context)


class AggregateStage(Stage):
    """Stage that aggregates multiple inputs."""

    def __init__(
        self,
        aggregator: Callable[[list[Any]], Any],
        name: str = "",
    ):
        super().__init__(name)
        self.aggregator = aggregator
        self._buffer: list[Any] = []

    def add(self, item: Any):
        """Add item to buffer."""
        self._buffer.append(item)

    def process(self, input_data: Any, context: dict[str, Any]) -> StageResult:
        # Add current input
        if input_data is not None:
            self._buffer.append(input_data)

        # Aggregate
        output = self.aggregator(self._buffer)
        self._buffer.clear()

        return StageResult(status=StageStatus.SUCCESS, output=output)


# =============================================================================
# PIPELINE
# =============================================================================


class Pipeline:
    """
    Composable processing pipeline.

    Example:
        pipeline = Pipeline("preprocess")
        pipeline.add(TransformStage(lambda x: x * 2, "double"))
        pipeline.add(TransformStage(np.clip, "clip"))

        result = pipeline.run(input_field)
    """

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self._stages: list[Stage] = []
        self._context: dict[str, Any] = {}
        self._results: list[StageResult] = []

    def add(self, stage: Stage) -> Pipeline:
        """Add stage to pipeline."""
        self._stages.append(stage)
        return self

    def add_function(self, func: Callable, name: str = "") -> Pipeline:
        """Add function as stage."""
        self._stages.append(FunctionStage(func, name))
        return self

    def add_transform(
        self,
        transform: Callable[[np.ndarray], np.ndarray],
        name: str = "",
    ) -> Pipeline:
        """Add transform stage."""
        self._stages.append(TransformStage(transform, name))
        return self

    def set_context(self, key: str, value: Any) -> Pipeline:
        """Set context value."""
        self._context[key] = value
        return self

    def run(
        self,
        input_data: Any,
        context: dict[str, Any] | None = None,
    ) -> StageResult:
        """
        Run pipeline on input.

        Returns:
            Final StageResult
        """
        # Merge contexts
        ctx = {**self._context}
        if context:
            ctx.update(context)

        self._results.clear()
        current = input_data

        for stage in self._stages:
            result = stage(current, ctx)
            self._results.append(result)

            if result.status == StageStatus.FAILED:
                return result

            if result.status == StageStatus.SKIPPED:
                continue

            current = result.output

        # Return final result
        return StageResult(
            status=StageStatus.SUCCESS,
            output=current,
            metadata={
                "stages_run": len(self._results),
                "stages_success": sum(1 for r in self._results if r.is_success),
            },
        )

    def __len__(self) -> int:
        return len(self._stages)

    def __iter__(self):
        return iter(self._stages)

    @property
    def results(self) -> list[StageResult]:
        """Get results from last run."""
        return self._results

    # -------------------------------------------------------------------------
    # Composition
    # -------------------------------------------------------------------------

    def then(self, other: Pipeline) -> Pipeline:
        """Chain another pipeline."""
        new_pipeline = Pipeline(f"{self.name}_then_{other.name}")
        new_pipeline._stages = self._stages + other._stages
        new_pipeline._context = {**self._context, **other._context}
        return new_pipeline

    def parallel(self, other: Pipeline) -> ParallelPipeline:
        """Run pipelines in parallel."""
        return ParallelPipeline([self, other])

    def repeat(self, times: int) -> Pipeline:
        """Repeat pipeline multiple times."""
        new_pipeline = Pipeline(f"{self.name}_x{times}")
        for _ in range(times):
            new_pipeline._stages.extend(self._stages)
        return new_pipeline


class ParallelPipeline:
    """Run multiple pipelines in parallel and aggregate results."""

    def __init__(
        self,
        pipelines: list[Pipeline],
        aggregator: Callable[[list[Any]], Any] | None = None,
    ):
        self.pipelines = pipelines
        self.aggregator = aggregator or (lambda x: x)

    def run(
        self,
        input_data: Any,
        context: dict[str, Any] | None = None,
    ) -> StageResult:
        """Run all pipelines and aggregate."""
        results = []
        outputs = []

        for pipeline in self.pipelines:
            result = pipeline.run(input_data, context)
            results.append(result)
            if result.is_success:
                outputs.append(result.output)

        # Check if any failed
        failed = [r for r in results if r.status == StageStatus.FAILED]
        if failed:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"{len(failed)} pipelines failed",
            )

        # Aggregate outputs
        aggregated = self.aggregator(outputs)

        return StageResult(
            status=StageStatus.SUCCESS,
            output=aggregated,
            metadata={"pipelines_run": len(self.pipelines)},
        )


# =============================================================================
# PIPELINE BUILDER
# =============================================================================


class PipelineBuilder:
    """
    Fluent builder for pipelines.

    Example:
        pipeline = (
            PipelineBuilder("preprocess")
            .transform(lambda x: x * 2)
            .filter(lambda x: x.max() > 0)
            .transform(np.clip, min=0, max=1)
            .build()
        )
    """

    def __init__(self, name: str = "pipeline"):
        self._pipeline = Pipeline(name)

    def transform(
        self,
        func: Callable,
        name: str = "",
        **kwargs,
    ) -> PipelineBuilder:
        """Add transform stage."""
        if kwargs:
            # Partial application
            transform = lambda x, **kw: func(x, **kwargs)
        else:
            transform = func
        self._pipeline.add_transform(transform, name)
        return self

    def filter(
        self,
        condition: Callable[[Any], bool],
        name: str = "",
    ) -> PipelineBuilder:
        """Add filter stage."""
        self._pipeline.add(FilterStage(condition, name))
        return self

    def function(
        self,
        func: Callable,
        name: str = "",
    ) -> PipelineBuilder:
        """Add function stage."""
        self._pipeline.add_function(func, name)
        return self

    def context(self, key: str, value: Any) -> PipelineBuilder:
        """Set context value."""
        self._pipeline.set_context(key, value)
        return self

    def build(self) -> Pipeline:
        """Build the pipeline."""
        return self._pipeline
