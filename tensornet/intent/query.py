"""
FieldQuery DSL
===============

Domain-specific language for field queries.

Provides a fluent, composable API for expressing
queries over simulation fields.

Example:
    # Find maximum velocity in inlet region
    query = (
        FieldQuery("velocity")
        .where(region="inlet")
        .aggregate("max")
    )

    # Complex query with multiple conditions
    query = (
        FieldQuery("pressure")
        .where(x_range=(0, 0.5))
        .where(lambda f: f > 1.0)
        .select("gradient")
        .aggregate("mean")
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# =============================================================================
# PREDICATES
# =============================================================================


class PredicateOp(Enum):
    """Predicate operation type."""

    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    LT = "lt"  # Less than
    LE = "le"  # Less or equal
    GT = "gt"  # Greater than
    GE = "ge"  # Greater or equal
    IN_RANGE = "in_range"
    IN_SET = "in_set"
    MATCHES = "matches"
    CUSTOM = "custom"


@dataclass
class Predicate:
    """
    Condition for filtering field values.

    Can represent:
    - Value comparisons (field > threshold)
    - Range checks (x in [0, 1])
    - Region membership
    - Custom callables
    """

    op: PredicateOp

    # For comparison ops
    field_name: str | None = None
    value: Any = None

    # For range ops
    low: float | None = None
    high: float | None = None

    # For custom ops
    func: Callable | None = None

    # Metadata
    name: str = ""
    description: str = ""

    def evaluate(
        self,
        field_data: np.ndarray,
        coordinates: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Evaluate predicate on field data.

        Returns:
            Boolean mask where predicate is True
        """
        if self.op == PredicateOp.EQ:
            return np.isclose(field_data, self.value)

        elif self.op == PredicateOp.NE:
            return ~np.isclose(field_data, self.value)

        elif self.op == PredicateOp.LT:
            return field_data < self.value

        elif self.op == PredicateOp.LE:
            return field_data <= self.value

        elif self.op == PredicateOp.GT:
            return field_data > self.value

        elif self.op == PredicateOp.GE:
            return field_data >= self.value

        elif self.op == PredicateOp.IN_RANGE:
            return (field_data >= self.low) & (field_data <= self.high)

        elif self.op == PredicateOp.CUSTOM:
            if self.func is not None:
                return self.func(field_data)
            return np.ones_like(field_data, dtype=bool)

        else:
            return np.ones_like(field_data, dtype=bool)

    def __and__(self, other: Predicate) -> CompoundPredicate:
        return CompoundPredicate([self, other], "and")

    def __or__(self, other: Predicate) -> CompoundPredicate:
        return CompoundPredicate([self, other], "or")

    def __invert__(self) -> Predicate:
        return NotPredicate(self)

    @classmethod
    def gt(cls, value: float, name: str = "") -> Predicate:
        """Greater than predicate."""
        return cls(op=PredicateOp.GT, value=value, name=name)

    @classmethod
    def lt(cls, value: float, name: str = "") -> Predicate:
        """Less than predicate."""
        return cls(op=PredicateOp.LT, value=value, name=name)

    @classmethod
    def in_range(cls, low: float, high: float, name: str = "") -> Predicate:
        """In range predicate."""
        return cls(op=PredicateOp.IN_RANGE, low=low, high=high, name=name)

    @classmethod
    def custom(cls, func: Callable, name: str = "") -> Predicate:
        """Custom function predicate."""
        return cls(op=PredicateOp.CUSTOM, func=func, name=name)


@dataclass
class CompoundPredicate(Predicate):
    """Combination of multiple predicates."""

    predicates: list[Predicate] = field(default_factory=list)
    combiner: str = "and"  # "and" or "or"

    def __post_init__(self):
        self.op = PredicateOp.CUSTOM

    def evaluate(
        self,
        field_data: np.ndarray,
        coordinates: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if not self.predicates:
            return np.ones_like(field_data, dtype=bool)

        masks = [p.evaluate(field_data, coordinates) for p in self.predicates]

        if self.combiner == "and":
            result = masks[0]
            for m in masks[1:]:
                result = result & m
            return result
        else:  # "or"
            result = masks[0]
            for m in masks[1:]:
                result = result | m
            return result


@dataclass
class NotPredicate(Predicate):
    """Negated predicate."""

    inner: Predicate = None

    def __post_init__(self):
        self.op = PredicateOp.CUSTOM

    def evaluate(
        self,
        field_data: np.ndarray,
        coordinates: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if self.inner is None:
            return np.ones_like(field_data, dtype=bool)
        return ~self.inner.evaluate(field_data, coordinates)


# =============================================================================
# SELECTORS
# =============================================================================


class SelectorType(Enum):
    """Type of selector operation."""

    IDENTITY = "identity"  # Return field as-is
    GRADIENT = "gradient"  # Compute gradient
    MAGNITUDE = "magnitude"  # Vector magnitude
    COMPONENT = "component"  # Single component
    SLICE = "slice"  # Slice along axis
    DOWNSAMPLE = "downsample"  # Reduce resolution
    DERIVATIVE = "derivative"  # Time derivative
    CUSTOM = "custom"


@dataclass
class Selector:
    """
    Operation to select/transform field data.
    """

    selector_type: SelectorType = SelectorType.IDENTITY

    # For component selection
    component: int | None = None
    axis: int | None = None

    # For slicing
    slice_idx: int | None = None

    # For downsampling
    factor: int = 2

    # Custom function
    func: Callable | None = None

    def apply(self, field_data: np.ndarray) -> np.ndarray:
        """Apply selector to field data."""
        if self.selector_type == SelectorType.IDENTITY:
            return field_data

        elif self.selector_type == SelectorType.GRADIENT:
            grad = np.gradient(field_data, axis=self.axis)
            # np.gradient returns tuple/list of arrays when axis is None
            if isinstance(grad, (list, tuple)):
                # Return magnitude of gradient
                return np.sqrt(sum(g**2 for g in grad))
            return grad

        elif self.selector_type == SelectorType.MAGNITUDE:
            # Assume last axis is vector components
            return np.linalg.norm(field_data, axis=-1)

        elif self.selector_type == SelectorType.COMPONENT:
            if self.component is not None:
                return field_data[..., self.component]
            return field_data

        elif self.selector_type == SelectorType.SLICE:
            if self.axis is not None and self.slice_idx is not None:
                slices = [slice(None)] * field_data.ndim
                slices[self.axis] = self.slice_idx
                return field_data[tuple(slices)]
            return field_data

        elif self.selector_type == SelectorType.DOWNSAMPLE:
            slices = [slice(None, None, self.factor)] * field_data.ndim
            return field_data[tuple(slices)]

        elif self.selector_type == SelectorType.CUSTOM:
            if self.func is not None:
                return self.func(field_data)
            return field_data

        return field_data

    @classmethod
    def gradient(cls, axis: int | None = None) -> Selector:
        return cls(selector_type=SelectorType.GRADIENT, axis=axis)

    @classmethod
    def magnitude(cls) -> Selector:
        return cls(selector_type=SelectorType.MAGNITUDE)

    @classmethod
    def component(cls, idx: int) -> Selector:
        return cls(selector_type=SelectorType.COMPONENT, component=idx)


# =============================================================================
# AGGREGATORS
# =============================================================================


class AggregatorType(Enum):
    """Type of aggregation."""

    NONE = "none"
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    STD = "std"
    ARGMIN = "argmin"
    ARGMAX = "argmax"
    COUNT = "count"
    INTEGRAL = "integral"
    CUSTOM = "custom"


# =============================================================================
# QUERY RESULT
# =============================================================================


@dataclass
class QueryResult:
    """
    Result of executing a FieldQuery.
    """

    value: Any = None
    filtered_data: np.ndarray | None = None
    mask: np.ndarray | None = None

    # Query info
    field_name: str = ""
    query_type: str = ""

    # Statistics
    count: int = 0
    execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": (
                self.value
                if not isinstance(self.value, np.ndarray)
                else self.value.tolist()
            ),
            "field_name": self.field_name,
            "query_type": self.query_type,
            "count": self.count,
        }


@dataclass
class Aggregator:
    """
    Aggregation operation over field values.
    """

    agg_type: AggregatorType = AggregatorType.NONE
    axis: int | None = None  # None = all axes
    weights: np.ndarray | None = None
    func: Callable | None = None

    def apply(
        self,
        values: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Apply aggregation to values."""
        # Apply mask if provided
        if mask is not None:
            if self.axis is not None:
                # Masked reduction along axis is complex
                values = np.where(mask, values, np.nan)
            else:
                values = values[mask]

        if self.agg_type == AggregatorType.NONE:
            return values

        elif self.agg_type == AggregatorType.SUM:
            if self.weights is not None:
                return np.nansum(values * self.weights, axis=self.axis)
            return np.nansum(values, axis=self.axis)

        elif self.agg_type == AggregatorType.MEAN:
            if self.weights is not None:
                return np.nansum(values * self.weights, axis=self.axis) / np.nansum(
                    self.weights
                )
            return np.nanmean(values, axis=self.axis)

        elif self.agg_type == AggregatorType.MIN:
            return np.nanmin(values, axis=self.axis)

        elif self.agg_type == AggregatorType.MAX:
            return np.nanmax(values, axis=self.axis)

        elif self.agg_type == AggregatorType.MEDIAN:
            return np.nanmedian(values, axis=self.axis)

        elif self.agg_type == AggregatorType.STD:
            return np.nanstd(values, axis=self.axis)

        elif self.agg_type == AggregatorType.ARGMIN:
            return np.unravel_index(np.nanargmin(values), values.shape)

        elif self.agg_type == AggregatorType.ARGMAX:
            return np.unravel_index(np.nanargmax(values), values.shape)

        elif self.agg_type == AggregatorType.COUNT:
            if mask is not None:
                return np.sum(mask)
            return values.size

        elif self.agg_type == AggregatorType.CUSTOM:
            if self.func is not None:
                return self.func(values)
            return values

        return values

    @classmethod
    def sum(cls, axis: int | None = None) -> Aggregator:
        return cls(agg_type=AggregatorType.SUM, axis=axis)

    @classmethod
    def mean(cls, axis: int | None = None) -> Aggregator:
        return cls(agg_type=AggregatorType.MEAN, axis=axis)

    @classmethod
    def max(cls, axis: int | None = None) -> Aggregator:
        return cls(agg_type=AggregatorType.MAX, axis=axis)

    @classmethod
    def min(cls, axis: int | None = None) -> Aggregator:
        return cls(agg_type=AggregatorType.MIN, axis=axis)


# =============================================================================
# FIELD QUERY
# =============================================================================


@dataclass
class FieldQuery:
    """
    Composable query over simulation fields.

    Supports fluent interface for building queries:

    Example:
        query = (
            FieldQuery("velocity")
            .where(Predicate.gt(0.1))
            .select(Selector.magnitude())
            .aggregate(Aggregator.max())
        )

        result = query.execute(field_data)
    """

    field_name: str = ""

    # Query components
    predicates: list[Predicate] = field(default_factory=list)
    selectors: list[Selector] = field(default_factory=list)
    aggregator: Aggregator | None = None

    # Execution options
    use_mask: bool = True
    return_indices: bool = False

    def __init__(self, field_name: str = ""):
        self.field_name = field_name
        self.predicates = []
        self.selectors = []
        self.aggregator = None
        self.use_mask = True
        self.return_indices = False

    def where(
        self,
        predicate: Predicate | Callable | None = None,
        **kwargs,
    ) -> FieldQuery:
        """
        Add filter predicate.

        Args:
            predicate: Predicate object or callable
            **kwargs: Field comparisons (e.g., x_min=0.5)
        """
        query = self._copy()

        if predicate is not None:
            if callable(predicate) and not isinstance(predicate, Predicate):
                query.predicates.append(Predicate.custom(predicate))
            else:
                query.predicates.append(predicate)

        # Handle kwargs
        for key, value in kwargs.items():
            if key.endswith("_min"):
                field = key[:-4]
                query.predicates.append(
                    Predicate(
                        op=PredicateOp.GE,
                        field_name=field,
                        value=value,
                    )
                )
            elif key.endswith("_max"):
                field = key[:-4]
                query.predicates.append(
                    Predicate(
                        op=PredicateOp.LE,
                        field_name=field,
                        value=value,
                    )
                )
            elif key.endswith("_range") and isinstance(value, (list, tuple)):
                field = key[:-6]
                query.predicates.append(
                    Predicate(
                        op=PredicateOp.IN_RANGE,
                        field_name=field,
                        low=value[0],
                        high=value[1],
                    )
                )
            else:
                query.predicates.append(
                    Predicate(
                        op=PredicateOp.EQ,
                        field_name=key,
                        value=value,
                    )
                )

        return query

    def select(
        self,
        selector: Selector | str | None = None,
        **kwargs,
    ) -> FieldQuery:
        """
        Add selector/transform.

        Args:
            selector: Selector object or string name
        """
        query = self._copy()

        if selector is not None:
            if isinstance(selector, str):
                selector_map = {
                    "gradient": Selector.gradient(),
                    "magnitude": Selector.magnitude(),
                    "x": Selector.component(0),
                    "y": Selector.component(1),
                    "z": Selector.component(2),
                }
                sel = selector_map.get(selector, Selector())
                query.selectors.append(sel)
            else:
                query.selectors.append(selector)

        return query

    def aggregate(
        self,
        aggregator: Aggregator | str | None = None,
    ) -> FieldQuery:
        """
        Set aggregation.

        Args:
            aggregator: Aggregator object or string name
        """
        query = self._copy()

        if aggregator is not None:
            if isinstance(aggregator, str):
                agg_map = {
                    "sum": Aggregator.sum(),
                    "mean": Aggregator.mean(),
                    "max": Aggregator.max(),
                    "min": Aggregator.min(),
                    "median": Aggregator(agg_type=AggregatorType.MEDIAN),
                    "std": Aggregator(agg_type=AggregatorType.STD),
                    "count": Aggregator(agg_type=AggregatorType.COUNT),
                }
                query.aggregator = agg_map.get(aggregator, Aggregator())
            else:
                query.aggregator = aggregator

        return query

    def maximize(self, field: str | None = None) -> FieldQuery:
        """Shortcut for max aggregation."""
        query = self._copy()
        if field:
            query.field_name = field
        query.aggregator = Aggregator.max()
        return query

    def minimize(self, field: str | None = None) -> FieldQuery:
        """Shortcut for min aggregation."""
        query = self._copy()
        if field:
            query.field_name = field
        query.aggregator = Aggregator.min()
        return query

    def average(self, field: str | None = None) -> FieldQuery:
        """Shortcut for mean aggregation."""
        query = self._copy()
        if field:
            query.field_name = field
        query.aggregator = Aggregator.mean()
        return query

    def execute(
        self,
        field_data: np.ndarray,
        coordinates: dict[str, np.ndarray] | None = None,
    ) -> QueryResult:
        """
        Execute query on field data.

        Args:
            field_data: Field array
            coordinates: Optional coordinate arrays

        Returns:
            QueryResult with value and metadata
        """
        import time

        start = time.time()

        result = field_data.copy()

        # Apply selectors
        for selector in self.selectors:
            result = selector.apply(result)

        # Build mask from predicates
        mask = None
        filtered_data = None
        if self.predicates and self.use_mask:
            mask = np.ones(result.shape, dtype=bool)
            for pred in self.predicates:
                mask = mask & pred.evaluate(result, coordinates)
            filtered_data = result[mask]

        # Apply aggregation
        if self.aggregator is not None:
            value = self.aggregator.apply(result, mask)
        elif mask is not None:
            value = result[mask]
        else:
            value = result

        return QueryResult(
            value=value,
            filtered_data=filtered_data,
            mask=mask,
            field_name=self.field_name,
            query_type=self.aggregator.agg_type.value if self.aggregator else "raw",
            count=np.sum(mask) if mask is not None else result.size,
            execution_time=time.time() - start,
        )

    def _copy(self) -> FieldQuery:
        """Create a copy for fluent interface."""
        query = FieldQuery(self.field_name)
        query.predicates = list(self.predicates)
        query.selectors = list(self.selectors)
        query.aggregator = self.aggregator
        query.use_mask = self.use_mask
        query.return_indices = self.return_indices
        return query

    def to_dict(self) -> dict[str, Any]:
        """Serialize query to dictionary."""
        return {
            "field_name": self.field_name,
            "predicates": len(self.predicates),
            "selectors": len(self.selectors),
            "aggregator": self.aggregator.agg_type.value if self.aggregator else None,
        }


# =============================================================================
# QUERY BUILDER
# =============================================================================


class QueryBuilder:
    """
    Factory for building common query patterns.
    """

    @staticmethod
    def max_in_region(
        field: str,
        region: str,
        region_mask: np.ndarray | None = None,
    ) -> FieldQuery:
        """Query for maximum value in a region."""
        query = FieldQuery(field)
        if region_mask is not None:
            query = query.where(Predicate.custom(lambda f: region_mask))
        return query.maximize()

    @staticmethod
    def mean_gradient(field: str) -> FieldQuery:
        """Query for mean gradient magnitude."""
        return FieldQuery(field).select("gradient").select("magnitude").average()

    @staticmethod
    def count_above_threshold(field: str, threshold: float) -> FieldQuery:
        """Count elements above threshold."""
        return FieldQuery(field).where(Predicate.gt(threshold)).aggregate("count")

    @staticmethod
    def percentile(field: str, p: float) -> FieldQuery:
        """Get percentile value."""
        return FieldQuery(field).aggregate(
            Aggregator(
                agg_type=AggregatorType.CUSTOM,
                func=lambda x: np.percentile(x, p),
            )
        )
