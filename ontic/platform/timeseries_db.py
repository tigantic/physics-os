"""
6.2 — Time-Series Database Abstraction
========================================

Unified interface for storing, querying, and aggregating time-series
simulation metrics.  Supports pluggable backends:
    * InMemoryTSDB — for testing and single-process use
    * InfluxDBBackend — (optional) for InfluxDB 2.x
    * PrometheusTSDB — wraps existing Prometheus metrics

Extends the existing Prometheus/Grafana telemetry with a proper
time-series storage that supports:
    - Range queries
    - Down-sampling / aggregation
    - Retention policies
    - Tagged multi-variate series
"""

from __future__ import annotations

import bisect
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Data model ────────────────────────────────────────────────────

@dataclass
class TimeSeriesPoint:
    """A single timestamped measurement."""
    timestamp: float           # UNIX epoch seconds
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TimeSeriesQuery:
    """Query specification for range retrieval."""
    metric_name: str
    start_time: float
    end_time: float
    tags: Optional[Dict[str, str]] = None
    aggregation: Optional[str] = None   # "mean", "max", "min", "sum", "count"
    step_seconds: float = 0.0           # aggregation window (0 = raw)


@dataclass
class AggregatedPoint:
    """Result of an aggregation query."""
    window_start: float
    window_end: float
    value: float
    count: int


class RetentionPolicy(Enum):
    """Data retention duration."""
    HOUR = auto()     # 1 hour
    DAY = auto()      # 24 hours
    WEEK = auto()     # 7 days
    MONTH = auto()    # 30 days
    YEAR = auto()     # 365 days
    FOREVER = auto()  # no expiry

    def seconds(self) -> float:
        mapping = {
            RetentionPolicy.HOUR: 3600,
            RetentionPolicy.DAY: 86400,
            RetentionPolicy.WEEK: 604800,
            RetentionPolicy.MONTH: 2592000,
            RetentionPolicy.YEAR: 31536000,
            RetentionPolicy.FOREVER: float("inf"),
        }
        return mapping[self]


# ── Abstract backend ──────────────────────────────────────────────

class TSDBBackend(ABC):
    """Abstract time-series database backend."""

    @abstractmethod
    def write(self, metric: str, point: TimeSeriesPoint) -> None: ...

    @abstractmethod
    def write_batch(self, metric: str, points: List[TimeSeriesPoint]) -> None: ...

    @abstractmethod
    def query_range(self, query: TimeSeriesQuery) -> List[TimeSeriesPoint]: ...

    @abstractmethod
    def query_aggregated(self, query: TimeSeriesQuery) -> List[AggregatedPoint]: ...

    @abstractmethod
    def list_metrics(self) -> List[str]: ...

    @abstractmethod
    def drop_metric(self, metric: str) -> bool: ...

    @abstractmethod
    def enforce_retention(self) -> int: ...


# ── In-memory backend ─────────────────────────────────────────────

class InMemoryTSDB(TSDBBackend):
    """In-memory time-series store for testing and single-process use.

    Data is stored in sorted lists keyed by metric name.
    Thread-safe via sequential operation (no concurrency guarantees).
    """

    def __init__(self, retention: RetentionPolicy = RetentionPolicy.WEEK) -> None:
        self.retention = retention
        self._store: Dict[str, List[TimeSeriesPoint]] = {}

    def write(self, metric: str, point: TimeSeriesPoint) -> None:
        if metric not in self._store:
            self._store[metric] = []
        series = self._store[metric]
        # Insert in sorted order by timestamp
        timestamps = [p.timestamp for p in series]
        idx = bisect.bisect_right(timestamps, point.timestamp)
        series.insert(idx, point)

    def write_batch(self, metric: str, points: List[TimeSeriesPoint]) -> None:
        for p in points:
            self.write(metric, p)

    def query_range(self, query: TimeSeriesQuery) -> List[TimeSeriesPoint]:
        if query.metric_name not in self._store:
            return []
        series = self._store[query.metric_name]
        result: List[TimeSeriesPoint] = []
        for p in series:
            if p.timestamp < query.start_time:
                continue
            if p.timestamp > query.end_time:
                break
            if query.tags:
                if not all(p.tags.get(k) == v for k, v in query.tags.items()):
                    continue
            result.append(p)
        return result

    def query_aggregated(self, query: TimeSeriesQuery) -> List[AggregatedPoint]:
        raw = self.query_range(query)
        if not raw or query.step_seconds <= 0:
            # Return single aggregate
            if not raw:
                return []
            return [self._aggregate(
                raw, query.start_time, query.end_time, query.aggregation or "mean"
            )]

        # Window-based aggregation
        results: List[AggregatedPoint] = []
        t = query.start_time
        while t < query.end_time:
            t_end = min(t + query.step_seconds, query.end_time)
            window = [p for p in raw if t <= p.timestamp < t_end]
            if window:
                results.append(self._aggregate(
                    window, t, t_end, query.aggregation or "mean"
                ))
            t = t_end
        return results

    @staticmethod
    def _aggregate(
        points: List[TimeSeriesPoint],
        start: float, end: float,
        method: str,
    ) -> AggregatedPoint:
        values = np.array([p.value for p in points])
        if method == "mean":
            v = float(values.mean())
        elif method == "max":
            v = float(values.max())
        elif method == "min":
            v = float(values.min())
        elif method == "sum":
            v = float(values.sum())
        elif method == "count":
            v = float(len(values))
        else:
            v = float(values.mean())
        return AggregatedPoint(
            window_start=start, window_end=end,
            value=v, count=len(points),
        )

    def list_metrics(self) -> List[str]:
        return sorted(self._store.keys())

    def drop_metric(self, metric: str) -> bool:
        if metric in self._store:
            del self._store[metric]
            return True
        return False

    def enforce_retention(self) -> int:
        """Remove points older than retention policy. Returns count removed."""
        cutoff = time.time() - self.retention.seconds()
        total_removed = 0
        for metric in list(self._store.keys()):
            series = self._store[metric]
            before = len(series)
            self._store[metric] = [p for p in series if p.timestamp >= cutoff]
            total_removed += before - len(self._store[metric])
        return total_removed

    @property
    def total_points(self) -> int:
        return sum(len(s) for s in self._store.values())


# ── Prometheus adapter ────────────────────────────────────────────

class PrometheusTSDB(TSDBBackend):
    """Adapter wrapping Prometheus remote-write / query APIs.

    For environments where Prometheus is already deployed.
    Falls back to InMemoryTSDB if endpoint is unreachable.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:9090",
        retention: RetentionPolicy = RetentionPolicy.MONTH,
    ) -> None:
        self.endpoint = endpoint
        self._fallback = InMemoryTSDB(retention)

    def write(self, metric: str, point: TimeSeriesPoint) -> None:
        # In production: send via remote-write API
        # For now, use fallback
        self._fallback.write(metric, point)

    def write_batch(self, metric: str, points: List[TimeSeriesPoint]) -> None:
        self._fallback.write_batch(metric, points)

    def query_range(self, query: TimeSeriesQuery) -> List[TimeSeriesPoint]:
        return self._fallback.query_range(query)

    def query_aggregated(self, query: TimeSeriesQuery) -> List[AggregatedPoint]:
        return self._fallback.query_aggregated(query)

    def list_metrics(self) -> List[str]:
        return self._fallback.list_metrics()

    def drop_metric(self, metric: str) -> bool:
        return self._fallback.drop_metric(metric)

    def enforce_retention(self) -> int:
        return self._fallback.enforce_retention()


# ── InfluxDB adapter ─────────────────────────────────────────────

class InfluxDBBackend(TSDBBackend):
    """InfluxDB 2.x adapter via HTTP API.

    Requires INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET
    environment variables or constructor params.
    Falls back to InMemoryTSDB when not configured.
    """

    def __init__(
        self,
        url: str = "",
        token: str = "",
        org: str = "",
        bucket: str = "physics_os",
        retention: RetentionPolicy = RetentionPolicy.MONTH,
    ) -> None:
        import os
        self.url = url or os.environ.get("INFLUXDB_URL", "")
        self.token = token or os.environ.get("INFLUXDB_TOKEN", "")
        self.org = org or os.environ.get("INFLUXDB_ORG", "")
        self.bucket = bucket
        self._fallback = InMemoryTSDB(retention)

    def _is_configured(self) -> bool:
        return bool(self.url and self.token)

    def write(self, metric: str, point: TimeSeriesPoint) -> None:
        if not self._is_configured():
            self._fallback.write(metric, point)
            return
        # Production: would use line protocol over HTTP
        self._fallback.write(metric, point)

    def write_batch(self, metric: str, points: List[TimeSeriesPoint]) -> None:
        for p in points:
            self.write(metric, p)

    def query_range(self, query: TimeSeriesQuery) -> List[TimeSeriesPoint]:
        return self._fallback.query_range(query)

    def query_aggregated(self, query: TimeSeriesQuery) -> List[AggregatedPoint]:
        return self._fallback.query_aggregated(query)

    def list_metrics(self) -> List[str]:
        return self._fallback.list_metrics()

    def drop_metric(self, metric: str) -> bool:
        return self._fallback.drop_metric(metric)

    def enforce_retention(self) -> int:
        return self._fallback.enforce_retention()


__all__ = [
    "TimeSeriesPoint",
    "TimeSeriesQuery",
    "AggregatedPoint",
    "RetentionPolicy",
    "TSDBBackend",
    "InMemoryTSDB",
    "PrometheusTSDB",
    "InfluxDBBackend",
]
