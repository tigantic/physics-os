"""
Observability Stack for Production Systems

Structured logging, metrics collection, health checks, and distributed tracing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum
from functools import wraps
from contextlib import contextmanager
import time
import threading
import logging
import json
import traceback
import uuid
import os
from datetime import datetime, timezone
from collections import defaultdict
import sys

# Thread-local storage for trace context
_trace_context = threading.local()


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogContext:
    """Contextual information for log entries."""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    service: str = "tensornet-discovery"
    version: str = "1.8.0"
    environment: str = "production"
    extra: Dict[str, Any] = field(default_factory=dict)


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""
    
    def __init__(self, context: Optional[LogContext] = None):
        super().__init__()
        self.context = context or LogContext()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.context.service,
            "version": self.context.version,
            "environment": self.context.environment,
        }
        
        # Add trace context
        if hasattr(_trace_context, 'trace_id'):
            log_entry["trace_id"] = _trace_context.trace_id
        if hasattr(_trace_context, 'span_id'):
            log_entry["span_id"] = _trace_context.span_id
        
        # Add request context
        if self.context.request_id:
            log_entry["request_id"] = self.context.request_id
        if self.context.user_id:
            log_entry["user_id"] = self.context.user_id
        
        # Add source location
        log_entry["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": traceback.format_exception(*record.exc_info),
            }
        
        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            log_entry["extra"] = record.extra_fields
        
        # Add context extra
        if self.context.extra:
            log_entry.setdefault("extra", {}).update(self.context.extra)
        
        return json.dumps(log_entry, default=str)


class StructuredLogger:
    """
    Structured logger with JSON output.
    
    Features:
    - JSON-formatted logs
    - Automatic trace context injection
    - Field enrichment
    - Multiple output handlers
    """
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        context: Optional[LogContext] = None,
        json_output: bool = True,
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            level: Minimum log level
            context: Contextual information
            json_output: Whether to output JSON (vs. plain text)
        """
        self.name = name
        self.context = context or LogContext()
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.value))
        
        # Remove existing handlers
        self._logger.handlers = []
        
        # Add handler
        handler = logging.StreamHandler(sys.stdout)
        if json_output:
            handler.setFormatter(StructuredFormatter(self.context))
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        self._logger.addHandler(handler)
        self._logger.propagate = False
    
    def _log(self, level: int, message: str, **kwargs) -> None:
        """Internal log method."""
        record = self._logger.makeRecord(
            self.name,
            level,
            "(unknown)",
            0,
            message,
            args=(),
            exc_info=kwargs.pop('exc_info', None),
        )
        if kwargs:
            record.extra_fields = kwargs
        self._logger.handle(record)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message."""
        self._log(
            logging.ERROR,
            message,
            exc_info=sys.exc_info() if exc_info else None,
            **kwargs
        )
    
    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log critical message."""
        self._log(
            logging.CRITICAL,
            message,
            exc_info=sys.exc_info() if exc_info else None,
            **kwargs
        )
    
    def with_context(self, **kwargs) -> "StructuredLogger":
        """Create a new logger with additional context."""
        new_context = LogContext(
            trace_id=self.context.trace_id,
            span_id=self.context.span_id,
            user_id=self.context.user_id,
            request_id=self.context.request_id,
            service=self.context.service,
            version=self.context.version,
            environment=self.context.environment,
            extra={**self.context.extra, **kwargs},
        )
        return StructuredLogger(
            self.name,
            context=new_context,
            json_output=isinstance(
                self._logger.handlers[0].formatter,
                StructuredFormatter
            ),
        )


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str = "tensornet") -> StructuredLogger:
    """Get or create a structured logger."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Single metric data."""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    description: str = ""
    unit: str = ""


@dataclass
class HistogramBuckets:
    """Histogram bucket configuration."""
    boundaries: List[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])
    counts: List[int] = field(default_factory=list)
    sum: float = 0.0
    count: int = 0
    
    def __post_init__(self):
        if not self.counts:
            self.counts = [0] * (len(self.boundaries) + 1)
    
    def observe(self, value: float) -> None:
        """Observe a value."""
        self.sum += value
        self.count += 1
        
        for i, boundary in enumerate(self.boundaries):
            if value <= boundary:
                self.counts[i] += 1
                return
        
        self.counts[-1] += 1  # +Inf bucket


class MetricsCollector:
    """
    Metrics collection and aggregation.
    
    Supports:
    - Counters (monotonically increasing)
    - Gauges (point-in-time values)
    - Histograms (distribution of values)
    - Labels for dimensionality
    """
    
    def __init__(self, namespace: str = "tensornet"):
        """
        Initialize metrics collector.
        
        Args:
            namespace: Metric namespace prefix
        """
        self.namespace = namespace
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Dict[str, HistogramBuckets]] = defaultdict(dict)
        self._descriptions: Dict[str, str] = {}
        self._lock = threading.Lock()
    
    def _labels_key(self, labels: Dict[str, str]) -> str:
        """Create a hashable key from labels."""
        return json.dumps(sorted(labels.items()), separators=(',', ':'))
    
    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> float:
        """
        Increment a counter.
        
        Args:
            name: Metric name
            value: Value to add (must be positive)
            labels: Metric labels
            description: Metric description
            
        Returns:
            New counter value
        """
        if value < 0:
            raise ValueError("Counter value must be non-negative")
        
        full_name = f"{self.namespace}_{name}"
        labels = labels or {}
        key = self._labels_key(labels)
        
        with self._lock:
            self._counters[full_name][key] += value
            if description:
                self._descriptions[full_name] = description
            return self._counters[full_name][key]
    
    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> float:
        """
        Set a gauge value.
        
        Args:
            name: Metric name
            value: Value to set
            labels: Metric labels
            description: Metric description
            
        Returns:
            Set value
        """
        full_name = f"{self.namespace}_{name}"
        labels = labels or {}
        key = self._labels_key(labels)
        
        with self._lock:
            self._gauges[full_name][key] = value
            if description:
                self._descriptions[full_name] = description
            return value
    
    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
        description: str = "",
    ) -> None:
        """
        Observe a histogram value.
        
        Args:
            name: Metric name
            value: Value to observe
            labels: Metric labels
            buckets: Histogram bucket boundaries
            description: Metric description
        """
        full_name = f"{self.namespace}_{name}"
        labels = labels or {}
        key = self._labels_key(labels)
        
        with self._lock:
            if key not in self._histograms[full_name]:
                self._histograms[full_name][key] = HistogramBuckets(
                    boundaries=buckets or HistogramBuckets().boundaries
                )
            
            self._histograms[full_name][key].observe(value)
            
            if description:
                self._descriptions[full_name] = description
    
    @contextmanager
    def timer(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ):
        """
        Context manager for timing operations.
        
        Args:
            name: Histogram metric name
            labels: Metric labels
            description: Metric description
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.histogram(name, duration, labels, description=description)
    
    def get_metrics(self) -> List[Metric]:
        """Get all metrics."""
        metrics = []
        
        with self._lock:
            # Counters
            for name, label_values in self._counters.items():
                for key, value in label_values.items():
                    labels = dict(json.loads(key)) if key else {}
                    metrics.append(Metric(
                        name=name,
                        type=MetricType.COUNTER,
                        value=value,
                        labels=labels,
                        description=self._descriptions.get(name, ""),
                    ))
            
            # Gauges
            for name, label_values in self._gauges.items():
                for key, value in label_values.items():
                    labels = dict(json.loads(key)) if key else {}
                    metrics.append(Metric(
                        name=name,
                        type=MetricType.GAUGE,
                        value=value,
                        labels=labels,
                        description=self._descriptions.get(name, ""),
                    ))
            
            # Histograms
            for name, label_values in self._histograms.items():
                for key, hist in label_values.items():
                    labels = dict(json.loads(key)) if key else {}
                    
                    # Sum metric
                    metrics.append(Metric(
                        name=f"{name}_sum",
                        type=MetricType.HISTOGRAM,
                        value=hist.sum,
                        labels=labels,
                        description=self._descriptions.get(name, ""),
                    ))
                    
                    # Count metric
                    metrics.append(Metric(
                        name=f"{name}_count",
                        type=MetricType.HISTOGRAM,
                        value=hist.count,
                        labels=labels,
                    ))
                    
                    # Bucket metrics
                    cumulative = 0
                    for i, boundary in enumerate(hist.boundaries):
                        cumulative += hist.counts[i]
                        bucket_labels = {**labels, "le": str(boundary)}
                        metrics.append(Metric(
                            name=f"{name}_bucket",
                            type=MetricType.HISTOGRAM,
                            value=cumulative,
                            labels=bucket_labels,
                        ))
                    
                    # +Inf bucket
                    cumulative += hist.counts[-1]
                    bucket_labels = {**labels, "le": "+Inf"}
                    metrics.append(Metric(
                        name=f"{name}_bucket",
                        type=MetricType.HISTOGRAM,
                        value=cumulative,
                        labels=bucket_labels,
                    ))
        
        return metrics
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        metrics = self.get_metrics()
        
        # Group by name
        by_name: Dict[str, List[Metric]] = defaultdict(list)
        for metric in metrics:
            by_name[metric.name].append(metric)
        
        for name, group in by_name.items():
            # Add HELP and TYPE
            if group[0].description:
                lines.append(f"# HELP {name} {group[0].description}")
            
            type_str = group[0].type.value
            if type_str == "histogram":
                type_str = "histogram" if "_bucket" in name or "_sum" in name or "_count" in name else "gauge"
            lines.append(f"# TYPE {name} {type_str}")
            
            # Add metrics
            for metric in group:
                label_str = ""
                if metric.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in metric.labels.items()]
                    label_str = "{" + ",".join(label_pairs) + "}"
                
                lines.append(f"{metric.name}{label_str} {metric.value}")
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# Global metrics registry
_metrics: Optional[MetricsCollector] = None


def get_metrics(namespace: str = "tensornet") -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector(namespace)
    return _metrics


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status for a component."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    last_check: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health."""
    status: HealthStatus
    components: List[ComponentHealth]
    version: str = "1.8.0"
    uptime_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "timestamp": self.timestamp,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                    "last_check": c.last_check,
                    "details": c.details,
                }
                for c in self.components
            ],
        }


class HealthChecker:
    """
    Health check orchestrator.
    
    Manages health checks for system components.
    """
    
    def __init__(self, service_name: str = "tensornet-discovery"):
        """
        Initialize health checker.
        
        Args:
            service_name: Service name for reporting
        """
        self.service_name = service_name
        self._checks: Dict[str, Callable[[], ComponentHealth]] = {}
        self._start_time = time.time()
        self._lock = threading.Lock()
    
    def register_check(
        self,
        name: str,
        check_fn: Callable[[], ComponentHealth],
    ) -> None:
        """
        Register a health check.
        
        Args:
            name: Component name
            check_fn: Function that returns ComponentHealth
        """
        with self._lock:
            self._checks[name] = check_fn
    
    def unregister_check(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)
    
    def check(self, component: Optional[str] = None) -> SystemHealth:
        """
        Run health checks.
        
        Args:
            component: Specific component to check, or all if None
            
        Returns:
            System health status
        """
        components = []
        
        with self._lock:
            checks = self._checks.copy()
        
        for name, check_fn in checks.items():
            if component and name != component:
                continue
            
            start = time.time()
            try:
                health = check_fn()
                health.latency_ms = (time.time() - start) * 1000
                health.last_check = datetime.now(timezone.utc).isoformat()
            except Exception as e:
                health = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                    latency_ms=(time.time() - start) * 1000,
                    last_check=datetime.now(timezone.utc).isoformat(),
                )
            
            components.append(health)
        
        # Calculate overall status
        if not components:
            overall = HealthStatus.HEALTHY
        elif any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in components):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        return SystemHealth(
            status=overall,
            components=components,
            uptime_seconds=time.time() - self._start_time,
        )
    
    def liveness(self) -> bool:
        """Check if service is alive."""
        return True  # If we can execute, we're alive
    
    def readiness(self) -> bool:
        """Check if service is ready to accept requests."""
        health = self.check()
        return health.status != HealthStatus.UNHEALTHY


@dataclass
class Span:
    """A span in a distributed trace."""
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    
    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0
    
    def set_tag(self, key: str, value: str) -> "Span":
        """Set a tag on the span."""
        self.tags[key] = value
        return self
    
    def log(self, event: str, **fields) -> "Span":
        """Add a log entry to the span."""
        self.logs.append({
            "timestamp": time.time(),
            "event": event,
            **fields,
        })
        return self
    
    def finish(self, status: str = "OK") -> None:
        """Finish the span."""
        self.end_time = time.time()
        self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentId": self.parent_id,
            "operationName": self.operation_name,
            "serviceName": self.service_name,
            "startTime": self.start_time,
            "endTime": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
        }


class Tracer:
    """
    Distributed tracing implementation.
    
    Creates and manages spans for distributed tracing.
    """
    
    def __init__(self, service_name: str = "tensornet-discovery"):
        """
        Initialize tracer.
        
        Args:
            service_name: Service name for spans
        """
        self.service_name = service_name
        self._spans: List[Span] = []
        self._lock = threading.Lock()
    
    def _generate_id(self) -> str:
        """Generate a random ID."""
        return uuid.uuid4().hex[:16]
    
    def start_span(
        self,
        operation_name: str,
        parent: Optional[Span] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Span:
        """
        Start a new span.
        
        Args:
            operation_name: Operation being traced
            parent: Parent span for context propagation
            tags: Initial tags
            
        Returns:
            New span
        """
        if parent:
            trace_id = parent.trace_id
            parent_id = parent.span_id
        else:
            trace_id = self._generate_id()
            parent_id = None
        
        span = Span(
            trace_id=trace_id,
            span_id=self._generate_id(),
            parent_id=parent_id,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=time.time(),
            tags=tags or {},
        )
        
        # Set thread-local context
        _trace_context.trace_id = span.trace_id
        _trace_context.span_id = span.span_id
        
        return span
    
    @contextmanager
    def trace(
        self,
        operation_name: str,
        parent: Optional[Span] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Context manager for tracing.
        
        Args:
            operation_name: Operation being traced
            parent: Parent span
            tags: Initial tags
        """
        span = self.start_span(operation_name, parent, tags)
        try:
            yield span
            span.finish("OK")
        except Exception as e:
            span.set_tag("error", "true")
            span.set_tag("error.message", str(e))
            span.log("error", message=str(e), type=type(e).__name__)
            span.finish("ERROR")
            raise
        finally:
            with self._lock:
                self._spans.append(span)
    
    def get_active_trace_id(self) -> Optional[str]:
        """Get the active trace ID from context."""
        return getattr(_trace_context, 'trace_id', None)
    
    def get_active_span_id(self) -> Optional[str]:
        """Get the active span ID from context."""
        return getattr(_trace_context, 'span_id', None)
    
    def get_spans(self, trace_id: Optional[str] = None) -> List[Span]:
        """Get recorded spans, optionally filtered by trace ID."""
        with self._lock:
            if trace_id:
                return [s for s in self._spans if s.trace_id == trace_id]
            return list(self._spans)
    
    def export_spans(self, format: str = "json") -> str:
        """Export spans to string format."""
        spans = self.get_spans()
        
        if format == "json":
            return json.dumps([s.to_dict() for s in spans], default=str, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def clear(self) -> None:
        """Clear recorded spans."""
        with self._lock:
            self._spans.clear()


def traced(operation_name: Optional[str] = None):
    """
    Decorator for tracing function execution.
    
    Args:
        operation_name: Operation name (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__
        
        tracer = Tracer()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.trace(operation_name):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator
