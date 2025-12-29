"""
Monitoring and Logging for Project HyperTensor.

Provides:
- Structured logging with multiple outputs
- Metrics collection and aggregation
- Telemetry for performance tracking
- Alerting for anomaly detection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum, auto
from collections import deque
import time
import json
import sys
import threading
from datetime import datetime


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogEntry:
    """
    Structured log entry.
    
    Attributes:
        timestamp: When the event occurred
        level: Log level
        message: Log message
        logger_name: Name of the logger
        context: Additional context data
        source: Source file/module
        line: Source line number
    """
    timestamp: float
    level: LogLevel
    message: str
    logger_name: str = "hypertensor"
    context: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    line: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'level': self.level.name,
            'message': self.message,
            'logger': self.logger_name,
            'context': self.context,
            'source': self.source,
            'line': self.line,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class LogFormatter:
    """
    Format log entries for output.
    """
    
    SIMPLE = "{datetime} [{level}] {message}"
    DETAILED = "{datetime} [{level}] {logger} ({source}:{line}) - {message}"
    JSON = "json"
    
    def __init__(self, format: str = SIMPLE):
        """
        Initialize formatter.
        
        Args:
            format: Format string or "json"
        """
        self.format = format
    
    def format(self, entry: LogEntry) -> str:
        """Format a log entry."""
        if self.format == "json":
            return entry.to_json()
        
        return self.format.format(
            datetime=datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            level=entry.level.name,
            message=entry.message,
            logger=entry.logger_name,
            source=entry.source or "unknown",
            line=entry.line or 0,
        )


class StructuredLogger:
    """
    Structured logger with multiple handlers.
    """
    
    _loggers: Dict[str, 'StructuredLogger'] = {}
    
    def __init__(
        self,
        name: str = "hypertensor",
        level: LogLevel = LogLevel.INFO,
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Minimum log level
        """
        self.name = name
        self.level = level
        self.handlers: List[Callable[[LogEntry], None]] = []
        self._buffer: deque = deque(maxlen=1000)
        
        # Default console handler
        self.add_handler(self._console_handler)
    
    @classmethod
    def get_logger(cls, name: str = "hypertensor") -> 'StructuredLogger':
        """Get or create a logger by name."""
        if name not in cls._loggers:
            cls._loggers[name] = cls(name)
        return cls._loggers[name]
    
    def add_handler(self, handler: Callable[[LogEntry], None]):
        """Add a log handler."""
        self.handlers.append(handler)
    
    def _console_handler(self, entry: LogEntry):
        """Default console handler."""
        level_colors = {
            LogLevel.DEBUG: '\033[36m',    # Cyan
            LogLevel.INFO: '\033[32m',     # Green
            LogLevel.WARNING: '\033[33m',  # Yellow
            LogLevel.ERROR: '\033[31m',    # Red
            LogLevel.CRITICAL: '\033[35m', # Magenta
        }
        reset = '\033[0m'
        
        color = level_colors.get(entry.level, '')
        dt = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
        
        print(f"{dt} {color}[{entry.level.name}]{reset} {entry.message}", file=sys.stderr)
    
    def _log(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Internal logging method."""
        if level.value < self.level.value:
            return
        
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            logger_name=self.name,
            context=context or {},
            source=kwargs.get('source'),
            line=kwargs.get('line'),
        )
        
        self._buffer.append(entry)
        
        for handler in self.handlers:
            try:
                handler(entry)
            except Exception:
                pass
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def get_buffer(self) -> List[LogEntry]:
        """Get buffered log entries."""
        return list(self._buffer)


# =============================================================================
# Metrics Collection
# =============================================================================


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = auto()      # Monotonically increasing
    GAUGE = auto()        # Point-in-time value
    HISTOGRAM = auto()    # Distribution
    TIMER = auto()        # Duration measurements


@dataclass
class Metric:
    """
    Single metric value.
    
    Attributes:
        name: Metric name
        value: Current value
        metric_type: Type of metric
        labels: Metric labels/tags
        timestamp: When recorded
        unit: Unit of measurement
    """
    name: str
    value: float
    metric_type: MetricType = MetricType.GAUGE
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.name,
            'labels': self.labels,
            'timestamp': self.timestamp,
            'unit': self.unit,
        }


class MetricCollector:
    """
    Collects and aggregates metrics.
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize collector.
        
        Args:
            name: Collector name
        """
        self.name = name
        self._metrics: Dict[str, List[Metric]] = {}
        self._counters: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def record(self, name: str, value: float, **kwargs):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            **kwargs: Additional metric attributes
        """
        with self._lock:
            self._record_unlocked(name, value, **kwargs)
    
    def _record_unlocked(self, name: str, value: float, **kwargs):
        """Record metric without acquiring lock (caller must hold lock)."""
        metric = Metric(
            name=name,
            value=value,
            **kwargs,
        )
        
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(metric)
    
    def increment(self, name: str, value: float = 1.0, **kwargs):
        """
        Increment a counter.
        
        Args:
            name: Counter name
            value: Increment amount
        """
        with self._lock:
            if name not in self._counters:
                self._counters[name] = 0.0
            self._counters[name] += value
            
            self._record_unlocked(name, self._counters[name], 
                                  metric_type=MetricType.COUNTER, **kwargs)
    
    def timing(self, name: str, duration: float, **kwargs):
        """
        Record a timing metric.
        
        Args:
            name: Timer name
            duration: Duration in seconds
        """
        self.record(name, duration, metric_type=MetricType.TIMER, 
                   unit="seconds", **kwargs)
    
    def get_latest(self, name: str) -> Optional[Metric]:
        """Get the latest value for a metric."""
        with self._lock:
            if name in self._metrics and self._metrics[name]:
                return self._metrics[name][-1]
        return None
    
    def get_all(self, name: str) -> List[Metric]:
        """Get all values for a metric."""
        with self._lock:
            return list(self._metrics.get(name, []))
    
    def summary(self) -> Dict[str, Dict]:
        """Get summary statistics for all metrics."""
        with self._lock:
            summary = {}
            for name, metrics in self._metrics.items():
                if not metrics:
                    continue
                
                values = [m.value for m in metrics]
                summary[name] = {
                    'count': len(values),
                    'latest': values[-1],
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                }
            return summary
    
    def clear(self, name: Optional[str] = None):
        """Clear metrics."""
        with self._lock:
            if name:
                self._metrics.pop(name, None)
            else:
                self._metrics.clear()


class MetricsRegistry:
    """
    Global registry for metric collectors.
    """
    
    _collectors: Dict[str, MetricCollector] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_collector(cls, name: str = "default") -> MetricCollector:
        """Get or create a collector."""
        with cls._lock:
            if name not in cls._collectors:
                cls._collectors[name] = MetricCollector(name)
            return cls._collectors[name]
    
    @classmethod
    def record(cls, name: str, value: float, collector: str = "default", **kwargs):
        """Record a metric in a collector."""
        cls.get_collector(collector).record(name, value, **kwargs)
    
    @classmethod
    def summary(cls) -> Dict[str, Dict]:
        """Get summary from all collectors."""
        with cls._lock:
            return {
                coll_name: coll.summary() 
                for coll_name, coll in cls._collectors.items()
            }


# =============================================================================
# Telemetry
# =============================================================================


@dataclass
class TelemetryEvent:
    """
    Telemetry event for performance tracking.
    
    Attributes:
        name: Event name
        start_time: Start timestamp
        end_time: End timestamp
        metadata: Event metadata
        parent_id: Parent event ID for tracing
    """
    name: str
    start_time: float
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    event_id: str = ""
    
    def __post_init__(self):
        if not self.event_id:
            import uuid
            self.event_id = str(uuid.uuid4())[:8]
    
    @property
    def duration(self) -> Optional[float]:
        """Get event duration."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None
    
    def finish(self):
        """Mark event as finished."""
        self.end_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'metadata': self.metadata,
            'parent_id': self.parent_id,
        }


class TelemetryCollector:
    """
    Collects telemetry events for tracing.
    """
    
    def __init__(self, max_events: int = 10000):
        """
        Initialize collector.
        
        Args:
            max_events: Maximum events to buffer
        """
        self._events: deque = deque(maxlen=max_events)
        self._active: Dict[str, TelemetryEvent] = {}
        self._lock = threading.Lock()
    
    def start_event(
        self,
        name: str,
        parent_id: Optional[str] = None,
        **metadata,
    ) -> TelemetryEvent:
        """
        Start a new telemetry event.
        
        Args:
            name: Event name
            parent_id: Parent event ID
            **metadata: Event metadata
            
        Returns:
            The started event
        """
        event = TelemetryEvent(
            name=name,
            start_time=time.time(),
            parent_id=parent_id,
            metadata=metadata,
        )
        
        with self._lock:
            self._active[event.event_id] = event
        
        return event
    
    def end_event(self, event_id: str) -> Optional[TelemetryEvent]:
        """
        End a telemetry event.
        
        Args:
            event_id: Event ID to end
            
        Returns:
            The completed event
        """
        with self._lock:
            if event_id in self._active:
                event = self._active.pop(event_id)
                event.finish()
                self._events.append(event)
                return event
        return None
    
    def get_events(
        self,
        name: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[TelemetryEvent]:
        """
        Get telemetry events.
        
        Args:
            name: Filter by name
            since: Filter by timestamp
            
        Returns:
            Matching events
        """
        with self._lock:
            events = list(self._events)
        
        if name:
            events = [e for e in events if e.name == name]
        if since:
            events = [e for e in events if e.start_time >= since]
        
        return events
    
    def get_trace(self, event_id: str) -> List[TelemetryEvent]:
        """
        Get full trace for an event.
        
        Args:
            event_id: Root event ID
            
        Returns:
            Events in the trace
        """
        with self._lock:
            all_events = list(self._events)
        
        # Find root and all children
        trace = []
        for event in all_events:
            if event.event_id == event_id or event.parent_id == event_id:
                trace.append(event)
        
        return sorted(trace, key=lambda e: e.start_time)


# =============================================================================
# Alerting
# =============================================================================


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class Alert:
    """
    System alert.
    
    Attributes:
        name: Alert name
        severity: Alert severity
        message: Alert message
        timestamp: When alert was raised
        context: Additional context
        resolved: Whether alert is resolved
    """
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    
    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'severity': self.severity.name,
            'message': self.message,
            'timestamp': self.timestamp,
            'context': self.context,
            'resolved': self.resolved,
        }


class AlertManager:
    """
    Manages system alerts.
    """
    
    def __init__(self):
        """Initialize alert manager."""
        self._alerts: List[Alert] = []
        self._handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler."""
        self._handlers.append(handler)
    
    def raise_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        **context,
    ) -> Alert:
        """
        Raise a new alert.
        
        Args:
            name: Alert name
            severity: Severity level
            message: Alert message
            **context: Additional context
            
        Returns:
            The raised alert
        """
        alert = Alert(
            name=name,
            severity=severity,
            message=message,
            context=context,
        )
        
        with self._lock:
            self._alerts.append(alert)
        
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                pass
        
        return alert
    
    def get_active(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active (unresolved) alerts."""
        with self._lock:
            alerts = [a for a in self._alerts if not a.resolved]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts
    
    def resolve(self, name: str):
        """Resolve alerts by name."""
        with self._lock:
            for alert in self._alerts:
                if alert.name == name and not alert.resolved:
                    alert.resolve()


# =============================================================================
# Convenience Functions
# =============================================================================


def get_logger(name: str = "hypertensor") -> StructuredLogger:
    """Get a structured logger."""
    return StructuredLogger.get_logger(name)


def log_info(message: str, **context):
    """Log an info message."""
    get_logger().info(message, context=context)


def log_warning(message: str, **context):
    """Log a warning message."""
    get_logger().warning(message, context=context)


def log_error(message: str, **context):
    """Log an error message."""
    get_logger().error(message, context=context)


def record_metric(name: str, value: float, **kwargs):
    """Record a metric value."""
    MetricsRegistry.record(name, value, **kwargs)
