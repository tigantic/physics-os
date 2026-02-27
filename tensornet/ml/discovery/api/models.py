"""
Pydantic Models for Discovery API

Request/Response schemas for all API endpoints.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime
import json


class DomainType(str, Enum):
    """Supported discovery domains."""
    DEFI = "defi"
    PLASMA = "plasma"
    MOLECULAR = "molecular"
    MARKETS = "markets"
    RAW = "raw"


class LiveMode(str, Enum):
    """Live data modes."""
    HISTORICAL = "historical"
    REPLAY = "replay"
    STREAM = "stream"


class HistoricalEvent(str, Enum):
    """Available historical events."""
    FLASH_CRASH_2010 = "flash-crash-2010"
    GME_2021 = "gme-2021"
    LEHMAN_2008 = "lehman-2008"


class SeverityLevel(str, Enum):
    """Finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class DiscoveryRequest:
    """
    Request to run discovery analysis.
    
    Attributes:
        domain: Target analysis domain (defi, plasma, molecular, markets, raw)
        data: Input data for analysis (domain-specific format)
        demo: If True, run demo mode with synthetic data.
              ⚠️ WARNING: Demo mode uses SYNTHETIC data for testing only.
              Do not use demo mode for production analysis.
        config: Optional configuration overrides
        use_gpu: Whether to use GPU acceleration if available
        verbose: Enable verbose output
    """
    domain: DomainType
    data: Optional[Dict[str, Any]] = None
    demo: bool = False
    config: Optional[Dict[str, Any]] = None
    use_gpu: bool = True
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain": self.domain.value,
            "data": self.data,
            "demo": self.demo,
            "config": self.config,
            "use_gpu": self.use_gpu,
            "verbose": self.verbose,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscoveryRequest":
        """Create from dictionary."""
        return cls(
            domain=DomainType(d["domain"]),
            data=d.get("data"),
            demo=d.get("demo", False),
            config=d.get("config"),
            use_gpu=d.get("use_gpu", True),
            verbose=d.get("verbose", False),
        )


@dataclass
class Finding:
    """A single discovery finding."""
    id: str
    severity: SeverityLevel
    category: str
    title: str
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    primitive: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "primitive": self.primitive,
            "confidence": self.confidence,
        }


@dataclass
class Hypothesis:
    """A generated hypothesis."""
    id: str
    statement: str
    confidence: float
    supporting_findings: List[str] = field(default_factory=list)
    suggested_tests: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "statement": self.statement,
            "confidence": self.confidence,
            "supporting_findings": self.supporting_findings,
            "suggested_tests": self.suggested_tests,
        }


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics."""
    total_time_ms: float
    stage_times: Dict[str, float] = field(default_factory=dict)
    gpu_used: bool = False
    memory_peak_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_time_ms": self.total_time_ms,
            "stage_times": self.stage_times,
            "gpu_used": self.gpu_used,
            "memory_peak_mb": self.memory_peak_mb,
        }


@dataclass
class DiscoveryResponse:
    """
    Response from discovery analysis.
    
    Attributes:
        success: Whether analysis completed successfully
        domain: Domain that was analyzed
        findings: List of discoveries/findings
        hypotheses: Generated hypotheses
        metrics: Execution metrics
        attestation_hash: SHA-256 hash of results for verification
        error: Error message if failed
    """
    success: bool
    domain: DomainType
    findings: List[Finding] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    metrics: Optional[PipelineMetrics] = None
    attestation_hash: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "domain": self.domain.value,
            "findings": [f.to_dict() for f in self.findings],
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "attestation_hash": self.attestation_hash,
            "timestamp": self.timestamp,
            "error": self.error,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class LiveDataRequest:
    """
    Request for live data analysis.
    
    Attributes:
        mode: Analysis mode (historical, replay, stream)
        event: Historical event name (for historical/replay modes)
        symbol: Trading symbol (for stream mode)
        duration: Duration in seconds (for stream mode)
        window_size: Analysis window size in bars
        use_gpu: Whether to use GPU acceleration
    """
    mode: LiveMode
    event: Optional[HistoricalEvent] = None
    symbol: str = "BTC-USD"
    duration: int = 60
    window_size: int = 20
    use_gpu: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "event": self.event.value if self.event else None,
            "symbol": self.symbol,
            "duration": self.duration,
            "window_size": self.window_size,
            "use_gpu": self.use_gpu,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LiveDataRequest":
        """Create from dictionary."""
        return cls(
            mode=LiveMode(d["mode"]),
            event=HistoricalEvent(d["event"]) if d.get("event") else None,
            symbol=d.get("symbol", "BTC-USD"),
            duration=d.get("duration", 60),
            window_size=d.get("window_size", 20),
            use_gpu=d.get("use_gpu", True),
        )


@dataclass
class StreamingAlert:
    """A real-time streaming alert."""
    timestamp: str
    alert_type: str
    severity: SeverityLevel
    message: str
    price: float
    bar_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "message": self.message,
            "price": self.price,
            "bar_index": self.bar_index,
        }


@dataclass
class LiveDataResponse:
    """
    Response from live data analysis.
    
    Attributes:
        success: Whether analysis completed successfully
        mode: Analysis mode used
        event: Historical event analyzed (if applicable)
        bars_analyzed: Number of bars analyzed
        alerts: Streaming alerts generated
        findings: Aggregate findings
        statistics: Analysis statistics
        error: Error message if failed
    """
    success: bool
    mode: LiveMode
    event: Optional[HistoricalEvent] = None
    bars_analyzed: int = 0
    alerts: List[StreamingAlert] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "mode": self.mode.value,
            "event": self.event.value if self.event else None,
            "bars_analyzed": self.bars_analyzed,
            "alerts": [a.to_dict() for a in self.alerts],
            "findings": [f.to_dict() for f in self.findings],
            "statistics": self.statistics,
            "timestamp": self.timestamp,
            "error": self.error,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class GPUDevice:
    """Information about a single GPU device."""
    index: int
    name: str
    memory_total_mb: int
    memory_free_mb: int
    compute_capability: str
    utilization_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_mb": self.memory_total_mb,
            "memory_free_mb": self.memory_free_mb,
            "compute_capability": self.compute_capability,
            "utilization_percent": self.utilization_percent,
        }


@dataclass
class GPUStatus:
    """
    GPU status and capabilities.
    
    Attributes:
        available: Whether GPU is available
        cuda_version: CUDA version string
        icicle_available: Whether Icicle is available
        devices: List of GPU devices
        active_device: Currently active device index
    """
    available: bool
    cuda_version: Optional[str] = None
    icicle_available: bool = False
    devices: List[GPUDevice] = field(default_factory=list)
    active_device: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "available": self.available,
            "cuda_version": self.cuda_version,
            "icicle_available": self.icicle_available,
            "devices": [d.to_dict() for d in self.devices],
            "active_device": self.active_device,
        }


@dataclass
class HealthResponse:
    """
    API health check response.
    
    Attributes:
        status: Health status ("healthy", "degraded", "unhealthy")
        version: API version
        uptime_seconds: Server uptime
        gpu: GPU status
        active_streams: Number of active streaming connections
        requests_processed: Total requests processed
    """
    status: str
    version: str
    uptime_seconds: float
    gpu: GPUStatus
    active_streams: int = 0
    requests_processed: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "gpu": self.gpu.to_dict(),
            "active_streams": self.active_streams,
            "requests_processed": self.requests_processed,
            "timestamp": self.timestamp,
        }


@dataclass
class StreamingRequest:
    """
    Request to start a streaming analysis session.
    
    Attributes:
        symbol: Trading symbol to stream
        exchange: Exchange to connect to
        window_size: Analysis window size
        bar_interval_seconds: Bar aggregation interval
        alert_threshold: Minimum alert severity to emit
        use_gpu: Whether to use GPU acceleration
    """
    symbol: str = "BTC-USD"
    exchange: str = "coinbase"
    window_size: int = 20
    bar_interval_seconds: int = 60
    alert_threshold: SeverityLevel = SeverityLevel.LOW
    use_gpu: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "window_size": self.window_size,
            "bar_interval_seconds": self.bar_interval_seconds,
            "alert_threshold": self.alert_threshold.value,
            "use_gpu": self.use_gpu,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StreamingRequest":
        """Create from dictionary."""
        return cls(
            symbol=d.get("symbol", "BTC-USD"),
            exchange=d.get("exchange", "coinbase"),
            window_size=d.get("window_size", 20),
            bar_interval_seconds=d.get("bar_interval_seconds", 60),
            alert_threshold=SeverityLevel(d.get("alert_threshold", "low")),
            use_gpu=d.get("use_gpu", True),
        )


@dataclass
class StreamingResponse:
    """
    Response for streaming session management.
    
    Attributes:
        success: Whether operation succeeded
        session_id: Unique session identifier
        status: Session status ("started", "running", "stopped", "error")
        websocket_url: WebSocket URL for streaming updates
        message: Status message
    """
    success: bool
    session_id: Optional[str] = None
    status: str = "unknown"
    websocket_url: Optional[str] = None
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "session_id": self.session_id,
            "status": self.status,
            "websocket_url": self.websocket_url,
            "message": self.message,
        }


@dataclass
class BatchDiscoveryRequest:
    """
    Request for batch discovery across multiple inputs.
    
    Attributes:
        domain: Target domain
        inputs: List of input data items
        parallel: Whether to process in parallel
        use_gpu: Whether to use GPU
    """
    domain: DomainType
    inputs: List[Dict[str, Any]]
    parallel: bool = True
    use_gpu: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain": self.domain.value,
            "inputs": self.inputs,
            "parallel": self.parallel,
            "use_gpu": self.use_gpu,
        }


@dataclass
class BatchDiscoveryResponse:
    """
    Response from batch discovery.
    
    Attributes:
        success: Whether batch completed successfully
        results: List of individual results
        total_time_ms: Total execution time
        failed_count: Number of failed items
    """
    success: bool
    results: List[DiscoveryResponse] = field(default_factory=list)
    total_time_ms: float = 0.0
    failed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "results": [r.to_dict() for r in self.results],
            "total_time_ms": self.total_time_ms,
            "failed_count": self.failed_count,
        }
