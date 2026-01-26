"""
Autonomous Discovery Engine - Cross-Primitive Analysis

Chains Genesis primitives: OT → SGW → RMT → TG → RKHS → PH → GA
Uses proven patterns from cross_primitive_pipeline.py

Phase 0: Core engine ✅
Phase 1: DeFi pipeline ✅
Phase 1.5: Hypothesis generator ✅
Phase 2: Fusion Plasma pipeline ✅
Phase 3: Molecular/Drug Discovery pipeline ✅
Phase 4: Financial Markets pipeline ✅
Phase 5: Live Data Connectors ✅
Phase 6: Unification API ✅
Phase 7: Production Hardening ✅
"""

from .engine_v2 import DiscoveryEngineV2 as DiscoveryEngine, Finding, DiscoveryResult
from .ingest.defi import DeFiIngester, ContractData, DeFiSnapshot
from .ingest.plasma import PlasmaIngester, PlasmaShot, PlasmaProfile, MagneticField3D
from .ingest.molecular import MolecularIngester, ProteinStructure, BindingSite, create_synthetic_protein
from .ingest.markets import (
    MarketsIngester, MarketSnapshot, OHLCV, Trade, OrderBookSnapshot, OrderBookLevel,
    MarketRegime, create_synthetic_flash_crash
)
from .pipelines.defi_pipeline import DeFiDiscoveryPipeline, DeFiDiscoveryConfig
from .pipelines.plasma_pipeline import PlasmaDiscoveryPipeline, PlasmaPipelineResult
from .pipelines.molecular_pipeline import MolecularDiscoveryPipeline, MolecularPipelineResult
from .pipelines.markets_pipeline import MarketsDiscoveryPipeline, MarketsPipelineResult, RegimeChange
from .connectors.coinbase_l2 import CoinbaseL2Connector, SimulatedL2Connector, L2Snapshot, L2Update
from .connectors.historical import HistoricalDataLoader, HistoricalEvent
from .connectors.streaming import StreamingPipeline, ReplayPipeline, StreamingConfig, StreamingResult
from .hypothesis.generator import HypothesisGenerator, Hypothesis
from .api.server import DiscoveryAPIServer, ServerConfig, create_app
from .api.gpu import GPUBackend, IcicleAccelerator, gpu_available, get_gpu_info
from .api.distributed import DistributedCoordinator, WorkerNode, DistributedConfig
from .api.models import (
    DiscoveryRequest, DiscoveryResponse, LiveDataRequest, LiveDataResponse,
    HealthResponse, GPUStatus, StreamingRequest, StreamingResponse
)
from .production.resilience import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState,
    RateLimiter, RateLimiterConfig, RateLimitExceeded,
    RetryPolicy, RetryStrategy,
    Bulkhead, BulkheadConfig, BulkheadFull,
    Timeout, TimeoutError as ResilienceTimeoutError,
    resilient, ResilienceConfig,
)
from .production.observability import (
    StructuredLogger, LogLevel, LogContext,
    MetricsCollector, MetricType, Metric,
    HealthChecker, HealthStatus, ComponentHealth, SystemHealth,
    Tracer, Span,
    get_logger, get_metrics,
)
from .production.security import (
    InputValidator, ValidationError,
    APIKeyAuth, AuthenticationError, AuthorizationError,
    RequestSigner,
    AuditLogger, AuditEventType, AuditEvent,
    CSPPolicy, get_security_headers, sanitize_input,
)
from .production.performance import (
    CacheManager, CachePolicy, CacheStats,
    ConnectionPool,
    BatchOptimizer, BatchConfig,
    MemoryManager, MemoryConfig,
    PerformanceProfiler,
    cached,
)

__all__ = [
    # Core
    "DiscoveryEngine",
    "Finding", 
    "DiscoveryResult",
    # DeFi (Phase 1)
    "DeFiIngester",
    "ContractData",
    "DeFiSnapshot",
    "DeFiDiscoveryPipeline",
    "DeFiDiscoveryConfig",
    # Hypothesis (Phase 1.5)
    "HypothesisGenerator",
    "Hypothesis",
    # Plasma (Phase 2)
    "PlasmaIngester",
    "PlasmaShot",
    "PlasmaProfile",
    "MagneticField3D",
    "PlasmaDiscoveryPipeline",
    "PlasmaPipelineResult",
    # Molecular (Phase 3)
    "MolecularIngester",
    "ProteinStructure",
    "BindingSite",
    "create_synthetic_protein",
    "MolecularDiscoveryPipeline",
    "MolecularPipelineResult",
    # Markets (Phase 4)
    "MarketsIngester",
    "MarketSnapshot",
    "OHLCV",
    "Trade",
    "OrderBookSnapshot",
    "OrderBookLevel",
    "MarketRegime",
    "create_synthetic_flash_crash",
    "MarketsDiscoveryPipeline",
    "MarketsPipelineResult",
    "RegimeChange",
    # Live Data Connectors (Phase 5)
    "CoinbaseL2Connector",
    "SimulatedL2Connector",
    "L2Snapshot",
    "L2Update",
    "HistoricalDataLoader",
    "HistoricalEvent",
    "StreamingPipeline",
    "ReplayPipeline",
    "StreamingConfig",
    "StreamingResult",
    # Unification API (Phase 6)
    "DiscoveryAPIServer",
    "ServerConfig",
    "create_app",
    "GPUBackend",
    "IcicleAccelerator",
    "gpu_available",
    "get_gpu_info",
    "DistributedCoordinator",
    "WorkerNode",
    "DistributedConfig",
    "DiscoveryRequest",
    "DiscoveryResponse",
    "LiveDataRequest",
    "LiveDataResponse",
    "HealthResponse",
    "GPUStatus",
    "StreamingRequest",
    "StreamingResponse",
    # Production Hardening (Phase 7)
    # Resilience
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitState",
    "RateLimiter",
    "RateLimiterConfig",
    "RateLimitExceeded",
    "RetryPolicy",
    "RetryStrategy",
    "Bulkhead",
    "BulkheadConfig",
    "BulkheadFull",
    "Timeout",
    "ResilienceTimeoutError",
    "resilient",
    "ResilienceConfig",
    # Observability
    "StructuredLogger",
    "LogLevel",
    "LogContext",
    "MetricsCollector",
    "MetricType",
    "Metric",
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "SystemHealth",
    "Tracer",
    "Span",
    "get_logger",
    "get_metrics",
    # Security
    "InputValidator",
    "ValidationError",
    "APIKeyAuth",
    "AuthenticationError",
    "AuthorizationError",
    "RequestSigner",
    "AuditLogger",
    "AuditEventType",
    "AuditEvent",
    "CSPPolicy",
    "get_security_headers",
    "sanitize_input",
    # Performance
    "CacheManager",
    "CachePolicy",
    "CacheStats",
    "ConnectionPool",
    "BatchOptimizer",
    "BatchConfig",
    "MemoryManager",
    "MemoryConfig",
    "PerformanceProfiler",
    "cached",
]
