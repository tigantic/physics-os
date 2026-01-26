"""
Autonomous Discovery Engine - REST API Module

Phase 6: Unification - FastAPI-based REST API for all discovery domains.

Components:
    - Server: FastAPI application with comprehensive endpoints
    - Routes: Domain-specific discovery endpoints
    - Models: Pydantic request/response schemas
    - GPU: Icicle GPU acceleration backend
    - Distributed: Multi-GPU coordination
"""

from .server import (
    create_app,
    DiscoveryAPIServer,
)

from .models import (
    DiscoveryRequest,
    DiscoveryResponse,
    LiveDataRequest,
    LiveDataResponse,
    HealthResponse,
    GPUStatus,
    StreamingRequest,
    StreamingResponse,
)

from .gpu import (
    GPUBackend,
    IcicleAccelerator,
    gpu_available,
    get_gpu_info,
)

from .distributed import (
    DistributedCoordinator,
    WorkerNode,
    DistributedConfig,
)

__all__ = [
    # Server
    "create_app",
    "DiscoveryAPIServer",
    # Models
    "DiscoveryRequest",
    "DiscoveryResponse", 
    "LiveDataRequest",
    "LiveDataResponse",
    "HealthResponse",
    "GPUStatus",
    "StreamingRequest",
    "StreamingResponse",
    # GPU
    "GPUBackend",
    "IcicleAccelerator",
    "gpu_available",
    "get_gpu_info",
    # Distributed
    "DistributedCoordinator",
    "WorkerNode",
    "DistributedConfig",
]
