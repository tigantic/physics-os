#!/usr/bin/env python3
# Copyright 2025 Tigantic Labs. All Rights Reserved.
"""
HyperTensor REST API Server

Provides HTTP endpoints for field operations, useful for:
- Web applications
- Microservice architectures  
- Language-agnostic integration

Usage:
    uvicorn sdk.server.main:app --reload
    
    # Or with custom settings
    python -m sdk.server.main --port 8000 --workers 4
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field as PydanticField
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not installed. Install with: pip install fastapi uvicorn")


# =============================================================================
# MODELS
# =============================================================================

class ErrorDetail(BaseModel):
    """Structured error detail for debugging."""
    code: str = PydanticField(description="Error code for programmatic handling")
    message: str = PydanticField(description="Human-readable error message")
    correlation_id: Optional[str] = PydanticField(None, description="Request correlation ID for log tracing")
    field: Optional[str] = PydanticField(None, description="Field name if validation error")


class ErrorResponse(BaseModel):
    """Standard error response schema.
    
    Used for all HTTP 4xx and 5xx responses for consistent client handling.
    
    Example:
        {
            "error": {
                "code": "FIELD_NOT_FOUND",
                "message": "Field with handle 42 does not exist",
                "correlation_id": "abc123"
            }
        }
    """
    error: ErrorDetail


class FieldConfig(BaseModel):
    """Field initialization configuration."""
    size_x: int = PydanticField(64, ge=8, le=512)
    size_y: int = PydanticField(64, ge=8, le=512)
    size_z: int = PydanticField(64, ge=8, le=512)
    field_type: str = PydanticField("vector", pattern="^(scalar|vector|tensor)$")
    max_rank: int = PydanticField(32, ge=1, le=256)


class SampleRequest(BaseModel):
    """Field sampling request."""
    points: List[List[float]]  # [[x, y, z], ...]
    max_rank: int = 16


class SampleResponse(BaseModel):
    """Field sampling response."""
    values: List[List[float]]  # [[v0, v1, v2, v3], ...]
    rank_used: int
    error_estimate: float


class SliceRequest(BaseModel):
    """Field slice extraction request."""
    axis: int = PydanticField(2, ge=0, le=2)
    position: float = PydanticField(0.5, ge=0.0, le=1.0)
    resolution_x: int = PydanticField(256, ge=1, le=2048)
    resolution_y: int = PydanticField(256, ge=1, le=2048)
    max_rank: int = 16


class ImpulseRequest(BaseModel):
    """Field impulse request."""
    position: List[float]  # [x, y, z]
    direction: List[float]  # [dx, dy, dz]
    strength: float = 1.0
    radius: float = 0.1


class FieldStats(BaseModel):
    """Field statistics."""
    max_rank: int
    avg_rank: float
    n_cores: int
    truncation_error: float
    divergence_norm: float
    energy: float
    compression_ratio: float
    memory_bytes: int
    step_count: int
    state_hash: str


class FieldHandle(BaseModel):
    """Field handle response."""
    handle: int
    stats: FieldStats


# =============================================================================
# STATE
# =============================================================================

@dataclass
class ServerState:
    """Server state container with thread-safe access."""
    fields: Dict[int, Any] = field(default_factory=dict)
    next_handle: int = 1
    start_time: float = field(default_factory=time.time)
    request_count: int = 0
    
    def __post_init__(self):
        """Initialize lock for thread-safe state mutations."""
        import asyncio
        self._lock = asyncio.Lock()
    
    async def allocate_handle(self) -> int:
        """Thread-safe handle allocation."""
        async with self._lock:
            handle = self.next_handle
            self.next_handle += 1
            return handle
    
    async def add_field(self, handle: int, field_obj: Any):
        """Thread-safe field addition."""
        async with self._lock:
            self.fields[handle] = field_obj
            self.request_count += 1
    
    async def remove_field(self, handle: int) -> bool:
        """Thread-safe field removal."""
        async with self._lock:
            if handle in self.fields:
                del self.fields[handle]
                self.request_count += 1
                return True
            return False
    
    async def get_field(self, handle: int) -> Optional[Any]:
        """Thread-safe field retrieval."""
        async with self._lock:
            self.request_count += 1
            return self.fields.get(handle)


state = ServerState()


# =============================================================================
# APP
# =============================================================================

if HAS_FASTAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """App lifespan context."""
        print("HyperTensor Server starting...")
        yield
        print("HyperTensor Server shutting down...")
        state.fields.clear()

    app = FastAPI(
        title="HyperTensor API",
        description="REST API for HyperTensor field operations",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware - configurable via environment
    # SECURITY: Default to localhost-only in production
    CORS_ORIGINS = os.environ.get(
        "HYPERTENSOR_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8080"
    ).split(",")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # =========================================================================
    # ERROR HANDLING
    # =========================================================================
    
    # Error codes for consistent client handling
    class ErrorCodes:
        """Error codes for programmatic handling."""
        INTERNAL_ERROR = "INTERNAL_ERROR"
        FIELD_NOT_FOUND = "FIELD_NOT_FOUND"
        HYPERTENSOR_UNAVAILABLE = "HYPERTENSOR_UNAVAILABLE"
        VALIDATION_ERROR = "VALIDATION_ERROR"
        OPERATION_FAILED = "OPERATION_FAILED"
    
    def _sanitize_error(e: Exception, code: str = ErrorCodes.INTERNAL_ERROR) -> dict:
        """Sanitize error messages to prevent information leakage.
        
        Returns structured error detail for use with HTTPException.
        """
        import logging
        import uuid
        
        # Generate correlation ID for log tracing
        correlation_id = str(uuid.uuid4())[:8]
        
        # Log full error internally with correlation ID
        logging.getLogger(__name__).exception(
            f"Internal error [correlation_id={correlation_id}]"
        )
        
        # Return generic message to client with correlation ID
        return {
            "error": {
                "code": code,
                "message": "Internal server error. Check server logs for details.",
                "correlation_id": correlation_id,
            }
        }
    
    def _field_not_found_error(handle: int) -> dict:
        """Create a structured error for field not found."""
        return {
            "error": {
                "code": ErrorCodes.FIELD_NOT_FOUND,
                "message": f"Field with handle {handle} does not exist",
            }
        }

    # =========================================================================
    # ENDPOINTS
    # =========================================================================

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "uptime": time.time() - state.start_time,
            "active_fields": len(state.fields),
            "request_count": state.request_count,
        }

    @app.post("/fields", response_model=FieldHandle)
    async def create_field(config: FieldConfig):
        """Create a new field."""
        try:
            from tensornet.substrate import Field, FieldType
            import math
            
            field_type_map = {
                "scalar": FieldType.SCALAR,
                "vector": FieldType.VECTOR,
                "tensor": FieldType.TENSOR,
            }
            
            bits = max(
                math.ceil(math.log2(config.size_x)),
                math.ceil(math.log2(config.size_y)),
                math.ceil(math.log2(config.size_z)),
            )
            
            field = Field.create(
                dims=3,
                bits_per_dim=bits,
                field_type=field_type_map[config.field_type],
            )
            
            # Thread-safe handle allocation and field storage
            handle = await state.allocate_handle()
            await state.add_field(handle, field)
            
            stats = field.stats()
            
            return FieldHandle(
                handle=handle,
                stats=FieldStats(
                    max_rank=stats.max_rank,
                    avg_rank=stats.avg_rank,
                    n_cores=stats.n_cores,
                    truncation_error=stats.truncation_error,
                    divergence_norm=stats.divergence_norm,
                    energy=stats.energy,
                    compression_ratio=stats.compression_ratio,
                    memory_bytes=stats.qtt_memory_bytes,
                    step_count=stats.step_count,
                    state_hash=stats.state_hash,
                ),
            )
            
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail={"error": {"code": ErrorCodes.HYPERTENSOR_UNAVAILABLE, "message": "HyperTensor not available"}}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=_sanitize_error(e))

    @app.delete("/fields/{handle}")
    async def delete_field(handle: int):
        """Delete a field."""
        removed = await state.remove_field(handle)
        if not removed:
            raise HTTPException(status_code=404, detail=_field_not_found_error(handle))
        
        return {"status": "deleted", "handle": handle}

    @app.get("/fields/{handle}/stats", response_model=FieldStats)
    async def get_stats(handle: int):
        """Get field statistics."""
        field = await state.get_field(handle)
        if field is None:
            raise HTTPException(status_code=404, detail=_field_not_found_error(handle))
        
        stats = field.stats()
        
        return FieldStats(
            max_rank=stats.max_rank,
            avg_rank=stats.avg_rank,
            n_cores=stats.n_cores,
            truncation_error=stats.truncation_error,
            divergence_norm=stats.divergence_norm,
            energy=stats.energy,
            compression_ratio=stats.compression_ratio,
            memory_bytes=stats.qtt_memory_bytes,
            step_count=stats.step_count,
            state_hash=stats.state_hash,
        )

    @app.post("/fields/{handle}/sample", response_model=SampleResponse)
    async def sample_field(handle: int, request: SampleRequest):
        """Sample field at given points."""
        field = await state.get_field(handle)
        if field is None:
            raise HTTPException(status_code=404, detail=_field_not_found_error(handle))
        
        points = np.array(request.points, dtype=np.float32)
        
        try:
            values = field.sample(points, max_rank=request.max_rank)
            
            return SampleResponse(
                values=values.tolist(),
                rank_used=request.max_rank,
                error_estimate=0.0,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=_sanitize_error(e))

    @app.post("/fields/{handle}/step")
    async def step_field(handle: int, dt: float = Query(0.016, ge=0.0, le=1.0)):
        """Step field simulation."""
        field = await state.get_field(handle)
        if field is None:
            raise HTTPException(status_code=404, detail=_field_not_found_error(handle))
        
        try:
            field.step(dt)
            return {"status": "stepped", "dt": dt}
        except Exception as e:
            raise HTTPException(status_code=500, detail=_sanitize_error(e))

    @app.post("/fields/{handle}/impulse")
    async def apply_impulse(handle: int, request: ImpulseRequest):
        """Apply impulse to field."""
        field = await state.get_field(handle)
        if field is None:
            raise HTTPException(status_code=404, detail=_field_not_found_error(handle))
        
        try:
            from tensornet.operators import Impulse
            
            impulse = Impulse(
                center=np.array(request.position),
                direction=np.array(request.direction),
                strength=request.strength,
                radius=request.radius,
            )
            
            # Update field with impulse applied (thread-safe)
            new_field = impulse(field)
            await state.add_field(handle, new_field)
            return {"status": "applied"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=_sanitize_error(e))

    @app.get("/fields")
    async def list_fields():
        """List all active fields."""
        # Read-only access to fields dict - thread-safe for reads
        async with state._lock:
            state.request_count += 1
            return {
                "fields": list(state.fields.keys()),
                "count": len(state.fields),
            }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HyperTensor REST API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    import uvicorn
    uvicorn.run(
        "sdk.server.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
