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
    """Server state container."""
    fields: Dict[int, Any] = field(default_factory=dict)
    next_handle: int = 1
    start_time: float = field(default_factory=time.time)
    request_count: int = 0


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

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
        state.request_count += 1
        
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
            
            handle = state.next_handle
            state.next_handle += 1
            state.fields[handle] = field
            
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
            raise HTTPException(500, "HyperTensor not available")
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.delete("/fields/{handle}")
    async def delete_field(handle: int):
        """Delete a field."""
        state.request_count += 1
        
        if handle not in state.fields:
            raise HTTPException(404, f"Field {handle} not found")
        
        del state.fields[handle]
        return {"status": "deleted", "handle": handle}

    @app.get("/fields/{handle}/stats", response_model=FieldStats)
    async def get_stats(handle: int):
        """Get field statistics."""
        state.request_count += 1
        
        if handle not in state.fields:
            raise HTTPException(404, f"Field {handle} not found")
        
        field = state.fields[handle]
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
        state.request_count += 1
        
        if handle not in state.fields:
            raise HTTPException(404, f"Field {handle} not found")
        
        field = state.fields[handle]
        points = np.array(request.points, dtype=np.float32)
        
        try:
            values = field.sample(points, max_rank=request.max_rank)
            
            return SampleResponse(
                values=values.tolist(),
                rank_used=request.max_rank,
                error_estimate=0.0,
            )
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.post("/fields/{handle}/step")
    async def step_field(handle: int, dt: float = Query(0.016, ge=0.0, le=1.0)):
        """Step field simulation."""
        state.request_count += 1
        
        if handle not in state.fields:
            raise HTTPException(404, f"Field {handle} not found")
        
        field = state.fields[handle]
        
        try:
            field.step(dt)
            return {"status": "stepped", "dt": dt}
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.post("/fields/{handle}/impulse")
    async def apply_impulse(handle: int, request: ImpulseRequest):
        """Apply impulse to field."""
        state.request_count += 1
        
        if handle not in state.fields:
            raise HTTPException(404, f"Field {handle} not found")
        
        field = state.fields[handle]
        
        try:
            from tensornet.operators import Impulse
            
            impulse = Impulse(
                center=np.array(request.position),
                direction=np.array(request.direction),
                strength=request.strength,
                radius=request.radius,
            )
            
            state.fields[handle] = impulse(field)
            return {"status": "applied"}
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.get("/fields")
    async def list_fields():
        """List all active fields."""
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
