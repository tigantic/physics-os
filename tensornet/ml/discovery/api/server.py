"""
FastAPI Server for Autonomous Discovery Engine

Production-grade REST API with WebSocket streaming support.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable
from datetime import datetime
import time
import uuid
import asyncio
import logging
import hashlib
import json
import traceback

logger = logging.getLogger(__name__)

# Optional FastAPI imports - graceful degradation
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None

from .models import (
    DomainType,
    LiveMode,
    HistoricalEvent,
    SeverityLevel,
    DiscoveryRequest,
    DiscoveryResponse,
    LiveDataRequest,
    LiveDataResponse,
    HealthResponse,
    GPUStatus,
    GPUDevice,
    StreamingRequest,
    StreamingResponse,
    Finding,
    Hypothesis,
    PipelineMetrics,
    StreamingAlert,
    BatchDiscoveryRequest,
    BatchDiscoveryResponse,
)

from .gpu import (
    GPUBackend,
    IcicleAccelerator,
    gpu_available,
    get_gpu_info,
    AcceleratorConfig,
)

from .distributed import (
    DistributedCoordinator,
    DistributedConfig,
    DistributedPipeline,
)


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_concurrent_requests: int = 100
    request_timeout: float = 300.0
    enable_gpu: bool = True
    enable_distributed: bool = False
    api_key: Optional[str] = None


@dataclass
class ServerStats:
    """Server statistics."""
    start_time: float = field(default_factory=time.time)
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    active_streams: int = 0
    active_requests: int = 0
    
    @property
    def uptime_seconds(self) -> float:
        """Get server uptime in seconds."""
        return time.time() - self.start_time


class StreamingSession:
    """A WebSocket streaming session."""
    
    def __init__(
        self,
        session_id: str,
        websocket: Any,
        config: StreamingRequest
    ):
        """
        Initialize streaming session.
        
        Args:
            session_id: Unique session ID
            websocket: WebSocket connection
            config: Streaming configuration
        """
        self.id = session_id
        self.websocket = websocket
        self.config = config
        self.active = True
        self.start_time = time.time()
        self.bars_sent = 0
        self.alerts_sent = 0
    
    async def send_alert(self, alert: StreamingAlert) -> None:
        """Send an alert through the WebSocket."""
        if self.active:
            try:
                await self.websocket.send_json(alert.to_dict())
                self.alerts_sent += 1
            except Exception as e:
                logger.error(f"Error sending alert: {e}")
                self.active = False
    
    async def send_bar(self, bar: Dict[str, Any]) -> None:
        """Send a bar update through the WebSocket."""
        if self.active:
            try:
                await self.websocket.send_json({"type": "bar", "data": bar})
                self.bars_sent += 1
            except Exception as e:
                logger.error(f"Error sending bar: {e}")
                self.active = False


class DiscoveryAPIServer:
    """
    Main API server for Autonomous Discovery Engine.
    
    Provides:
    - REST endpoints for discovery analysis
    - WebSocket streaming for real-time data
    - GPU acceleration support
    - Distributed execution support
    """
    
    def __init__(self, config: Optional[ServerConfig] = None):
        """
        Initialize the API server.
        
        Args:
            config: Server configuration
        """
        self.config = config or ServerConfig()
        self.stats = ServerStats()
        self._app: Optional[FastAPI] = None
        self._gpu_backend: Optional[GPUBackend] = None
        self._icicle: Optional[IcicleAccelerator] = None
        self._distributed: Optional[DistributedPipeline] = None
        self._streaming_sessions: Dict[str, StreamingSession] = {}
        self._pipelines: Dict[str, Any] = {}
        
        if FASTAPI_AVAILABLE:
            self._create_app()
        else:
            logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    def _create_app(self) -> None:
        """Create the FastAPI application."""
        self._app = FastAPI(
            title="Autonomous Discovery Engine API",
            description="Cross-Primitive Discovery using QTT Genesis Stack",
            version="1.6.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        
        # Add CORS middleware
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
        
        # Startup/shutdown events
        self._app.on_event("startup")(self._on_startup)
        self._app.on_event("shutdown")(self._on_shutdown)
    
    async def _on_startup(self) -> None:
        """Handle server startup."""
        logger.info("Starting Autonomous Discovery Engine API...")
        
        # Initialize GPU backend
        if self.config.enable_gpu and gpu_available():
            self._gpu_backend = GPUBackend(AcceleratorConfig())
            self._icicle = IcicleAccelerator(AcceleratorConfig())
            logger.info(f"GPU acceleration enabled: {self._gpu_backend.accelerator_type.value}")
        
        # Initialize distributed execution
        if self.config.enable_distributed:
            self._distributed = DistributedPipeline()
            self._distributed.initialize()
            logger.info("Distributed execution enabled")
        
        # Load pipelines
        self._load_pipelines()
        
        logger.info(f"Server started on {self.config.host}:{self.config.port}")
    
    async def _on_shutdown(self) -> None:
        """Handle server shutdown."""
        logger.info("Shutting down...")
        
        # Close streaming sessions
        for session in list(self._streaming_sessions.values()):
            session.active = False
        
        # Shutdown distributed
        if self._distributed:
            self._distributed.shutdown()
        
        logger.info("Server stopped")
    
    def _load_pipelines(self) -> None:
        """Load discovery pipelines."""
        try:
            from ..pipelines.defi_pipeline import DeFiDiscoveryPipeline
            self._pipelines["defi"] = DeFiDiscoveryPipeline
            logger.info("Loaded DeFi pipeline")
        except ImportError as e:
            logger.warning(f"Could not load DeFi pipeline: {e}")
        
        try:
            from ..pipelines.plasma_pipeline import PlasmaDiscoveryPipeline
            self._pipelines["plasma"] = PlasmaDiscoveryPipeline
            logger.info("Loaded Plasma pipeline")
        except ImportError as e:
            logger.warning(f"Could not load Plasma pipeline: {e}")
        
        try:
            from ..pipelines.molecular_pipeline import MolecularDiscoveryPipeline
            self._pipelines["molecular"] = MolecularDiscoveryPipeline
            logger.info("Loaded Molecular pipeline")
        except ImportError as e:
            logger.warning(f"Could not load Molecular pipeline: {e}")
        
        try:
            from ..pipelines.markets_pipeline import MarketsDiscoveryPipeline
            self._pipelines["markets"] = MarketsDiscoveryPipeline
            logger.info("Loaded Markets pipeline")
        except ImportError as e:
            logger.warning(f"Could not load Markets pipeline: {e}")
    
    def _register_routes(self) -> None:
        """Register API routes."""
        
        # Health check
        @self._app.get("/health", response_model=None)
        async def health_check() -> Dict[str, Any]:
            """Health check endpoint."""
            gpu_status = self._get_gpu_status()
            return HealthResponse(
                status="healthy",
                version="1.6.0",
                uptime_seconds=self.stats.uptime_seconds,
                gpu=gpu_status,
                active_streams=len(self._streaming_sessions),
                requests_processed=self.stats.requests_total,
            ).to_dict()
        
        # Info endpoint
        @self._app.get("/info")
        async def get_info() -> Dict[str, Any]:
            """Get server information."""
            return {
                "name": "Autonomous Discovery Engine",
                "version": "1.6.0",
                "status": "Phase 6 - Unification",
                "domains": ["defi", "plasma", "molecular", "markets", "raw"],
                "primitives": ["OT", "SGW", "RMT", "TG", "RKHS", "PH", "GA"],
                "capabilities": {
                    "gpu_acceleration": gpu_available(),
                    "distributed_mode": self.config.enable_distributed,
                    "streaming": True,
                    "batch_processing": True,
                },
                "loaded_pipelines": list(self._pipelines.keys()),
            }
        
        # Discovery endpoint
        @self._app.post("/discover", response_model=None)
        async def discover(request: Dict[str, Any]) -> Dict[str, Any]:
            """Run discovery analysis."""
            self.stats.requests_total += 1
            self.stats.active_requests += 1
            
            try:
                req = DiscoveryRequest.from_dict(request)
                result = await self._run_discovery(req)
                self.stats.requests_success += 1
                return result.to_dict()
            except Exception as e:
                self.stats.requests_failed += 1
                logger.error(f"Discovery error: {e}\n{traceback.format_exc()}")
                return DiscoveryResponse(
                    success=False,
                    domain=DomainType(request.get("domain", "raw")),
                    error=str(e),
                ).to_dict()
            finally:
                self.stats.active_requests -= 1
        
        # Batch discovery endpoint
        @self._app.post("/discover/batch", response_model=None)
        async def discover_batch(request: Dict[str, Any]) -> Dict[str, Any]:
            """Run batch discovery analysis."""
            self.stats.requests_total += 1
            
            try:
                domain = DomainType(request["domain"])
                inputs = request["inputs"]
                parallel = request.get("parallel", True)
                
                results = []
                start = time.time()
                
                if parallel and self._distributed:
                    # Use distributed execution
                    raw_results = self._distributed.discover_batch(
                        domain.value, inputs
                    )
                    for r in raw_results:
                        if r:
                            results.append(self._convert_pipeline_result(domain, r))
                        else:
                            results.append(DiscoveryResponse(
                                success=False, domain=domain, error="Execution failed"
                            ))
                else:
                    # Sequential execution
                    for inp in inputs:
                        req = DiscoveryRequest(domain=domain, data=inp)
                        result = await self._run_discovery(req)
                        results.append(result)
                
                elapsed = (time.time() - start) * 1000
                failed = sum(1 for r in results if not r.success)
                
                return BatchDiscoveryResponse(
                    success=failed == 0,
                    results=results,
                    total_time_ms=elapsed,
                    failed_count=failed,
                ).to_dict()
                
            except Exception as e:
                logger.error(f"Batch discovery error: {e}")
                return BatchDiscoveryResponse(
                    success=False,
                    failed_count=len(request.get("inputs", [])),
                ).to_dict()
        
        # Live data endpoint
        @self._app.post("/live", response_model=None)
        async def live_data(request: Dict[str, Any]) -> Dict[str, Any]:
            """Run live data analysis."""
            self.stats.requests_total += 1
            
            try:
                req = LiveDataRequest.from_dict(request)
                result = await self._run_live_analysis(req)
                return result.to_dict()
            except Exception as e:
                logger.error(f"Live data error: {e}")
                return LiveDataResponse(
                    success=False,
                    mode=LiveMode(request.get("mode", "historical")),
                    error=str(e),
                ).to_dict()
        
        # GPU status endpoint
        @self._app.get("/gpu", response_model=None)
        async def gpu_status() -> Dict[str, Any]:
            """Get GPU status."""
            return self._get_gpu_status().to_dict()
        
        # Streaming management endpoints
        @self._app.post("/stream/start", response_model=None)
        async def start_stream(
            request: Dict[str, Any],
            background_tasks: BackgroundTasks
        ) -> Dict[str, Any]:
            """Start a streaming session."""
            try:
                config = StreamingRequest.from_dict(request)
                session_id = str(uuid.uuid4())
                
                return StreamingResponse(
                    success=True,
                    session_id=session_id,
                    status="ready",
                    websocket_url=f"ws://{self.config.host}:{self.config.port}/stream/{session_id}",
                    message="Connect to WebSocket URL to receive updates",
                ).to_dict()
                
            except Exception as e:
                return StreamingResponse(
                    success=False,
                    status="error",
                    message=str(e),
                ).to_dict()
        
        @self._app.get("/stream/sessions")
        async def list_sessions() -> Dict[str, Any]:
            """List active streaming sessions."""
            return {
                "sessions": [
                    {
                        "id": s.id,
                        "active": s.active,
                        "uptime_seconds": time.time() - s.start_time,
                        "bars_sent": s.bars_sent,
                        "alerts_sent": s.alerts_sent,
                    }
                    for s in self._streaming_sessions.values()
                ]
            }
        
        # WebSocket streaming
        @self._app.websocket("/stream/{session_id}")
        async def websocket_stream(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for streaming."""
            await websocket.accept()
            
            # Create session with default config
            session = StreamingSession(
                session_id=session_id,
                websocket=websocket,
                config=StreamingRequest()
            )
            self._streaming_sessions[session_id] = session
            self.stats.active_streams += 1
            
            try:
                # Send initial message
                await websocket.send_json({
                    "type": "connected",
                    "session_id": session_id,
                    "message": "Streaming session started"
                })
                
                # Run streaming analysis
                await self._stream_analysis(session)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {session_id}")
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})
            finally:
                session.active = False
                del self._streaming_sessions[session_id]
                self.stats.active_streams -= 1
        
        # Distributed status endpoint
        @self._app.get("/distributed/status")
        async def distributed_status() -> Dict[str, Any]:
            """Get distributed execution status."""
            if not self._distributed or not self._distributed.coordinator:
                return {"enabled": False, "message": "Distributed mode not enabled"}
            
            return {
                "enabled": True,
                "status": self._distributed.coordinator.get_status(),
            }
    
    async def _run_discovery(self, request: DiscoveryRequest) -> DiscoveryResponse:
        """
        Run discovery analysis.
        
        Args:
            request: Discovery request
            
        Returns:
            Discovery response
        """
        start = time.time()
        
        # Get pipeline for domain
        pipeline_class = self._pipelines.get(request.domain.value)
        if not pipeline_class:
            return DiscoveryResponse(
                success=False,
                domain=request.domain,
                error=f"Pipeline not available for domain: {request.domain.value}",
            )
        
        # Create pipeline instance
        pipeline = pipeline_class()
        
        # Run discovery
        if request.demo:
            result = pipeline.demo()
        elif request.data:
            result = pipeline.discover(request.data)
        else:
            return DiscoveryResponse(
                success=False,
                domain=request.domain,
                error="Either demo=true or data must be provided",
            )
        
        # Convert result to response
        elapsed = (time.time() - start) * 1000
        return self._convert_pipeline_result(request.domain, result, elapsed)
    
    def _convert_pipeline_result(
        self,
        domain: DomainType,
        result: Dict[str, Any],
        elapsed_ms: float = 0.0
    ) -> DiscoveryResponse:
        """Convert pipeline result to API response."""
        findings = []
        for i, f in enumerate(result.get("findings", [])):
            findings.append(Finding(
                id=f.get("id", f"finding_{i}"),
                severity=SeverityLevel(f.get("severity", "info")),
                category=f.get("category", "unknown"),
                title=f.get("title", ""),
                description=f.get("description", ""),
                evidence=f.get("evidence", {}),
                primitive=f.get("primitive"),
                confidence=f.get("confidence", 0.0),
            ))
        
        hypotheses = []
        for i, h in enumerate(result.get("hypotheses", [])):
            hypotheses.append(Hypothesis(
                id=h.get("id", f"hypothesis_{i}"),
                statement=h.get("statement", ""),
                confidence=h.get("confidence", 0.0),
                supporting_findings=h.get("supporting_findings", []),
            ))
        
        # Create attestation hash
        attestation_data = json.dumps({
            "domain": domain.value,
            "findings_count": len(findings),
            "timestamp": datetime.utcnow().isoformat(),
        })
        attestation_hash = hashlib.sha256(attestation_data.encode()).hexdigest()
        
        return DiscoveryResponse(
            success=True,
            domain=domain,
            findings=findings,
            hypotheses=hypotheses,
            metrics=PipelineMetrics(
                total_time_ms=elapsed_ms,
                gpu_used=self._gpu_backend is not None,
            ),
            attestation_hash=attestation_hash,
        )
    
    async def _run_live_analysis(self, request: LiveDataRequest) -> LiveDataResponse:
        """
        Run live data analysis.
        
        Args:
            request: Live data request
            
        Returns:
            Live data response
        """
        try:
            from ..connectors import (
                HistoricalDataLoader,
                SimulatedL2Connector,
                StreamingPipeline,
                ReplayPipeline,
                StreamingConfig,
            )
        except ImportError as e:
            return LiveDataResponse(
                success=False,
                mode=request.mode,
                error=f"Connectors not available: {e}",
            )
        
        if request.mode == LiveMode.HISTORICAL:
            # Load historical event
            loader = HistoricalDataLoader()
            
            if request.event == HistoricalEvent.FLASH_CRASH_2010:
                event = loader.load_2010_flash_crash()
            elif request.event == HistoricalEvent.GME_2021:
                event = loader.load_2021_gme_squeeze()
            elif request.event == HistoricalEvent.LEHMAN_2008:
                event = loader.load_2008_lehman_week()
            else:
                return LiveDataResponse(
                    success=False,
                    mode=request.mode,
                    error=f"Unknown event: {request.event}",
                )
            
            return LiveDataResponse(
                success=True,
                mode=request.mode,
                event=request.event,
                bars_analyzed=len(event.bars),
                statistics={
                    "name": event.name,
                    "date": event.date,
                    "bars": len(event.bars),
                    "key_stat": event.metadata.get("peak_drawdown", event.metadata),
                },
            )
        
        elif request.mode == LiveMode.REPLAY:
            # Replay historical event through pipeline
            loader = HistoricalDataLoader()
            
            if request.event == HistoricalEvent.FLASH_CRASH_2010:
                event = loader.load_2010_flash_crash()
            elif request.event == HistoricalEvent.GME_2021:
                event = loader.load_2021_gme_squeeze()
            elif request.event == HistoricalEvent.LEHMAN_2008:
                event = loader.load_2008_lehman_week()
            else:
                return LiveDataResponse(
                    success=False,
                    mode=request.mode,
                    error=f"Unknown event: {request.event}",
                )
            
            replay = ReplayPipeline(window_size=request.window_size)
            result = replay.replay(event)
            
            alerts = []
            for a in result.get("alerts", [])[:10]:  # Limit to 10 alerts
                alerts.append(StreamingAlert(
                    timestamp=a.get("timestamp", ""),
                    alert_type=a.get("type", "unknown"),
                    severity=SeverityLevel(a.get("severity", "low")),
                    message=a.get("message", ""),
                    price=a.get("price", 0.0),
                    bar_index=a.get("bar_index", 0),
                ))
            
            return LiveDataResponse(
                success=True,
                mode=request.mode,
                event=request.event,
                bars_analyzed=result.get("bars_analyzed", 0),
                alerts=alerts,
                statistics=result.get("statistics", {}),
            )
        
        elif request.mode == LiveMode.STREAM:
            # Simulated streaming
            logger.warning(
                "⚠️ SIMULATED STREAMING: Using synthetic market data. "
                "Results are for testing/development only."
            )
            connector = SimulatedL2Connector(symbol=request.symbol)
            config = StreamingConfig(
                window_size=request.window_size,
                bar_interval_seconds=5,
            )
            pipeline = StreamingPipeline(config)
            
            # Run for limited duration
            connector.start()
            bars_analyzed = 0
            
            for _ in range(min(request.duration, 30)):
                updates = connector.get_updates(timeout=1.0)
                for update in updates[:5]:
                    snapshot = connector.get_snapshot()
                    if snapshot:
                        pipeline.process_snapshot(snapshot)
                        bars_analyzed += 1
            
            connector.stop()
            
            return LiveDataResponse(
                success=True,
                mode=request.mode,
                bars_analyzed=bars_analyzed,
                statistics=pipeline.get_statistics(),
            )
        
        return LiveDataResponse(
            success=False,
            mode=request.mode,
            error=f"Unknown mode: {request.mode}",
        )
    
    async def _stream_analysis(self, session: StreamingSession) -> None:
        """
        Run streaming analysis for a session.
        
        Args:
            session: Streaming session
        """
        try:
            from ..connectors import SimulatedL2Connector, StreamingPipeline, StreamingConfig
        except ImportError as e:
            await session.websocket.send_json({
                "type": "error",
                "message": f"Connectors not available: {e}"
            })
            return
        
        # Create simulated connector
        connector = SimulatedL2Connector(symbol=session.config.symbol)
        config = StreamingConfig(
            window_size=session.config.window_size,
            bar_interval_seconds=session.config.bar_interval_seconds,
        )
        pipeline = StreamingPipeline(config)
        
        # Alert callback
        async def on_alert(alert: Dict[str, Any]) -> None:
            await session.send_alert(StreamingAlert(
                timestamp=alert.get("timestamp", datetime.utcnow().isoformat()),
                alert_type=alert.get("type", "unknown"),
                severity=SeverityLevel(alert.get("severity", "low")),
                message=alert.get("message", ""),
                price=alert.get("price", 0.0),
                bar_index=alert.get("bar_index", 0),
            ))
        
        # Start connector
        connector.start()
        
        try:
            while session.active:
                updates = connector.get_updates(timeout=0.5)
                
                for update in updates[:5]:
                    snapshot = connector.get_snapshot()
                    if snapshot:
                        result = pipeline.process_snapshot(snapshot)
                        
                        # Send bar update
                        await session.send_bar({
                            "mid_price": (snapshot.best_bid + snapshot.best_ask) / 2,
                            "spread": snapshot.best_ask - snapshot.best_bid,
                            "bid_depth": sum(snapshot.bids.values()),
                            "ask_depth": sum(snapshot.asks.values()),
                        })
                        
                        # Check for alerts
                        if result and result.get("alert"):
                            await on_alert(result["alert"])
                
                await asyncio.sleep(0.1)
                
        finally:
            connector.stop()
    
    def _get_gpu_status(self) -> GPUStatus:
        """Get current GPU status."""
        info = get_gpu_info()
        
        devices = []
        for d in info.get("devices", []):
            devices.append(GPUDevice(
                index=d["index"],
                name=d["name"],
                memory_total_mb=d["memory_total_mb"],
                memory_free_mb=d["memory_free_mb"],
                compute_capability=d["compute_capability"],
                utilization_percent=0.0,  # Would need nvidia-smi for real value
            ))
        
        return GPUStatus(
            available=info["available"],
            cuda_version=info["cuda_version"],
            icicle_available=info["icicle_available"],
            devices=devices,
            active_device=0 if devices else None,
        )
    
    @property
    def app(self) -> Optional[FastAPI]:
        """Get the FastAPI application."""
        return self._app
    
    def run(self) -> None:
        """Run the server."""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        import uvicorn
        uvicorn.run(
            self._app,
            host=self.config.host,
            port=self.config.port,
            log_level="info" if not self.config.debug else "debug",
        )


def create_app(config: Optional[ServerConfig] = None) -> Optional[FastAPI]:
    """
    Create a FastAPI application instance.
    
    Args:
        config: Server configuration
        
    Returns:
        FastAPI application
    """
    server = DiscoveryAPIServer(config)
    return server.app


# CLI entry point
def main():
    """Main entry point for API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Discovery Engine API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed mode")
    
    args = parser.parse_args()
    
    config = ServerConfig(
        host=args.host,
        port=args.port,
        debug=args.debug,
        enable_gpu=not args.no_gpu,
        enable_distributed=args.distributed,
    )
    
    server = DiscoveryAPIServer(config)
    server.run()


if __name__ == "__main__":
    main()
