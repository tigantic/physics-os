"""
FieldOS Kernel
===============

Main orchestrator that integrates all layers.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union, Type
from enum import Enum
import time
import threading
import logging

from .field import Field, FieldType
from .pipeline import Pipeline, Stage, StageResult


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FieldOSConfig:
    """
    Configuration for FieldOS kernel.
    """
    # Field defaults
    default_dtype: str = "float32"
    default_device: str = "cpu"
    
    # Memory
    max_fields: int = 1000
    max_memory_mb: float = 4096.0
    gc_threshold: float = 0.9  # Trigger GC at 90% memory
    
    # Execution
    max_threads: int = 4
    timeout_seconds: float = 300.0
    enable_profiling: bool = False
    
    # Provenance
    enable_provenance: bool = True
    provenance_store: Optional[str] = None
    
    # Plugins
    plugin_dirs: List[str] = field(default_factory=list)
    enabled_plugins: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_dtype": self.default_dtype,
            "default_device": self.default_device,
            "max_fields": self.max_fields,
            "max_memory_mb": self.max_memory_mb,
            "gc_threshold": self.gc_threshold,
            "max_threads": self.max_threads,
            "timeout_seconds": self.timeout_seconds,
            "enable_profiling": self.enable_profiling,
            "enable_provenance": self.enable_provenance,
            "provenance_store": self.provenance_store,
            "plugin_dirs": self.plugin_dirs,
            "enabled_plugins": self.enabled_plugins,
        }


# =============================================================================
# KERNEL STATE
# =============================================================================

class KernelState(Enum):
    """State of the FieldOS kernel."""
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


@dataclass
class KernelStats:
    """Runtime statistics for the kernel."""
    uptime_seconds: float = 0.0
    fields_created: int = 0
    fields_active: int = 0
    pipelines_run: int = 0
    total_operations: int = 0
    memory_used_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uptime_seconds": self.uptime_seconds,
            "fields_created": self.fields_created,
            "fields_active": self.fields_active,
            "pipelines_run": self.pipelines_run,
            "total_operations": self.total_operations,
            "memory_used_mb": self.memory_used_mb,
        }


# =============================================================================
# FIELDOS KERNEL
# =============================================================================

class FieldOS:
    """
    The FieldOS Kernel - Main orchestrator.
    
    Integrates all layers:
    - Substrate: QTT-based field oracle
    - FieldOps: Physics operators
    - HyperVisual: Tile-based rendering
    - HyperSim: Gymnasium RL environment
    - HyperEnv: Multi-agent training
    - Provenance: Merkle DAG audit
    - Intent: Natural language steering
    
    Example:
        os = FieldOS()
        os.start()
        
        # Create field
        field = os.create_field("temperature", (256, 256))
        
        # Run pipeline
        result = os.run_pipeline(my_pipeline, field)
        
        # Query with intent
        answer = os.query("What is the maximum temperature?")
        
        os.shutdown()
    """
    
    _instance: Optional['FieldOS'] = None
    
    def __new__(cls, config: Optional[FieldOSConfig] = None):
        """Singleton pattern - one kernel per process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[FieldOSConfig] = None):
        if self._initialized:
            return
        
        self.config = config or FieldOSConfig()
        self._state = KernelState.UNINITIALIZED
        self._stats = KernelStats()
        self._start_time: Optional[float] = None
        
        # Field registry
        self._fields: Dict[str, Field] = {}
        
        # Pipeline registry
        self._pipelines: Dict[str, Pipeline] = {}
        
        # Event handlers
        self._handlers: Dict[str, List[Callable]] = {}
        
        # Threading
        self._lock = threading.RLock()
        
        # Optional components (lazy loaded)
        self._intent_engine = None
        self._provenance_store = None
        self._plugin_manager = None
        
        self._initialized = True
        logger.info("FieldOS kernel created")
    
    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None
    
    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    
    def start(self) -> 'FieldOS':
        """Start the kernel."""
        with self._lock:
            if self._state == KernelState.RUNNING:
                return self
            
            self._start_time = time.time()
            self._state = KernelState.READY
            
            # Initialize components
            self._init_components()
            
            self._state = KernelState.RUNNING
            self._emit("started", {"time": self._start_time})
            logger.info("FieldOS kernel started")
        
        return self
    
    def pause(self):
        """Pause kernel execution."""
        with self._lock:
            if self._state == KernelState.RUNNING:
                self._state = KernelState.PAUSED
                self._emit("paused", {})
    
    def resume(self):
        """Resume kernel execution."""
        with self._lock:
            if self._state == KernelState.PAUSED:
                self._state = KernelState.RUNNING
                self._emit("resumed", {})
    
    def shutdown(self):
        """Shutdown the kernel."""
        with self._lock:
            self._state = KernelState.SHUTTING_DOWN
            
            # Save state if needed
            self._save_state()
            
            # Clear resources
            self._fields.clear()
            self._pipelines.clear()
            
            self._state = KernelState.UNINITIALIZED
            self._emit("shutdown", {})
            logger.info("FieldOS kernel shutdown")
    
    def _init_components(self):
        """Initialize kernel components."""
        # Load plugins
        if self.config.plugin_dirs:
            self._load_plugins()
        
        # Initialize provenance if enabled
        if self.config.enable_provenance:
            self._init_provenance()
    
    def _load_plugins(self):
        """Load plugins from configured directories."""
        # Will be implemented when plugin.py is created
        pass
    
    def _init_provenance(self):
        """Initialize provenance store."""
        # Will connect to provenance layer
        pass
    
    def _save_state(self):
        """Save kernel state on shutdown."""
        pass
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def state(self) -> KernelState:
        return self._state
    
    @property
    def is_running(self) -> bool:
        return self._state == KernelState.RUNNING
    
    @property
    def stats(self) -> KernelStats:
        if self._start_time:
            self._stats.uptime_seconds = time.time() - self._start_time
        self._stats.fields_active = len(self._fields)
        return self._stats
    
    # -------------------------------------------------------------------------
    # Field Management
    # -------------------------------------------------------------------------
    
    def create_field(
        self,
        name: str,
        shape: tuple,
        field_type: FieldType = FieldType.SCALAR,
        initial: Union[float, np.ndarray] = 0.0,
        **metadata,
    ) -> Field:
        """
        Create a new field.
        
        Args:
            name: Field name (unique identifier)
            shape: Field shape
            field_type: Type of field
            initial: Initial value
            **metadata: Additional metadata
            
        Returns:
            Created Field
        """
        with self._lock:
            if name in self._fields:
                raise ValueError(f"Field '{name}' already exists")
            
            if len(self._fields) >= self.config.max_fields:
                raise RuntimeError(f"Maximum fields ({self.config.max_fields}) reached")
            
            # Create based on type
            if field_type == FieldType.SCALAR:
                field = Field.scalar(name, shape, initial)
            elif field_type == FieldType.VECTOR:
                field = Field.vector(name, shape)
            else:
                data = np.full(shape, initial, dtype=self.config.default_dtype)
                field = Field.from_array(name, data, field_type)
            
            self._fields[name] = field
            self._stats.fields_created += 1
            self._emit("field_created", {"name": name})
            
            return field
    
    def get_field(self, name: str) -> Optional[Field]:
        """Get field by name."""
        return self._fields.get(name)
    
    def list_fields(self) -> List[str]:
        """List all field names."""
        return list(self._fields.keys())
    
    def delete_field(self, name: str):
        """Delete a field."""
        with self._lock:
            if name in self._fields:
                del self._fields[name]
                self._emit("field_deleted", {"name": name})
    
    def register_field(self, field: Field) -> Field:
        """Register an existing field."""
        with self._lock:
            if field.metadata.name in self._fields:
                raise ValueError(f"Field '{field.metadata.name}' already exists")
            self._fields[field.metadata.name] = field
            return field
    
    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------
    
    def register_pipeline(self, name: str, pipeline: Pipeline) -> Pipeline:
        """Register a pipeline."""
        self._pipelines[name] = pipeline
        return pipeline
    
    def get_pipeline(self, name: str) -> Optional[Pipeline]:
        """Get registered pipeline."""
        return self._pipelines.get(name)
    
    def run_pipeline(
        self,
        pipeline: Union[str, Pipeline],
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> StageResult:
        """
        Run a pipeline.
        
        Args:
            pipeline: Pipeline name or Pipeline object
            input_data: Input to pipeline
            context: Additional context
            
        Returns:
            StageResult from pipeline execution
        """
        if not self.is_running:
            raise RuntimeError("Kernel not running")
        
        # Resolve pipeline
        if isinstance(pipeline, str):
            pipeline = self._pipelines.get(pipeline)
            if pipeline is None:
                raise ValueError(f"Pipeline '{pipeline}' not found")
        
        # Build context
        ctx = context or {}
        ctx["kernel"] = self
        ctx["fields"] = self._fields
        
        # Execute
        self._stats.pipelines_run += 1
        result = pipeline.run(input_data, ctx)
        
        self._emit("pipeline_completed", {
            "pipeline": pipeline.name,
            "success": result.is_success,
        })
        
        return result
    
    # -------------------------------------------------------------------------
    # Intent Interface
    # -------------------------------------------------------------------------
    
    def query(self, intent: str) -> Any:
        """
        Query using natural language.
        
        Args:
            intent: Natural language query
            
        Returns:
            Query result
        """
        if self._intent_engine is None:
            self._init_intent_engine()
        
        # Process intent
        result = self._intent_engine.execute(intent, self._fields)
        self._stats.total_operations += 1
        
        return result
    
    def _init_intent_engine(self):
        """Initialize intent engine lazily."""
        try:
            from ..intent import IntentEngine
            self._intent_engine = IntentEngine()
        except ImportError:
            logger.warning("Intent module not available")
            # Create stub
            class StubEngine:
                def execute(self, intent, fields):
                    return {"error": "Intent engine not available"}
            self._intent_engine = StubEngine()
    
    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------
    
    def on(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
    
    def off(self, event: str, handler: Callable):
        """Unregister event handler."""
        if event in self._handlers:
            self._handlers[event] = [h for h in self._handlers[event] if h != handler]
    
    def _emit(self, event: str, data: Dict[str, Any]):
        """Emit event to handlers."""
        if event in self._handlers:
            for handler in self._handlers[event]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
    
    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------
    
    def __enter__(self) -> 'FieldOS':
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_kernel() -> FieldOS:
    """Get or create the FieldOS kernel."""
    return FieldOS()


def create_field(name: str, shape: tuple, **kwargs) -> Field:
    """Create field using global kernel."""
    return get_kernel().create_field(name, shape, **kwargs)


def run(pipeline: Union[str, Pipeline], input_data: Any) -> StageResult:
    """Run pipeline using global kernel."""
    return get_kernel().run_pipeline(pipeline, input_data)


def query(intent: str) -> Any:
    """Query using global kernel."""
    return get_kernel().query(intent)
