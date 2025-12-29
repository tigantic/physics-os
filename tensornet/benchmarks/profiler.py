"""
TensorRT inference profiler for detailed performance analysis.

This module provides profiling utilities to analyze TensorRT
inference at the layer and operation level.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
import time
import torch
import numpy as np


class OperationType(Enum):
    """Types of neural network operations."""
    CONVOLUTION = auto()
    MATRIX_MULTIPLY = auto()
    ACTIVATION = auto()
    NORMALIZATION = auto()
    POOLING = auto()
    ELEMENTWISE = auto()
    RESHAPE = auto()
    MEMORY = auto()
    REDUCTION = auto()
    ATTENTION = auto()
    OTHER = auto()


@dataclass
class ProfileConfig:
    """Configuration for profiling."""
    warmup_runs: int = 5
    profile_runs: int = 20
    layer_level: bool = True
    operation_level: bool = True
    memory_tracking: bool = True
    timeline_export: bool = False
    
    # CUDA profiling
    use_cuda_events: bool = True
    record_shapes: bool = True
    with_stack: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'warmup_runs': self.warmup_runs,
            'profile_runs': self.profile_runs,
            'layer_level': self.layer_level,
            'operation_level': self.operation_level,
            'memory_tracking': self.memory_tracking,
        }


@dataclass
class LayerProfile:
    """Profile for a single layer."""
    name: str
    layer_type: str
    operation_type: OperationType
    
    # Timing
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    percentage: float = 0.0
    
    # Memory
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    weight_size_bytes: int = 0
    workspace_bytes: int = 0
    
    # Shape info
    input_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    output_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    
    # Performance metrics
    flops: int = 0
    memory_bandwidth_gbps: float = 0.0
    compute_utilization: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'layer_type': self.layer_type,
            'operation_type': self.operation_type.name,
            'total_time_ms': round(self.total_time_ms, 4),
            'avg_time_ms': round(self.avg_time_ms, 4),
            'percentage': round(self.percentage, 2),
            'input_shapes': self.input_shapes,
            'output_shapes': self.output_shapes,
            'flops': self.flops,
        }


@dataclass
class OperationProfile:
    """Profile for a specific operation type."""
    operation_type: OperationType
    count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    percentage: float = 0.0
    total_flops: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation_type': self.operation_type.name,
            'count': self.count,
            'total_time_ms': round(self.total_time_ms, 4),
            'avg_time_ms': round(self.avg_time_ms, 4),
            'percentage': round(self.percentage, 2),
            'total_flops': self.total_flops,
        }


@dataclass
class ProfileResult:
    """Complete profiling result."""
    model_name: str
    total_time_ms: float
    
    # Layer profiles
    layer_profiles: List[LayerProfile] = field(default_factory=list)
    
    # Operation profiles
    operation_profiles: Dict[OperationType, OperationProfile] = field(default_factory=dict)
    
    # Memory summary
    total_memory_bytes: int = 0
    peak_memory_bytes: int = 0
    
    # Device info
    device_info: Dict[str, Any] = field(default_factory=dict)
    
    # Config
    config: Optional[ProfileConfig] = None
    
    def get_top_layers(self, n: int = 10) -> List[LayerProfile]:
        """Get top N layers by time."""
        return sorted(
            self.layer_profiles,
            key=lambda x: x.total_time_ms,
            reverse=True
        )[:n]
    
    def get_operation_breakdown(self) -> Dict[str, float]:
        """Get time breakdown by operation type."""
        return {
            op.name: profile.percentage
            for op, profile in self.operation_profiles.items()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'total_time_ms': round(self.total_time_ms, 4),
            'layers': [l.to_dict() for l in self.layer_profiles],
            'operations': {
                op.name: p.to_dict() 
                for op, p in self.operation_profiles.items()
            },
            'total_memory_bytes': self.total_memory_bytes,
            'peak_memory_bytes': self.peak_memory_bytes,
            'device_info': self.device_info,
        }


class LayerTimer:
    """Timer for individual layer profiling."""
    
    def __init__(self, use_cuda: bool = True):
        """Initialize timer."""
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.measurements: List[float] = []
        
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
    
    def start(self):
        """Start timing."""
        if self.use_cuda:
            self.start_event.record()
        else:
            self._start_time = time.perf_counter()
    
    def stop(self):
        """Stop timing and record measurement."""
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed = self.start_event.elapsed_time(self.end_event)
        else:
            elapsed = (time.perf_counter() - self._start_time) * 1000
        
        self.measurements.append(elapsed)
        return elapsed
    
    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        if not self.measurements:
            return {'total': 0, 'avg': 0, 'min': 0, 'max': 0}
        
        return {
            'total': sum(self.measurements),
            'avg': sum(self.measurements) / len(self.measurements),
            'min': min(self.measurements),
            'max': max(self.measurements),
        }


class TensorRTProfiler:
    """
    Profiler for TensorRT inference.
    
    Provides detailed layer-by-layer and operation-level profiling.
    """
    
    def __init__(self, config: Optional[ProfileConfig] = None):
        """
        Initialize profiler.
        
        Args:
            config: Profiling configuration
        """
        self.config = config or ProfileConfig()
        self._hooks: List[Any] = []
        self._layer_timers: Dict[str, LayerTimer] = {}
        self._layer_info: Dict[str, Dict[str, Any]] = {}
    
    def profile(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
    ) -> ProfileResult:
        """
        Profile model inference.
        
        Args:
            model: Model to profile
            input_tensor: Input tensor
        
        Returns:
            ProfileResult with detailed analysis
        """
        model_name = model.__class__.__name__
        
        # Register hooks
        self._register_hooks(model)
        
        try:
            # Warmup
            for _ in range(self.config.warmup_runs):
                with torch.no_grad():
                    _ = model(input_tensor)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # Profile runs
            total_times = []
            for _ in range(self.config.profile_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(input_tensor)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                elapsed = (time.perf_counter() - start) * 1000
                total_times.append(elapsed)
            
            avg_total_time = sum(total_times) / len(total_times)
            
        finally:
            # Remove hooks
            self._remove_hooks()
        
        # Build layer profiles
        layer_profiles = self._build_layer_profiles(avg_total_time)
        
        # Build operation profiles
        operation_profiles = self._build_operation_profiles(layer_profiles, avg_total_time)
        
        # Get memory info
        memory_info = self._get_memory_info()
        
        return ProfileResult(
            model_name=model_name,
            total_time_ms=avg_total_time,
            layer_profiles=layer_profiles,
            operation_profiles=operation_profiles,
            total_memory_bytes=memory_info.get('total', 0),
            peak_memory_bytes=memory_info.get('peak', 0),
            device_info=self._get_device_info(),
            config=self.config,
        )
    
    def _register_hooks(self, model: torch.nn.Module):
        """Register forward hooks for profiling."""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                timer = LayerTimer(self.config.use_cuda)
                self._layer_timers[name] = timer
                self._layer_info[name] = {
                    'type': module.__class__.__name__,
                    'module': module,
                }
                
                def make_hooks(layer_name, layer_timer):
                    def pre_hook(module, input):
                        layer_timer.start()
                    
                    def post_hook(module, input, output):
                        layer_timer.stop()
                        # Record shapes
                        if self.config.record_shapes:
                            if layer_name not in self._layer_info:
                                self._layer_info[layer_name] = {}
                            self._layer_info[layer_name]['input_shapes'] = [
                                tuple(i.shape) for i in input if hasattr(i, 'shape')
                            ]
                            if hasattr(output, 'shape'):
                                self._layer_info[layer_name]['output_shapes'] = [tuple(output.shape)]
                            elif isinstance(output, tuple):
                                self._layer_info[layer_name]['output_shapes'] = [
                                    tuple(o.shape) for o in output if hasattr(o, 'shape')
                                ]
                    
                    return pre_hook, post_hook
                
                pre_hook, post_hook = make_hooks(name, timer)
                h1 = module.register_forward_pre_hook(pre_hook)
                h2 = module.register_forward_hook(post_hook)
                self._hooks.extend([h1, h2])
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def _build_layer_profiles(self, total_time_ms: float) -> List[LayerProfile]:
        """Build layer profiles from timing data."""
        profiles = []
        
        for name, timer in self._layer_timers.items():
            stats = timer.get_stats()
            info = self._layer_info.get(name, {})
            
            layer_type = info.get('type', 'Unknown')
            op_type = self._classify_operation(layer_type)
            
            profile = LayerProfile(
                name=name,
                layer_type=layer_type,
                operation_type=op_type,
                total_time_ms=stats['total'],
                avg_time_ms=stats['avg'],
                min_time_ms=stats['min'],
                max_time_ms=stats['max'],
                percentage=100 * stats['avg'] / total_time_ms if total_time_ms > 0 else 0,
                input_shapes=info.get('input_shapes', []),
                output_shapes=info.get('output_shapes', []),
            )
            
            # Estimate FLOPs
            module = info.get('module')
            if module is not None:
                profile.flops = self._estimate_flops(module, info.get('input_shapes', []))
            
            profiles.append(profile)
        
        return profiles
    
    def _build_operation_profiles(
        self,
        layer_profiles: List[LayerProfile],
        total_time_ms: float,
    ) -> Dict[OperationType, OperationProfile]:
        """Build operation-level profiles."""
        op_stats: Dict[OperationType, Dict[str, Any]] = {}
        
        for layer in layer_profiles:
            op = layer.operation_type
            if op not in op_stats:
                op_stats[op] = {'count': 0, 'time': 0.0, 'flops': 0}
            
            op_stats[op]['count'] += 1
            op_stats[op]['time'] += layer.avg_time_ms
            op_stats[op]['flops'] += layer.flops
        
        profiles = {}
        for op, stats in op_stats.items():
            profiles[op] = OperationProfile(
                operation_type=op,
                count=stats['count'],
                total_time_ms=stats['time'],
                avg_time_ms=stats['time'] / stats['count'] if stats['count'] > 0 else 0,
                percentage=100 * stats['time'] / total_time_ms if total_time_ms > 0 else 0,
                total_flops=stats['flops'],
            )
        
        return profiles
    
    def _classify_operation(self, layer_type: str) -> OperationType:
        """Classify layer type to operation type."""
        layer_type = layer_type.lower()
        
        if 'conv' in layer_type:
            return OperationType.CONVOLUTION
        elif 'linear' in layer_type or 'matmul' in layer_type:
            return OperationType.MATRIX_MULTIPLY
        elif any(act in layer_type for act in ['relu', 'gelu', 'silu', 'sigmoid', 'tanh']):
            return OperationType.ACTIVATION
        elif any(norm in layer_type for norm in ['norm', 'batch', 'layer', 'instance']):
            return OperationType.NORMALIZATION
        elif 'pool' in layer_type:
            return OperationType.POOLING
        elif any(elem in layer_type for elem in ['add', 'mul', 'cat', 'concat']):
            return OperationType.ELEMENTWISE
        elif any(reshape in layer_type for reshape in ['reshape', 'view', 'flatten', 'squeeze']):
            return OperationType.RESHAPE
        elif 'attention' in layer_type or 'multihead' in layer_type:
            return OperationType.ATTENTION
        elif any(red in layer_type for red in ['mean', 'sum', 'max', 'min', 'softmax']):
            return OperationType.REDUCTION
        else:
            return OperationType.OTHER
    
    def _estimate_flops(
        self,
        module: torch.nn.Module,
        input_shapes: List[Tuple[int, ...]],
    ) -> int:
        """Estimate FLOPs for a module."""
        if not input_shapes:
            return 0
        
        input_shape = input_shapes[0]
        
        if isinstance(module, torch.nn.Conv2d):
            # FLOPs = 2 * Cout * Hout * Wout * Cin * Kh * Kw
            if len(input_shape) >= 4:
                _, c_in, h_in, w_in = input_shape
                c_out = module.out_channels
                kh, kw = module.kernel_size
                h_out = (h_in + 2 * module.padding[0] - kh) // module.stride[0] + 1
                w_out = (w_in + 2 * module.padding[1] - kw) // module.stride[1] + 1
                return 2 * c_out * h_out * w_out * c_in * kh * kw // module.groups
        
        elif isinstance(module, torch.nn.Linear):
            # FLOPs = 2 * batch * in_features * out_features
            if len(input_shape) >= 2:
                batch = np.prod(input_shape[:-1])
                return int(2 * batch * module.in_features * module.out_features)
        
        return 0
    
    def _get_memory_info(self) -> Dict[str, int]:
        """Get memory usage info."""
        if not torch.cuda.is_available():
            return {'total': 0, 'peak': 0}
        
        return {
            'total': torch.cuda.memory_allocated(),
            'peak': torch.cuda.max_memory_allocated(),
        }
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device info."""
        info = {'cuda_available': torch.cuda.is_available()}
        
        if torch.cuda.is_available():
            info['device_name'] = torch.cuda.get_device_name()
            info['device_capability'] = torch.cuda.get_device_capability()
        
        return info


def profile_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    config: Optional[ProfileConfig] = None,
) -> ProfileResult:
    """
    Profile a model with given input shape.
    
    Args:
        model: Model to profile
        input_shape: Input tensor shape
        config: Profiling configuration
    
    Returns:
        ProfileResult with detailed analysis
    """
    config = config or ProfileConfig()
    profiler = TensorRTProfiler(config)
    
    device = next(model.parameters()).device if list(model.parameters()) else 'cpu'
    input_tensor = torch.randn(input_shape, device=device)
    
    return profiler.profile(model, input_tensor)


def profile_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 20,
) -> Dict[str, Any]:
    """
    Quick inference profiling.
    
    Args:
        model: Model to profile
        input_tensor: Input tensor
        num_runs: Number of profiling runs
    
    Returns:
        Dictionary with profiling results
    """
    config = ProfileConfig(profile_runs=num_runs)
    profiler = TensorRTProfiler(config)
    
    result = profiler.profile(model, input_tensor)
    
    return {
        'total_time_ms': result.total_time_ms,
        'top_layers': [l.to_dict() for l in result.get_top_layers(5)],
        'operation_breakdown': result.get_operation_breakdown(),
    }
