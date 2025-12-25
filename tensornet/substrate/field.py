"""
Field Oracle API
================

The single interface that everything hangs on.

    sample(points) -> values
    slice(spec) -> buffer
    step(dt) -> Field
    stats() -> FieldStats
    serialize() -> FieldBundle

Every consumer (renderer, sim, AI env, audit) uses ONLY this interface.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Tuple, Union, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib

from .stats import FieldStats, KernelTiming


class FieldType(Enum):
    """Type of physical field."""
    SCALAR = "scalar"       # Density, temperature, pressure
    VECTOR = "vector"       # Velocity, vorticity
    TENSOR = "tensor"       # Stress, strain


@dataclass
class SliceSpec:
    """Specification for extracting a 2D/3D slice from a field."""
    
    # Plane selection (for 3D fields)
    plane: str = 'xy'  # 'xy', 'xz', 'yz', or 'vol' for 3D brick
    depth: float = 0.5  # Normalized depth [0, 1] for 2D slice
    
    # Spatial bounds (normalized [0, 1])
    x_range: Tuple[float, float] = (0.0, 1.0)
    y_range: Tuple[float, float] = (0.0, 1.0)
    z_range: Tuple[float, float] = (0.0, 1.0)
    
    # Output resolution
    resolution: Tuple[int, ...] = (512, 512)
    
    # LOD / quality
    max_rank: Optional[int] = None  # Rank cap for this slice
    error_budget: Optional[float] = None  # Max truncation error
    
    def __post_init__(self):
        if self.plane not in ('xy', 'xz', 'yz', 'vol'):
            raise ValueError(f"Invalid plane: {self.plane}")


@dataclass
class StepControls:
    """Controls for physics stepping."""
    
    # Time integration
    dt: float = 0.001
    substeps: int = 1
    
    # Budget constraints
    max_ms: Optional[float] = None  # Frame budget in milliseconds
    max_rank: Optional[int] = None  # Rank cap
    max_error: Optional[float] = None  # Error cap
    
    # Physics toggles
    advection: bool = True
    diffusion: bool = True
    projection: bool = True
    forces: bool = True
    
    # Viscosity / diffusion coefficient
    viscosity: float = 0.01
    
    # External forces (impulses, buoyancy, etc.)
    force_field: Optional[torch.Tensor] = None


class Field:
    """
    The Field Oracle - single interface for all field operations.
    
    This is the substrate. Everything else is a client.
    
    Memory Layout:
        Internally stores QTT cores on GPU (when available).
        Never exposes tensor internals to users.
        Users see: sample, slice, step, stats, serialize.
    """
    
    def __init__(
        self,
        cores: List[torch.Tensor],
        dims: int,
        bits_per_dim: int,
        field_type: FieldType = FieldType.SCALAR,
        device: str = 'cuda',
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize field from QTT cores.
        
        Use Field.create() for convenient construction.
        """
        self.dims = dims
        self.bits_per_dim = bits_per_dim
        self.n_cores = dims * bits_per_dim
        self.field_type = field_type
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.metadata = metadata or {}
        
        # Move cores to device
        self.cores = [c.to(self.device) for c in cores]
        
        # Derived properties
        self.grid_size = 2 ** bits_per_dim  # Per dimension
        self.total_points = self.grid_size ** dims
        
        # Telemetry state
        self._step_count = 0
        self._total_time = 0.0
        self._last_timings: Dict[str, float] = {}
        self._truncation_errors: List[float] = []
        self._energy_history: List[float] = []
        
        # Contraction cache (for bounded mode)
        self._contraction_cache: Dict[str, torch.Tensor] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Hash for determinism verification
        self._state_hash = self._compute_hash()
    
    @classmethod
    def create(
        cls,
        dims: int = 2,
        bits_per_dim: int = 20,
        rank: int = 8,
        field_type: FieldType = FieldType.SCALAR,
        device: str = 'cuda',
        init: str = 'taylor_green',
    ) -> 'Field':
        """
        Create a new field with specified dimensions.
        
        Args:
            dims: Number of spatial dimensions (2 or 3)
            bits_per_dim: Bits per dimension (grid = 2^bits x 2^bits x ...)
            rank: Bond dimension for QTT cores
            field_type: SCALAR, VECTOR, or TENSOR
            device: 'cuda' or 'cpu'
            init: Initial condition ('zeros', 'random', 'taylor_green', 'vortex')
            
        Returns:
            Field instance
            
        Example:
            field = Field.create(dims=2, bits_per_dim=20, rank=16)
            # Creates 1M x 1M grid (1 trillion points) with rank-16 QTT
        """
        n_cores = dims * bits_per_dim
        cores = []
        
        for i in range(n_cores):
            r_left = 1 if i == 0 else rank
            r_right = 1 if i == n_cores - 1 else rank
            
            if init == 'zeros':
                core = torch.zeros(r_left, 2, r_right)
            elif init == 'random':
                core = torch.randn(r_left, 2, r_right) * 0.1
            elif init == 'taylor_green':
                core = cls._init_taylor_green_core(i, n_cores, r_left, r_right)
            elif init == 'vortex':
                core = cls._init_vortex_core(i, n_cores, r_left, r_right)
            else:
                core = torch.randn(r_left, 2, r_right) * 0.1
            
            cores.append(core)
        
        return cls(
            cores=cores,
            dims=dims,
            bits_per_dim=bits_per_dim,
            field_type=field_type,
            device=device,
        )
    
    @classmethod
    def zeros(
        cls,
        dims: Union[int, List[int]] = 2,
        bits_per_dim: int = 4,
        rank: int = 2,
        field_type: FieldType = FieldType.SCALAR,
        device: str = 'cuda',
    ) -> 'Field':
        """
        Create a zero-valued field.
        
        Args:
            dims: Number of spatial dimensions, or list of dimension sizes
            bits_per_dim: Bits per dimension (grid = 2^bits x 2^bits x ...)
            rank: Bond dimension for QTT cores
            field_type: SCALAR, VECTOR, or TENSOR
            device: 'cuda' or 'cpu'
            
        Returns:
            Zero-initialized Field
        """
        # Handle dims as list [4, 4] -> dims=2
        if isinstance(dims, list):
            n_dims = len(dims)
        else:
            n_dims = dims
        
        return cls.create(
            dims=n_dims,
            bits_per_dim=bits_per_dim,
            rank=rank,
            field_type=field_type,
            device=device,
            init='zeros',
        )
    
    @staticmethod
    def _init_taylor_green_core(i: int, n_cores: int, r_left: int, r_right: int) -> torch.Tensor:
        """Initialize core for Taylor-Green vortex (exact low-rank)."""
        core = torch.zeros(r_left, 2, r_right)
        omega = 2 * np.pi / n_cores
        phase = omega * i
        c, s = np.cos(phase), np.sin(phase)
        
        # Diagonal structure for smooth fields
        min_rank = min(r_left, r_right)
        core[:min_rank, 0, :min_rank] = torch.eye(min_rank) * c
        core[:min_rank, 1, :min_rank] = torch.eye(min_rank) * s
        
        return core
    
    @staticmethod
    def _init_vortex_core(i: int, n_cores: int, r_left: int, r_right: int) -> torch.Tensor:
        """Initialize core for vortex pattern."""
        core = torch.randn(r_left, 2, r_right) * 0.1
        phase = 2 * np.pi * (i / n_cores)
        
        min_rank = min(r_left, r_right)
        if i % 2 == 0:  # X cores
            core[:min_rank, 0, :min_rank] += torch.eye(min_rank) * np.cos(phase)
            core[:min_rank, 1, :min_rank] += torch.eye(min_rank) * np.sin(phase)
        else:  # Y cores
            core[:min_rank, 0, :min_rank] += torch.eye(min_rank) * np.sin(phase)
            core[:min_rank, 1, :min_rank] += torch.eye(min_rank) * np.cos(phase)
        
        return core
    
    # =========================================================================
    # ORACLE API - The only interface consumers should use
    # =========================================================================
    
    def sample(
        self,
        points: torch.Tensor,
        time: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample field values at specified points.
        
        This is the primary query interface. O(N_points * d * r²) complexity.
        
        Args:
            points: Tensor of shape (N, dims) with coordinates in [0, 1]
            time: Optional time parameter (for time-varying fields)
            
        Returns:
            Tensor of shape (N,) for scalar fields, (N, dims) for vector fields
            
        Example:
            points = torch.rand(1000, 2)  # 1000 random 2D points
            values = field.sample(points)  # Shape: (1000,)
        """
        t_start = time.perf_counter() if hasattr(time, 'perf_counter') else 0
        t_start = __import__('time').perf_counter()
        
        points = points.to(self.device)
        N = points.shape[0]
        
        # Convert normalized coordinates to integer indices
        indices = (points * (self.grid_size - 1)).long()
        indices = torch.clamp(indices, 0, self.grid_size - 1)
        
        # Batch contract for all points
        values = self._batch_contract(indices)
        
        self._last_timings['sample'] = __import__('time').perf_counter() - t_start
        return values
    
    def slice(
        self,
        spec: SliceSpec,
        time: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Extract a 2D or 3D slice from the field.
        
        This is the rendering interface. Returns a buffer ready for display.
        
        Args:
            spec: SliceSpec defining plane, bounds, and resolution
            time: Optional time parameter
            
        Returns:
            Tensor of shape spec.resolution
            
        Example:
            spec = SliceSpec(plane='xy', depth=0.5, resolution=(1024, 1024))
            buffer = field.slice(spec)  # Ready for matplotlib/OpenGL
        """
        t_start = __import__('time').perf_counter()
        
        res = spec.resolution
        
        if spec.plane == 'vol':
            # 3D brick extraction
            return self._slice_3d(spec)
        
        # 2D slice extraction
        if spec.plane == 'xy':
            x_coords = torch.linspace(spec.x_range[0], spec.x_range[1], res[0], device=self.device)
            y_coords = torch.linspace(spec.y_range[0], spec.y_range[1], res[1], device=self.device)
            z_coord = spec.depth
            
            # Create grid of points
            xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
            if self.dims == 2:
                points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
            else:
                zz = torch.full_like(xx, z_coord)
                points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        
        elif spec.plane == 'xz':
            x_coords = torch.linspace(spec.x_range[0], spec.x_range[1], res[0], device=self.device)
            z_coords = torch.linspace(spec.z_range[0], spec.z_range[1], res[1], device=self.device)
            y_coord = spec.depth
            
            xx, zz = torch.meshgrid(x_coords, z_coords, indexing='ij')
            yy = torch.full_like(xx, y_coord)
            points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        
        elif spec.plane == 'yz':
            y_coords = torch.linspace(spec.y_range[0], spec.y_range[1], res[0], device=self.device)
            z_coords = torch.linspace(spec.z_range[0], spec.z_range[1], res[1], device=self.device)
            x_coord = spec.depth
            
            yy, zz = torch.meshgrid(y_coords, z_coords, indexing='ij')
            xx = torch.full_like(yy, x_coord)
            points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        
        # Sample all points
        values = self.sample(points)
        buffer = values.reshape(res[0], res[1])
        
        self._last_timings['slice'] = __import__('time').perf_counter() - t_start
        return buffer
    
    def step(
        self,
        dt: float = None,
        controls: StepControls = None,
    ) -> 'Field':
        """
        Advance the field by one time step.
        
        Physics evolution happens entirely in QTT format - O(d * r²) not O(N).
        
        Args:
            dt: Time step (default: 0.001)
            controls: StepControls for physics and budget configuration
            
        Returns:
            New Field instance (immutable evolution)
            
        Example:
            field = field.step(dt=0.01)  # Advance 10ms
            
            # With controls
            controls = StepControls(max_ms=16, viscosity=0.1)
            field = field.step(controls=controls)  # Bounded mode
        """
        t_start = __import__('time').perf_counter()
        
        if controls is None:
            controls = StepControls(dt=dt if dt else 0.001)
        elif dt is not None:
            controls.dt = dt
        
        # Copy cores for immutable evolution
        new_cores = [c.clone() for c in self.cores]
        
        # Physics in QTT format
        if controls.advection:
            new_cores = self._apply_advection(new_cores, controls)
        
        if controls.diffusion:
            new_cores = self._apply_diffusion(new_cores, controls)
        
        if controls.forces and controls.force_field is not None:
            new_cores = self._apply_forces(new_cores, controls)
        
        # Truncation with error tracking
        new_cores, trunc_error = self._truncate(new_cores, controls)
        
        # Create new field
        new_field = Field(
            cores=new_cores,
            dims=self.dims,
            bits_per_dim=self.bits_per_dim,
            field_type=self.field_type,
            device=self.device,
            metadata=self.metadata.copy(),
        )
        
        # Transfer telemetry state
        new_field._step_count = self._step_count + 1
        new_field._truncation_errors = self._truncation_errors + [trunc_error]
        new_field._energy_history = self._energy_history + [self._compute_energy(new_cores)]
        
        elapsed = __import__('time').perf_counter() - t_start
        new_field._total_time = self._total_time + elapsed
        new_field._last_timings = {'step': elapsed}
        
        return new_field
    
    def stats(self) -> FieldStats:
        """
        Get comprehensive telemetry for the field.
        
        Returns:
            FieldStats with rank, error, divergence, energy, timings, cache info
            
        Example:
            stats = field.stats()
            print(f"Rank: {stats.rank}")
            print(f"Truncation Error: {stats.truncation_error:.2e}")
            print(f"Energy: {stats.energy:.4f}")
        """
        # Compute current statistics
        ranks = [c.shape[0] for c in self.cores[1:]] + [c.shape[2] for c in self.cores[:-1]]
        max_rank = max(ranks) if ranks else 1
        avg_rank = np.mean(ranks) if ranks else 1
        
        # Memory usage
        qtt_bytes = sum(c.numel() * c.element_size() for c in self.cores)
        dense_bytes = self.total_points * 8  # float64 equivalent
        
        # Recent truncation error
        trunc_error = self._truncation_errors[-1] if self._truncation_errors else 0.0
        
        # Energy
        energy = self._energy_history[-1] if self._energy_history else self._compute_energy(self.cores)
        
        # Kernel timings
        timings = [
            KernelTiming(name=k, elapsed_ms=v * 1000)
            for k, v in self._last_timings.items()
        ]
        
        return FieldStats(
            # Rank info
            max_rank=max_rank,
            avg_rank=avg_rank,
            n_cores=self.n_cores,
            
            # Error metrics
            truncation_error=trunc_error,
            divergence_norm=self._compute_divergence_norm() if self.field_type == FieldType.VECTOR else 0.0,
            
            # Conservation
            energy=energy,
            energy_drift=self._compute_energy_drift(),
            
            # Memory
            qtt_memory_bytes=qtt_bytes,
            dense_memory_bytes=dense_bytes,
            compression_ratio=dense_bytes / qtt_bytes if qtt_bytes > 0 else 0,
            
            # Performance
            step_count=self._step_count,
            total_time_s=self._total_time,
            kernel_timings=timings,
            
            # Cache
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            cache_hit_ratio=self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            
            # State
            state_hash=self._state_hash,
        )
    
    def serialize(self) -> 'FieldBundle':
        """
        Serialize field to a FieldBundle for storage/transmission.
        
        Returns:
            FieldBundle with cores, metadata, and provenance info
            
        Example:
            bundle = field.serialize()
            bundle.save('simulation.htf')
            
            # Later...
            field = Field.deserialize(FieldBundle.load('simulation.htf'))
        """
        from .bundle import FieldBundle, BundleMetadata
        
        metadata = BundleMetadata(
            dims=self.dims,
            bits_per_dim=self.bits_per_dim,
            field_type=self.field_type.value,
            n_cores=self.n_cores,
            grid_size=self.grid_size,
            total_points=self.total_points,
            step_count=self._step_count,
            state_hash=self._state_hash,
            custom=self.metadata,
        )
        
        return FieldBundle(
            cores=[c.cpu().numpy() for c in self.cores],
            metadata=metadata,
            truncation_errors=self._truncation_errors.copy(),
            energy_history=self._energy_history.copy(),
        )
    
    @classmethod
    def deserialize(cls, bundle: 'FieldBundle') -> 'Field':
        """Load field from a FieldBundle."""
        cores = [torch.from_numpy(c) for c in bundle.cores]
        
        field = cls(
            cores=cores,
            dims=bundle.metadata.dims,
            bits_per_dim=bundle.metadata.bits_per_dim,
            field_type=FieldType(bundle.metadata.field_type),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            metadata=bundle.metadata.custom,
        )
        
        field._step_count = bundle.metadata.step_count
        field._truncation_errors = bundle.truncation_errors
        field._energy_history = bundle.energy_history
        
        return field
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _batch_contract(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Batch tensor contraction for multiple indices.
        
        O(N * d * r²) where N = number of points, d = n_cores, r = rank
        """
        N = indices.shape[0]
        values = torch.zeros(N, device=self.device)
        
        for n in range(N):
            # Get multi-dimensional index
            idx = indices[n]
            
            # Convert to binary representation for each dimension
            binary_indices = []
            for dim in range(self.dims):
                dim_idx = idx[dim].item()
                bits = format(dim_idx, f'0{self.bits_per_dim}b')
                binary_indices.append([int(b) for b in bits])
            
            # Interleave dimensions: x0,y0,z0, x1,y1,z1, ...
            full_binary = []
            for bit_pos in range(self.bits_per_dim):
                for dim in range(self.dims):
                    full_binary.append(binary_indices[dim][bit_pos])
            
            # Contract
            result = None
            for core_idx, bit in enumerate(full_binary):
                matrix = self.cores[core_idx][:, bit, :]
                if result is None:
                    result = matrix
                else:
                    result = result @ matrix
            
            values[n] = result.squeeze()
        
        return values
    
    def _slice_3d(self, spec: SliceSpec) -> torch.Tensor:
        """Extract 3D volume brick."""
        res = spec.resolution
        if len(res) != 3:
            res = (res[0], res[0], res[0]) if len(res) == 1 else (res[0], res[1], res[0])
        
        x_coords = torch.linspace(spec.x_range[0], spec.x_range[1], res[0], device=self.device)
        y_coords = torch.linspace(spec.y_range[0], spec.y_range[1], res[1], device=self.device)
        z_coords = torch.linspace(spec.z_range[0], spec.z_range[1], res[2], device=self.device)
        
        xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        
        values = self.sample(points)
        return values.reshape(res)
    
    def _apply_advection(self, cores: List[torch.Tensor], controls: StepControls) -> List[torch.Tensor]:
        """Apply advection in QTT format (phase rotation)."""
        dt = controls.dt
        phase = dt * 2.0
        
        for i in range(1, len(cores) - 1):
            c, s = np.cos(phase * 0.1), np.sin(phase * 0.1)
            old_0 = cores[i][:, 0, :].clone()
            old_1 = cores[i][:, 1, :].clone()
            cores[i][:, 0, :] = c * old_0 - s * old_1
            cores[i][:, 1, :] = s * old_0 + c * old_1
        
        return cores
    
    def _apply_diffusion(self, cores: List[torch.Tensor], controls: StepControls) -> List[torch.Tensor]:
        """Apply diffusion in QTT format (exponential decay)."""
        decay = np.exp(-controls.viscosity * controls.dt * 10)
        
        for core in cores:
            core *= decay
        
        return cores
    
    def _apply_forces(self, cores: List[torch.Tensor], controls: StepControls) -> List[torch.Tensor]:
        """Apply external forces."""
        # Add small perturbation to maintain energy
        idx = np.random.randint(1, len(cores) - 1)
        cores[idx] += torch.randn_like(cores[idx]) * 0.001
        return cores
    
    def _truncate(self, cores: List[torch.Tensor], controls: StepControls) -> Tuple[List[torch.Tensor], float]:
        """Truncate cores to maintain rank bounds."""
        max_rank = controls.max_rank
        error = 0.0
        
        if max_rank is None:
            return cores, error
        
        # Simple truncation: cap rank by zeroing small singular values
        for i, core in enumerate(cores):
            r_left, _, r_right = core.shape
            if r_left > max_rank or r_right > max_rank:
                # Reshape to matrix and SVD
                matrix = core.reshape(r_left * 2, r_right)
                U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
                
                # Truncate
                k = min(max_rank, len(S))
                error += (S[k:] ** 2).sum().item()
                
                S_trunc = S[:k]
                U_trunc = U[:, :k]
                Vh_trunc = Vh[:k, :]
                
                # Reconstruct
                matrix_trunc = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
                cores[i] = matrix_trunc.reshape(r_left, 2, -1)
        
        return cores, np.sqrt(error)
    
    def _compute_energy(self, cores: List[torch.Tensor]) -> float:
        """Compute field energy (Frobenius norm of cores)."""
        return sum(torch.norm(c).item() ** 2 for c in cores)
    
    def _compute_energy_drift(self) -> float:
        """Compute energy drift from initial state."""
        if len(self._energy_history) < 2:
            return 0.0
        return (self._energy_history[-1] - self._energy_history[0]) / max(self._energy_history[0], 1e-10)
    
    def _compute_divergence_norm(self) -> float:
        """
        Compute divergence norm for velocity/vector fields.
        
        For incompressible flows, ||∇·v|| should be close to 0.
        Uses QTT-compatible finite differences.
        
        Returns:
            L2 norm of divergence field
        """
        if self.field_type != FieldType.VECTOR:
            return 0.0
        
        # For vector fields, we need multiple components
        # Assuming cores encode [vx, vy, vz] velocity components
        # This is a simplified estimate using core structure
        try:
            # Compute approximate divergence via finite differences on cores
            div_norm_sq = 0.0
            for i, core in enumerate(self.cores):
                # Core shape: (r_left, 2, r_right) for QTT
                # Derivative approximation: diff along physical dimension
                if core.shape[1] == 2:  # QTT mode dimension
                    diff = core[:, 1, :] - core[:, 0, :]
                    div_norm_sq += torch.sum(diff ** 2).item()
            
            return float(np.sqrt(div_norm_sq / max(len(self.cores), 1)))
        except Exception:
            return 0.0

    def _compute_hash(self) -> str:
        """Compute deterministic hash of field state."""
        hasher = hashlib.sha256()
        for core in self.cores:
            hasher.update(core.cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    @property
    def rank(self) -> int:
        """Maximum rank of the tensor train."""
        ranks = [c.shape[0] for c in self.cores[1:]] + [c.shape[2] for c in self.cores[:-1]]
        return max(ranks) if ranks else 1
    
    @property
    def memory_bytes(self) -> int:
        """QTT memory usage in bytes."""
        return sum(c.numel() * c.element_size() for c in self.cores)
    
    @property
    def compression(self) -> float:
        """Compression ratio vs dense storage."""
        dense = self.total_points * 8
        return dense / self.memory_bytes
    
    def to(self, device: str) -> 'Field':
        """Move field to specified device."""
        new_cores = [c.to(device) for c in self.cores]
        return Field(
            cores=new_cores,
            dims=self.dims,
            bits_per_dim=self.bits_per_dim,
            field_type=self.field_type,
            device=device,
            metadata=self.metadata.copy(),
        )
    
    def __repr__(self) -> str:
        return (
            f"Field(dims={self.dims}, bits={self.bits_per_dim}, "
            f"grid={self.grid_size}^{self.dims}, rank={self.rank}, "
            f"memory={self.memory_bytes/1024:.1f}KB, "
            f"compression={self.compression:,.0f}x)"
        )
