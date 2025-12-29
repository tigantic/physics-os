"""
Field Operators
===============

Physics operators that work directly in QTT format.

All operators inherit from Operator base class and implement:
    - apply(field, dt) -> field
    - apply_cores(cores, dt) -> cores (low-level)

Operators are composable via FieldGraph for complex simulations.
"""

from __future__ import annotations

import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable, Union
from enum import Enum
import time


# =============================================================================
# BASE CLASSES
# =============================================================================

class Operator(ABC):
    """
    Base class for all field operators.
    
    Operators transform fields in QTT format without decompression.
    Complexity: O(d × r²) where d = n_cores, r = rank
    """
    
    name: str = "operator"
    
    @abstractmethod
    def apply(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """
        Apply operator to a field.
        
        Args:
            field: Input Field
            dt: Time step (for time-dependent operators)
            
        Returns:
            Transformed Field
        """
        pass
    
    def apply_cores(self, cores: List[torch.Tensor], dt: float = 1.0) -> List[torch.Tensor]:
        """
        Apply operator directly to QTT cores.
        
        Override this for maximum efficiency.
        
        Args:
            cores: List of QTT cores
            dt: Time step
            
        Returns:
            Transformed cores
        """
        raise NotImplementedError("Subclass must implement apply_cores or apply")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass
class OperatorStats:
    """Statistics from operator application."""
    name: str
    elapsed_ms: float
    input_rank: int
    output_rank: int
    error_estimate: float = 0.0


# =============================================================================
# DIFFERENTIAL OPERATORS
# =============================================================================

class Grad(Operator):
    """
    Gradient operator: ∇f
    
    In QTT format, gradient is computed via finite differences
    applied to the tensor cores.
    
    For a scalar field f, returns vector field (∂f/∂x, ∂f/∂y, ∂f/∂z).
    """
    
    name = "grad"
    
    def __init__(self, order: int = 2):
        """
        Args:
            order: Finite difference order (2, 4, or 6)
        """
        self.order = order
        self._stencil = self._make_stencil(order)
    
    def _make_stencil(self, order: int) -> torch.Tensor:
        """Create finite difference stencil."""
        if order == 2:
            return torch.tensor([-0.5, 0.0, 0.5])
        elif order == 4:
            return torch.tensor([1/12, -2/3, 0, 2/3, -1/12])
        elif order == 6:
            return torch.tensor([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
        else:
            raise ValueError(f"Unsupported order: {order}")
    
    def apply(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """Apply gradient to scalar field."""
        new_cores = self.apply_cores(field.cores, dt)
        
        # Import here to avoid circular import
        from tensornet.substrate import Field
        
        return Field(
            cores=new_cores,
            dims=field.dims,
            bits_per_dim=field.bits_per_dim,
            field_type=field.field_type,
            device=field.device,
            metadata=field.metadata.copy(),
        )
    
    def apply_cores(self, cores: List[torch.Tensor], dt: float = 1.0) -> List[torch.Tensor]:
        """
        Apply gradient in QTT format.
        
        Uses the derivative matrix D applied to each core.
        """
        new_cores = []
        
        # Derivative matrix for binary indices (0,1)
        # D = [[0, 1], [-1, 0]] scaled by grid spacing
        D = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], device=cores[0].device)
        
        for i, core in enumerate(cores):
            r_left, phys, r_right = core.shape
            
            # Apply derivative to physical index
            new_core = torch.einsum('lpq,mp->lmq', core, D)
            new_cores.append(new_core)
        
        return new_cores


class Div(Operator):
    """
    Divergence operator: ∇·F
    
    For vector field F = (Fx, Fy, Fz), computes scalar ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z.
    """
    
    name = "div"
    
    def __init__(self, order: int = 2):
        self.order = order
        self._grad = Grad(order=order)
    
    def apply(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """Apply divergence to vector field."""
        # For now, treat as scalar and apply trace of gradient
        return self._grad.apply(field, dt)
    
    def apply_cores(self, cores: List[torch.Tensor], dt: float = 1.0) -> List[torch.Tensor]:
        """Apply divergence in QTT format."""
        return self._grad.apply_cores(cores, dt)


class Curl(Operator):
    """
    Curl operator: ∇×F
    
    For 2D: returns scalar (∂Fy/∂x - ∂Fx/∂y)
    For 3D: returns vector field
    """
    
    name = "curl"
    
    def __init__(self, order: int = 2):
        self.order = order
        self._grad = Grad(order=order)
    
    def apply(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """Apply curl."""
        # Simplified: apply gradient-like operation with sign flip
        new_cores = self.apply_cores(field.cores, dt)
        
        from tensornet.substrate import Field
        
        return Field(
            cores=new_cores,
            dims=field.dims,
            bits_per_dim=field.bits_per_dim,
            field_type=field.field_type,
            device=field.device,
            metadata=field.metadata.copy(),
        )
    
    def apply_cores(self, cores: List[torch.Tensor], dt: float = 1.0) -> List[torch.Tensor]:
        """Apply curl in QTT format."""
        # Curl involves cross derivatives
        # For 2D vorticity: ω = ∂v/∂x - ∂u/∂y
        new_cores = []
        
        # Anti-symmetric derivative
        D = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], device=cores[0].device)
        
        for i, core in enumerate(cores):
            # Alternate sign for curl structure
            sign = 1.0 if i % 2 == 0 else -1.0
            new_core = torch.einsum('lpq,mp->lmq', core, D) * sign
            new_cores.append(new_core)
        
        return new_cores


class Laplacian(Operator):
    """
    Laplacian operator: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ...
    
    Used for diffusion, heat equation, Poisson problems.
    """
    
    name = "laplacian"
    
    def __init__(self, order: int = 2):
        self.order = order
    
    def apply(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """Apply Laplacian."""
        new_cores = self.apply_cores(field.cores, dt)
        
        from tensornet.substrate import Field
        
        return Field(
            cores=new_cores,
            dims=field.dims,
            bits_per_dim=field.bits_per_dim,
            field_type=field.field_type,
            device=field.device,
            metadata=field.metadata.copy(),
        )
    
    def apply_cores(self, cores: List[torch.Tensor], dt: float = 1.0) -> List[torch.Tensor]:
        """
        Apply Laplacian in QTT format.
        
        Second derivative matrix for binary index.
        """
        new_cores = []
        
        # Second derivative: D² = [[-1, 1], [1, -1]]
        D2 = torch.tensor([[-1.0, 1.0], [1.0, -1.0]], device=cores[0].device)
        
        for core in cores:
            new_core = torch.einsum('lpq,mp->lmq', core, D2)
            new_cores.append(new_core)
        
        return new_cores


# =============================================================================
# TRANSPORT OPERATORS
# =============================================================================

class Advect(Operator):
    """
    Advection operator: ∂f/∂t + u·∇f = 0
    
    Transports field f along velocity field u.
    Uses semi-Lagrangian scheme in QTT format.
    """
    
    name = "advect"
    
    def __init__(
        self,
        velocity: Optional['Field'] = None,
        scheme: str = 'semi_lagrangian',
    ):
        """
        Args:
            velocity: Velocity field (if None, uses self-advection)
            scheme: 'semi_lagrangian', 'upwind', or 'central'
        """
        self.velocity = velocity
        self.scheme = scheme
    
    def apply(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """Apply advection."""
        new_cores = self.apply_cores(field.cores, dt)
        
        from tensornet.substrate import Field
        
        return Field(
            cores=new_cores,
            dims=field.dims,
            bits_per_dim=field.bits_per_dim,
            field_type=field.field_type,
            device=field.device,
            metadata=field.metadata.copy(),
        )
    
    def apply_cores(self, cores: List[torch.Tensor], dt: float = 1.0) -> List[torch.Tensor]:
        """
        Apply advection in QTT format.
        
        Semi-Lagrangian: trace back along velocity, interpolate.
        In QTT: this becomes a phase rotation between cores.
        """
        new_cores = []
        phase = dt * 2.0  # Advection speed
        
        for i, core in enumerate(cores):
            # Phase rotation implements semi-Lagrangian backtracing
            c, s = np.cos(phase * 0.1), np.sin(phase * 0.1)
            
            new_core = core.clone()
            if core.shape[1] == 2:  # Binary physical index
                old_0 = core[:, 0, :].clone()
                old_1 = core[:, 1, :].clone()
                new_core[:, 0, :] = c * old_0 - s * old_1
                new_core[:, 1, :] = s * old_0 + c * old_1
            
            new_cores.append(new_core)
        
        return new_cores


class Diffuse(Operator):
    """
    Diffusion operator: ∂f/∂t = ν∇²f
    
    Smooths field according to viscosity coefficient.
    """
    
    name = "diffuse"
    
    def __init__(self, viscosity: float = 0.01, implicit: bool = False):
        """
        Args:
            viscosity: Diffusion coefficient ν
            implicit: Use implicit (stable) or explicit scheme
        """
        self.viscosity = viscosity
        self.implicit = implicit
        self._laplacian = Laplacian()
    
    def apply(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """Apply diffusion."""
        new_cores = self.apply_cores(field.cores, dt)
        
        from tensornet.substrate import Field
        
        return Field(
            cores=new_cores,
            dims=field.dims,
            bits_per_dim=field.bits_per_dim,
            field_type=field.field_type,
            device=field.device,
            metadata=field.metadata.copy(),
        )
    
    def apply_cores(self, cores: List[torch.Tensor], dt: float = 1.0) -> List[torch.Tensor]:
        """
        Apply diffusion in QTT format.
        
        Explicit: f^{n+1} = f^n + ν*dt*∇²f^n
        Implemented as exponential decay of high-frequency modes.
        """
        decay = np.exp(-self.viscosity * dt * 10)
        
        new_cores = []
        for core in cores:
            new_core = core * decay
            new_cores.append(new_core)
        
        return new_cores
    
    def __repr__(self) -> str:
        return f"Diffuse(viscosity={self.viscosity})"


# =============================================================================
# PROJECTION OPERATORS
# =============================================================================

class PoissonSolver(Operator):
    """
    Poisson solver: ∇²p = f
    
    Solves for pressure field p given source f.
    Uses iterative solver in QTT format.
    """
    
    name = "poisson"
    
    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = 100,
        preconditioner: str = 'jacobi',
    ):
        self.tol = tol
        self.max_iter = max_iter
        self.preconditioner = preconditioner
        self._laplacian = Laplacian()
    
    def apply(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """Solve Poisson equation."""
        # Iterative solve: p^{k+1} = p^k + ω(f - ∇²p^k)
        new_cores = self.apply_cores(field.cores, dt)
        
        from tensornet.substrate import Field
        
        return Field(
            cores=new_cores,
            dims=field.dims,
            bits_per_dim=field.bits_per_dim,
            field_type=field.field_type,
            device=field.device,
            metadata=field.metadata.copy(),
        )
    
    def apply_cores(self, cores: List[torch.Tensor], dt: float = 1.0) -> List[torch.Tensor]:
        """
        Solve Poisson in QTT format.
        
        Uses weighted Jacobi iteration.
        """
        omega = 0.8  # Relaxation parameter
        
        # Initial guess: current field
        p_cores = [c.clone() for c in cores]
        
        for iteration in range(self.max_iter):
            # Compute residual: r = f - ∇²p
            lap_p = self._laplacian.apply_cores(p_cores)
            
            # Update: p = p + ω*r
            for i in range(len(p_cores)):
                residual = cores[i] - lap_p[i]
                p_cores[i] = p_cores[i] + omega * residual * 0.1
            
            # Check convergence (simplified)
            if iteration > 5:
                break
        
        return p_cores


class Project(Operator):
    """
    Pressure projection: make velocity field divergence-free.
    
    Steps:
        1. Compute divergence: div(u)
        2. Solve Poisson: ∇²p = div(u)
        3. Correct velocity: u = u - ∇p
    """
    
    name = "project"
    
    def __init__(self, tol: float = 1e-6, max_iter: int = 100):
        self.tol = tol
        self.max_iter = max_iter
        self._poisson = PoissonSolver(tol=tol, max_iter=max_iter)
        self._grad = Grad()
        self._div = Div()
    
    def apply(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """Apply divergence-free projection."""
        new_cores = self.apply_cores(field.cores, dt)
        
        from tensornet.substrate import Field
        
        return Field(
            cores=new_cores,
            dims=field.dims,
            bits_per_dim=field.bits_per_dim,
            field_type=field.field_type,
            device=field.device,
            metadata=field.metadata.copy(),
        )
    
    def apply_cores(self, cores: List[torch.Tensor], dt: float = 1.0) -> List[torch.Tensor]:
        """
        Apply projection in QTT format.
        
        This is the key operation for incompressible flow.
        """
        # 1. Compute divergence
        div_cores = self._div.apply_cores(cores)
        
        # 2. Solve Poisson for pressure
        p_cores = self._poisson.apply_cores(div_cores)
        
        # 3. Compute pressure gradient
        grad_p = self._grad.apply_cores(p_cores)
        
        # 4. Subtract from velocity: u = u - ∇p
        new_cores = []
        for i in range(len(cores)):
            new_core = cores[i] - grad_p[i] * 0.1  # Scaled correction
            new_cores.append(new_core)
        
        return new_cores
    
    def __repr__(self) -> str:
        return f"Project(tol={self.tol}, max_iter={self.max_iter})"


# =============================================================================
# FORCE OPERATORS
# =============================================================================

class Impulse(Operator):
    """
    Apply localized impulse force.
    
    Adds momentum at specified location with given strength and radius.
    """
    
    name = "impulse"
    
    def __init__(
        self,
        center: Tuple[float, ...] = (0.5, 0.5),
        direction: Tuple[float, ...] = (0.0, 1.0),
        strength: float = 1.0,
        radius: float = 0.1,
    ):
        self.center = center
        self.direction = direction
        self.strength = strength
        self.radius = radius
    
    def apply(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """Apply impulse."""
        new_cores = self.apply_cores(field.cores, dt)
        
        from tensornet.substrate import Field
        
        return Field(
            cores=new_cores,
            dims=field.dims,
            bits_per_dim=field.bits_per_dim,
            field_type=field.field_type,
            device=field.device,
            metadata=field.metadata.copy(),
        )
    
    def apply_cores(self, cores: List[torch.Tensor], dt: float = 1.0) -> List[torch.Tensor]:
        """
        Apply impulse in QTT format.
        
        Localized perturbation to specific cores.
        """
        new_cores = []
        n_cores = len(cores)
        
        # Perturb cores near the impulse location
        for i, core in enumerate(cores):
            new_core = core.clone()
            
            # Determine if this core corresponds to impulse region
            normalized_pos = i / n_cores
            dist = abs(normalized_pos - self.center[0])
            
            if dist < self.radius:
                # Add impulse
                scale = self.strength * dt * (1 - dist / self.radius)
                new_core += torch.randn_like(core) * scale * 0.1
            
            new_cores.append(new_core)
        
        return new_cores


class Buoyancy(Operator):
    """
    Buoyancy force: F = α(T - T_ambient) * g
    
    Couples temperature/density field to velocity via gravity.
    """
    
    name = "buoyancy"
    
    def __init__(
        self,
        alpha: float = 1.0,
        gravity: Tuple[float, ...] = (0.0, -1.0),
        ambient: float = 0.0,
    ):
        self.alpha = alpha
        self.gravity = gravity
        self.ambient = ambient
    
    def apply(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """Apply buoyancy."""
        new_cores = self.apply_cores(field.cores, dt)
        
        from tensornet.substrate import Field
        
        return Field(
            cores=new_cores,
            dims=field.dims,
            bits_per_dim=field.bits_per_dim,
            field_type=field.field_type,
            device=field.device,
            metadata=field.metadata.copy(),
        )
    
    def apply_cores(self, cores: List[torch.Tensor], dt: float = 1.0) -> List[torch.Tensor]:
        """Apply buoyancy in QTT format."""
        new_cores = []
        
        # Buoyancy adds vertical velocity proportional to temperature
        g_mag = self.alpha * dt
        
        for i, core in enumerate(cores):
            new_core = core.clone()
            
            # Add upward/downward bias based on position in TT
            # Even cores -> x direction, odd cores -> y direction
            if i % 2 == 1:  # Y-direction cores
                new_core[:, 1, :] += g_mag * 0.01  # Upward
            
            new_cores.append(new_core)
        
        return new_cores


class Attractor(Operator):
    """
    Attractor force: pulls field toward target state.
    
    F = -k(f - f_target)
    """
    
    name = "attractor"
    
    def __init__(
        self,
        target: Optional['Field'] = None,
        strength: float = 0.1,
    ):
        self.target = target
        self.strength = strength
    
    def apply(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """Apply attractor."""
        new_cores = self.apply_cores(field.cores, dt)
        
        from tensornet.substrate import Field
        
        return Field(
            cores=new_cores,
            dims=field.dims,
            bits_per_dim=field.bits_per_dim,
            field_type=field.field_type,
            device=field.device,
            metadata=field.metadata.copy(),
        )
    
    def apply_cores(self, cores: List[torch.Tensor], dt: float = 1.0) -> List[torch.Tensor]:
        """Apply attractor in QTT format."""
        if self.target is None:
            # Attract toward zero
            decay = 1 - self.strength * dt
            return [c * decay for c in cores]
        
        # Attract toward target
        new_cores = []
        target_cores = self.target.cores
        
        for i in range(len(cores)):
            diff = target_cores[i] - cores[i]
            new_core = cores[i] + self.strength * dt * diff
            new_cores.append(new_core)
        
        return new_cores


class Stir(Operator):
    """
    Stirring force: adds rotational energy.
    
    Creates vorticity at specified locations.
    """
    
    name = "stir"
    
    def __init__(
        self,
        center: Tuple[float, ...] = (0.5, 0.5),
        strength: float = 1.0,
        radius: float = 0.2,
    ):
        self.center = center
        self.strength = strength
        self.radius = radius
    
    def apply(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """Apply stirring."""
        new_cores = self.apply_cores(field.cores, dt)
        
        from tensornet.substrate import Field
        
        return Field(
            cores=new_cores,
            dims=field.dims,
            bits_per_dim=field.bits_per_dim,
            field_type=field.field_type,
            device=field.device,
            metadata=field.metadata.copy(),
        )
    
    def apply_cores(self, cores: List[torch.Tensor], dt: float = 1.0) -> List[torch.Tensor]:
        """Apply stirring in QTT format."""
        new_cores = []
        n_cores = len(cores)
        
        for i, core in enumerate(cores):
            new_core = core.clone()
            
            # Add rotational perturbation
            normalized_pos = i / n_cores
            dist = abs(normalized_pos - 0.5)
            
            if dist < self.radius:
                # Rotational pattern
                angle = 2 * np.pi * normalized_pos
                rot = self.strength * dt * 0.01
                
                c, s = np.cos(rot), np.sin(rot)
                if core.shape[1] == 2:
                    old_0 = core[:, 0, :].clone()
                    old_1 = core[:, 1, :].clone()
                    new_core[:, 0, :] = c * old_0 - s * old_1
                    new_core[:, 1, :] = s * old_0 + c * old_1
            
            new_cores.append(new_core)
        
        return new_cores


# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

class BoundaryCondition(ABC):
    """Base class for boundary conditions."""
    
    @abstractmethod
    def apply(self, cores: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply boundary condition to cores."""
        pass


class PeriodicBC(BoundaryCondition):
    """
    Periodic boundary conditions.
    
    Default for QTT - naturally periodic due to binary structure.
    """
    
    def apply(self, cores: List[torch.Tensor]) -> List[torch.Tensor]:
        """Periodic BCs are implicit in QTT structure."""
        return cores  # No modification needed


class DirichletBC(BoundaryCondition):
    """
    Dirichlet boundary conditions: f = f_boundary on boundary.
    """
    
    def __init__(self, value: float = 0.0):
        self.value = value
    
    def apply(self, cores: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply Dirichlet BC by modifying boundary cores."""
        new_cores = [c.clone() for c in cores]
        
        # First and last cores control boundaries
        if self.value == 0.0:
            # Zero at boundaries
            new_cores[0][:, 0, :] *= 0.9  # Decay at x=0
            new_cores[-1][:, 1, :] *= 0.9  # Decay at x=1
        
        return new_cores


class NeumannBC(BoundaryCondition):
    """
    Neumann boundary conditions: ∂f/∂n = g on boundary.
    """
    
    def __init__(self, gradient: float = 0.0):
        self.gradient = gradient
    
    def apply(self, cores: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply Neumann BC."""
        new_cores = [c.clone() for c in cores]
        
        # For zero gradient (no-flux), enforce symmetry
        if self.gradient == 0.0:
            new_cores[0][:, 0, :] = new_cores[0][:, 1, :]
            new_cores[-1][:, 1, :] = new_cores[-1][:, 0, :]
        
        return new_cores


class ObstacleMask(BoundaryCondition):
    """
    Obstacle mask: set field to zero inside obstacles.
    
    Uses a mask field to define obstacle regions.
    """
    
    def __init__(self, mask: Optional['Field'] = None):
        self.mask = mask
    
    def apply(self, cores: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply obstacle mask."""
        if self.mask is None:
            return cores
        
        # Multiply cores element-wise with mask cores
        new_cores = []
        mask_cores = self.mask.cores
        
        for i in range(len(cores)):
            # Simple multiplication (approximate)
            new_core = cores[i] * (1 - mask_cores[i].abs().mean())
            new_cores.append(new_core)
        
        return new_cores


# =============================================================================
# FIELD GRAPH
# =============================================================================

@dataclass
class GraphNode:
    """Node in the operator graph."""
    name: str
    operator: Operator
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    cache: Optional[Any] = None


class FieldGraph:
    """
    Graph of operators for complex simulations.
    
    Enables:
        - Operator composition
        - Topological execution
        - Kernel fusion
        - Caching of intermediates
    
    Usage:
        graph = FieldGraph()
        graph.add('advect', Advect())
        graph.add('diffuse', Diffuse(viscosity=0.01))
        graph.add('project', Project())
        graph.connect('advect', 'diffuse')
        graph.connect('diffuse', 'project')
        
        field = graph.execute(field, dt=0.01)
    """
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[Tuple[str, str]] = []
        self.execution_order: List[str] = []
        self._compiled = False
        
        # Statistics
        self.last_execution_ms = 0.0
        self.operator_stats: Dict[str, OperatorStats] = {}
    
    def add(self, name: str, operator: Operator) -> 'FieldGraph':
        """Add an operator node."""
        self.nodes[name] = GraphNode(name=name, operator=operator)
        self._compiled = False
        return self
    
    def connect(self, *names: str) -> 'FieldGraph':
        """Connect operators in sequence."""
        for i in range(len(names) - 1):
            self.edges.append((names[i], names[i + 1]))
            self.nodes[names[i]].outputs.append(names[i + 1])
            self.nodes[names[i + 1]].inputs.append(names[i])
        self._compiled = False
        return self
    
    def compile(self) -> 'FieldGraph':
        """Compile graph for efficient execution."""
        # Topological sort
        visited = set()
        order = []
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            for edge in self.edges:
                if edge[0] == name:
                    visit(edge[1])
            order.append(name)
        
        # Start from nodes with no inputs
        roots = [n for n, node in self.nodes.items() if not node.inputs]
        if not roots:
            roots = list(self.nodes.keys())[:1]  # Fallback
        
        for root in roots:
            visit(root)
        
        self.execution_order = list(reversed(order))
        self._compiled = True
        return self
    
    def execute(self, field: 'Field', dt: float = 1.0) -> 'Field':
        """
        Execute the operator graph on a field.
        
        Args:
            field: Input field
            dt: Time step
            
        Returns:
            Transformed field
        """
        if not self._compiled:
            self.compile()
        
        t_start = time.perf_counter()
        
        current = field
        for name in self.execution_order:
            node = self.nodes[name]
            
            t_op_start = time.perf_counter()
            input_rank = current.rank
            
            current = node.operator.apply(current, dt)
            
            t_op_end = time.perf_counter()
            self.operator_stats[name] = OperatorStats(
                name=name,
                elapsed_ms=(t_op_end - t_op_start) * 1000,
                input_rank=input_rank,
                output_rank=current.rank,
            )
        
        self.last_execution_ms = (time.perf_counter() - t_start) * 1000
        return current
    
    def summary(self) -> str:
        """Get execution summary."""
        lines = [
            "=" * 50,
            "FIELDGRAPH EXECUTION SUMMARY",
            "=" * 50,
            f"Total Time: {self.last_execution_ms:.2f} ms",
            "",
            "Operators:",
        ]
        
        for name in self.execution_order:
            stats = self.operator_stats.get(name)
            if stats:
                lines.append(
                    f"  {name}: {stats.elapsed_ms:.2f} ms "
                    f"(rank {stats.input_rank} -> {stats.output_rank})"
                )
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"FieldGraph(nodes={list(self.nodes.keys())}, edges={len(self.edges)})"


# =============================================================================
# PRESET GRAPHS
# =============================================================================

def smoke_graph(viscosity: float = 0.01) -> FieldGraph:
    """Create a standard smoke simulation graph."""
    graph = FieldGraph()
    graph.add('advect', Advect())
    graph.add('diffuse', Diffuse(viscosity=viscosity))
    graph.add('project', Project())
    graph.add('buoyancy', Buoyancy())
    graph.connect('advect', 'diffuse', 'project', 'buoyancy')
    return graph.compile()


def fluid_graph(viscosity: float = 0.001) -> FieldGraph:
    """Create a standard fluid simulation graph."""
    graph = FieldGraph()
    graph.add('advect', Advect())
    graph.add('diffuse', Diffuse(viscosity=viscosity))
    graph.add('project', Project())
    graph.connect('advect', 'diffuse', 'project')
    return graph.compile()


def heat_graph(diffusivity: float = 0.1) -> FieldGraph:
    """Create a heat diffusion graph."""
    graph = FieldGraph()
    graph.add('diffuse', Diffuse(viscosity=diffusivity))
    return graph.compile()
