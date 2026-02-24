"""QTT Physics VM — Intermediate Representation.

Defines the operator bytecode that all physics domains compile into.
Every PDE, regardless of origin (fluids, EM, quantum, kinetic, diffusion),
becomes a sequence of these instructions executed on the same runtime.

The IR is register-based: instructions read from and write to numbered
registers, each holding a QTT tensor (list[NDArray] cores).

Opcodes
-------
Data movement : LOAD_FIELD, STORE_FIELD, LOAD_CONST, COPY
Arithmetic    : ADD, SUB, SCALE, NEGATE
Differential  : GRAD, LAPLACE, DIV, CURL
Nonlinear     : HADAMARD, ADVECT
QTT control   : TRUNCATE, CANONICALIZE
Boundaries    : BC_APPLY
Solvers       : LAPLACE_SOLVE
Integration   : INTEGRATE
Telemetry     : MEASURE
Control flow  : LOOP_START, LOOP_END
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class OpCode(enum.Enum):
    """Operator bytecode instructions for the QTT Physics VM."""

    # ── Data movement ────────────────────────────────────────────────
    LOAD_FIELD = "load_field"
    STORE_FIELD = "store_field"
    LOAD_CONST = "load_const"
    COPY = "copy"

    # ── Arithmetic (linear, rank-preserving or additive) ─────────────
    ADD = "add"
    SUB = "sub"
    SCALE = "scale"
    NEGATE = "negate"

    # ── Differential operators (MPO application) ─────────────────────
    GRAD = "grad"
    LAPLACE = "laplace"
    DIV = "div"
    CURL = "curl"

    # ── Nonlinear (Hadamard / rank-multiplicative) ───────────────────
    HADAMARD = "hadamard"
    ADVECT = "advect"

    # ── QTT rank control ─────────────────────────────────────────────
    TRUNCATE = "truncate"
    CANONICALIZE = "canonicalize"

    # ── Boundary conditions ──────────────────────────────────────────
    BC_APPLY = "bc_apply"

    # ── Solver primitives ────────────────────────────────────────────
    LAPLACE_SOLVE = "laplace_solve"

    # ── Reduction ────────────────────────────────────────────────────
    INTEGRATE = "integrate"

    # ── Telemetry ────────────────────────────────────────────────────
    MEASURE = "measure"

    # ── Control flow ─────────────────────────────────────────────────
    LOOP_START = "loop_start"
    LOOP_END = "loop_end"


class BCKind(enum.Enum):
    """Boundary condition types."""
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ABSORBING = "absorbing"


@dataclass(frozen=True)
class Instruction:
    """A single VM instruction.

    Parameters
    ----------
    opcode : OpCode
        The operation to perform.
    dst : int
        Destination register index (-1 if no destination).
    src : tuple[int, ...]
        Source register indices.
    params : dict[str, Any]
        Opcode-specific parameters (dimension, scalar value, BC type, etc.).
    """
    opcode: OpCode
    dst: int = -1
    src: tuple[int, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [f"{self.opcode.value}"]
        if self.dst >= 0:
            parts.append(f"r{self.dst}")
        for s in self.src:
            parts.append(f"r{s}")
        if self.params:
            pstr = ", ".join(f"{k}={v}" for k, v in self.params.items())
            parts.append(f"({pstr})")
        return " ".join(parts)


@dataclass(frozen=True)
class FieldSpec:
    """Specification for a named field in the simulation.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. "u", "E", "psi_re").
    n_dims : int
        Number of spatial dimensions (1, 2, 3).
    bits_per_dim : tuple[int, ...]
        QTT bits per spatial dimension. Total cores = sum(bits_per_dim).
    bc : BCKind
        Boundary condition type.
    bc_params : dict[str, Any]
        BC-specific values (e.g. left/right Dirichlet values).
    initial_fn : str
        Name of the initial condition function (resolved by compiler).
    conserved_quantity : str | None
        Description of the conserved integral for this field (if any).
    """
    name: str
    n_dims: int = 1
    bits_per_dim: tuple[int, ...] = (8,)
    bc: BCKind = BCKind.PERIODIC
    bc_params: dict[str, Any] = field(default_factory=dict)
    initial_fn: str = ""
    conserved_quantity: str | None = None


@dataclass(frozen=True)
class Program:
    """A compiled QTT Physics VM program.

    Fully describes the simulation: domain, discretization, initial
    conditions, boundary conditions, time integration, and the IR
    instruction sequence.

    Parameters
    ----------
    domain : str
        Physics domain name (e.g. "burgers", "maxwell").
    domain_label : str
        Human-readable domain description.
    n_registers : int
        Number of QTT registers required.
    fields : dict[str, FieldSpec]
        Named field specifications (initial conditions, BCs, etc.).
    instructions : list[Instruction]
        The compiled IR instruction sequence (one time step body).
    dt : float
        Time step size.
    n_steps : int
        Number of time steps to execute.
    params : dict[str, float]
        Domain-specific physical parameters (viscosity, wave speed, etc.).
    metadata : dict[str, Any]
        Additional metadata (equations, notes, etc.).
    """
    domain: str
    domain_label: str
    n_registers: int
    fields: dict[str, FieldSpec]
    instructions: list[Instruction]
    dt: float
    n_steps: int
    params: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_bits(self) -> int:
        """Primary grid resolution (bits) from the first field."""
        for spec in self.fields.values():
            return spec.bits_per_dim[0]
        raise ValueError("Program has no fields")

    @property
    def total_cores(self) -> int:
        """Total QTT cores for the primary field."""
        for spec in self.fields.values():
            return sum(spec.bits_per_dim)
        raise ValueError("Program has no fields")


# ── Instruction builders (convenience API for compilers) ─────────────

def load_field(dst: int, name: str) -> Instruction:
    """Load a named field into a register."""
    return Instruction(OpCode.LOAD_FIELD, dst=dst, params={"name": name})


def store_field(src: int, name: str) -> Instruction:
    """Store a register into a named field."""
    return Instruction(OpCode.STORE_FIELD, src=(src,), params={"name": name})


def load_const(dst: int, value: float) -> Instruction:
    """Load a scalar constant as a rank-1 QTT tensor."""
    return Instruction(OpCode.LOAD_CONST, dst=dst, params={"value": value})


def copy(dst: int, src: int) -> Instruction:
    """Copy a register."""
    return Instruction(OpCode.COPY, dst=dst, src=(src,))


def add(dst: int, a: int, b: int) -> Instruction:
    """dst = a + b."""
    return Instruction(OpCode.ADD, dst=dst, src=(a, b))


def sub(dst: int, a: int, b: int) -> Instruction:
    """dst = a - b."""
    return Instruction(OpCode.SUB, dst=dst, src=(a, b))


def scale(dst: int, src: int, alpha: float) -> Instruction:
    """dst = alpha * src."""
    return Instruction(OpCode.SCALE, dst=dst, src=(src,), params={"alpha": alpha})


def negate(dst: int, src: int) -> Instruction:
    """dst = -src."""
    return Instruction(OpCode.NEGATE, dst=dst, src=(src,))


def grad(dst: int, src: int, dim: int = 0) -> Instruction:
    """dst = ∂(src)/∂x_dim."""
    return Instruction(OpCode.GRAD, dst=dst, src=(src,), params={"dim": dim})


def laplace(dst: int, src: int, dim: int | None = None) -> Instruction:
    """dst = ∇²(src).  If dim is None, sum over all dimensions."""
    return Instruction(OpCode.LAPLACE, dst=dst, src=(src,), params={"dim": dim})


def div(dst: int, *component_regs: int) -> Instruction:
    """dst = ∇·(components), divergence of a vector field."""
    return Instruction(OpCode.DIV, dst=dst, src=tuple(component_regs))


def curl(dst_x: int, dst_y: int, dst_z: int,
         src_x: int, src_y: int, src_z: int) -> Instruction:
    """Curl of a 3D vector field (rarely used in 1D benchmark)."""
    return Instruction(
        OpCode.CURL, dst=dst_x,
        src=(src_x, src_y, src_z),
        params={"dst_y": dst_y, "dst_z": dst_z},
    )


def hadamard(dst: int, a: int, b: int) -> Instruction:
    """dst = a ⊙ b (pointwise / Hadamard product)."""
    return Instruction(OpCode.HADAMARD, dst=dst, src=(a, b))


def advect(dst: int, velocity: int, field_reg: int, dim: int = 0) -> Instruction:
    """dst = (velocity · ∇)(field) along dimension dim."""
    return Instruction(OpCode.ADVECT, dst=dst, src=(velocity, field_reg),
                       params={"dim": dim})


def truncate(reg: int) -> Instruction:
    """Truncate rank of register in-place."""
    return Instruction(OpCode.TRUNCATE, dst=reg, src=(reg,))


def canonicalize(reg: int, direction: str = "left") -> Instruction:
    """Put register in canonical form."""
    return Instruction(OpCode.CANONICALIZE, dst=reg, src=(reg,),
                       params={"direction": direction})


def bc_apply(reg: int, kind: BCKind,
             bc_params: dict[str, Any] | None = None) -> Instruction:
    """Apply boundary conditions to register."""
    p: dict[str, Any] = {"kind": kind}
    if bc_params:
        p["bc_params"] = bc_params
    return Instruction(OpCode.BC_APPLY, dst=reg, src=(reg,), params=p)


def laplace_solve(dst: int, rhs: int, dim: int | None = None) -> Instruction:
    """dst = ∇⁻²(rhs), Poisson solve."""
    return Instruction(OpCode.LAPLACE_SOLVE, dst=dst, src=(rhs,),
                       params={"dim": dim})


def integrate(dst: int, src: int, dim: int) -> Instruction:
    """dst = ∫ src d(x_dim), partial integration along one dimension."""
    return Instruction(OpCode.INTEGRATE, dst=dst, src=(src,),
                       params={"dim": dim})


def measure(reg: int, label: str = "") -> Instruction:
    """Record telemetry for a register."""
    return Instruction(OpCode.MEASURE, src=(reg,),
                       params={"label": label})


def loop_start(n_steps: int) -> Instruction:
    """Begin time-step loop."""
    return Instruction(OpCode.LOOP_START, params={"n_steps": n_steps})


def loop_end() -> Instruction:
    """End time-step loop."""
    return Instruction(OpCode.LOOP_END)
