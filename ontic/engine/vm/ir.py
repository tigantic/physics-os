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

    # ── Material / masking ───────────────────────────────────────────
    MASK_MULTIPLY = "mask_multiply"  # dst = mask_field ⊙ src (material scaling)
    SOURCE_ADD = "source_add"        # dst += source_field (current injection)
    DFT_ACCUMULATE = "dft_accumulate" # DFT bin accumulation for freq-domain
    PROBE_RECORD = "probe_record"    # record scalar at probe point each step

    # ── Control flow ─────────────────────────────────────────────────
    LOOP_START = "loop_start"
    LOOP_END = "loop_end"


class BCKind(enum.Enum):
    """Boundary condition types."""
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ABSORBING = "absorbing"
    PEC = "pec"  # Perfect Electric Conductor: E_tangential → 0 at boundary


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


def grad(dst: int, src: int, dim: int = 0,
         operator_variant: str = "grad_v1") -> Instruction:
    """dst = ∂(src)/∂x_dim.

    Parameters
    ----------
    operator_variant : str
        MPO variant tag: ``"grad_v1"`` (2nd order) or
        ``"grad_v2_high_order"`` (4th order).
    """
    params: dict[str, Any] = {"dim": dim}
    if operator_variant != "grad_v1":
        params["operator_variant"] = operator_variant
    return Instruction(OpCode.GRAD, dst=dst, src=(src,), params=params)


def laplace(dst: int, src: int, dim: int | None = None,
            operator_variant: str = "lap_v1") -> Instruction:
    """dst = ∇²(src).  If dim is None, sum over all dimensions.

    Parameters
    ----------
    operator_variant : str
        MPO variant tag: ``"lap_v1"`` (2nd order) or
        ``"lap_v2_high_order"`` (4th order).
    """
    params: dict[str, Any] = {"dim": dim}
    if operator_variant != "lap_v1":
        params["operator_variant"] = operator_variant
    return Instruction(OpCode.LAPLACE, dst=dst, src=(src,), params=params)


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


def laplace_solve(
    dst: int,
    rhs: int,
    dim: int | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
) -> Instruction:
    """dst = ∇⁻²(rhs), Poisson solve.

    Parameters
    ----------
    tol : float, optional
        CG convergence tolerance.  If None, the runtime uses its default.
    max_iter : int, optional
        Maximum CG iterations.  If None, the runtime uses its default.
    """
    params: dict[str, Any] = {"dim": dim}
    if tol is not None:
        params["poisson_tol"] = tol
    if max_iter is not None:
        params["poisson_max_iter"] = max_iter
    return Instruction(OpCode.LAPLACE_SOLVE, dst=dst, src=(rhs,),
                       params=params)


def integrate(dst: int, src: int, dim: int) -> Instruction:
    """dst = ∫ src d(x_dim), partial integration along one dimension."""
    return Instruction(OpCode.INTEGRATE, dst=dst, src=(src,),
                       params={"dim": dim})


def measure(reg: int, label: str = "") -> Instruction:
    """Record telemetry for a register."""
    return Instruction(OpCode.MEASURE, src=(reg,),
                       params={"label": label})


def mask_multiply(dst: int, mask: int, src: int) -> Instruction:
    """dst = mask ⊙ src (material / geometry masking via Hadamard)."""
    return Instruction(OpCode.MASK_MULTIPLY, dst=dst, src=(mask, src))


def source_add(
    dst: int,
    source: int,
    *,
    freq_center: float = 0.0,
    bandwidth: float = 0.0,
    t_peak: float = 0.0,
) -> Instruction:
    """dst += source * temporal_amplitude(step).

    When *freq_center* > 0 the runtime applies a Gaussian-modulated
    sinusoidal envelope:

        τ  = 1 / (2π · bandwidth)  (if bandwidth > 0, else very long)
        a(t) = sin(2π f₀ t) · exp(−(t − t_peak)² / 2τ²)

    The source register holds the *spatial* profile; the runtime
    multiplies by a(t) at each step.
    """
    params: dict[str, Any] = {}
    if freq_center > 0.0:
        params["freq_center"] = freq_center
        params["bandwidth"] = bandwidth
        params["t_peak"] = t_peak
    return Instruction(OpCode.SOURCE_ADD, dst=dst, src=(source,), params=params)


def dft_accumulate(
    dst: int,
    src: int,
    freq_bin: int = 0,
    component: str = "real",
    omega: float | None = None,
) -> Instruction:
    """Accumulate DFT bin for frequency-domain extraction.

    *component* is ``"real"`` (cos weight) or ``"imag"`` (sin weight).

    If *omega* (angular frequency, rad/time-unit) is provided, the DFT
    phase is computed as ``omega * step * dt`` using the physical
    frequency.  Otherwise the legacy bin-based formula
    ``2π * freq_bin * step / n_steps`` is used.
    """
    params: dict[str, object] = {
        "freq_bin": freq_bin,
        "component": component,
    }
    if omega is not None:
        params["omega"] = omega
    return Instruction(
        OpCode.DFT_ACCUMULATE,
        dst=dst,
        src=(src,),
        params=params,
    )


def probe_record(
    src: int,
    probe_name: str,
    coords: tuple[float, ...],
) -> Instruction:
    """Record field value at a probe point for time-domain extraction.

    The runtime evaluates the QTT tensor in register *src* at the
    physical *coords* and appends the scalar to a named time series.
    Cost: O(n_cores × r²) per evaluation — negligible vs. MPO ops.

    Parameters
    ----------
    src : int
        Source register containing the field to probe.
    probe_name : str
        Unique identifier for this probe (e.g. ``"V_z"``).  Time series
        are keyed by this name in the execution result.
    coords : tuple[float, ...]
        Physical coordinates at which to evaluate the field.
    """
    return Instruction(
        OpCode.PROBE_RECORD,
        src=(src,),
        params={"probe_name": probe_name, "coords": coords},
    )


def loop_start(n_steps: int) -> Instruction:
    """Begin time-step loop."""
    return Instruction(OpCode.LOOP_START, params={"n_steps": n_steps})


def loop_end() -> Instruction:
    """End time-step loop."""
    return Instruction(OpCode.LOOP_END)
