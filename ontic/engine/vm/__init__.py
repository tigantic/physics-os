"""QTT Physics VM — Universal compressed physics compute runtime.

Compiles different PDEs into the same operator bytecode and runs them
on one runtime.  If k=1 universality is real, then the backend is the
product — not the domain.

Architecture
------------
::

    Domain Equations          Operator IR           QTT Runtime
    ┌───────────────┐    ┌─────────────────┐    ┌──────────────┐
    │ Navier-Stokes │───▶│ grad, laplace,  │    │ ┌──────────┐ │
    │ Maxwell       │───▶│ hadamard, scale,│───▶│ │ Register │ │
    │ Schrödinger   │───▶│ truncate, bc,   │    │ │   File   │ │
    │ Vlasov-Poisson│───▶│ measure, ...    │    │ └──────────┘ │
    │ Adv-Diffusion │───▶│                 │    │ Rank Governor │
    └───────────────┘    └─────────────────┘    │ Telemetry    │
                                                 └──────────────┘

Usage
-----
Run the unified benchmark across all 5 physics domains::

    python -m ontic.vm.benchmark --n-bits 8 --n-steps 100

Programmatic::

    from ontic.engine.vm import QTTRuntime, RankGovernor
    from ontic.engine.vm.compilers import BurgersCompiler

    program = BurgersCompiler(n_bits=8, n_steps=100).compile()
    runtime = QTTRuntime()
    result = runtime.execute(program)
    print(result.telemetry.summary_line())
"""

from .ir import (
    BCKind,
    FieldSpec,
    Instruction,
    OpCode,
    Program,
)
from .qtt_tensor import QTTTensor
from .operators import OperatorCache, mpo_apply, poisson_solve
from .rank_governor import RankGovernor, TruncationPolicy
from .runtime import ExecutionResult, QTTRuntime
from .telemetry import ProgramTelemetry, StepTelemetry, TelemetryCollector
from .benchmark import run_benchmark, save_results, generate_markdown_report

__all__ = [
    # IR
    "BCKind",
    "FieldSpec",
    "Instruction",
    "OpCode",
    "Program",
    # Tensor
    "QTTTensor",
    # Operators
    "OperatorCache",
    "mpo_apply",
    "poisson_solve",
    # Runtime
    "QTTRuntime",
    "ExecutionResult",
    # Rank control
    "RankGovernor",
    "TruncationPolicy",
    # Telemetry
    "ProgramTelemetry",
    "StepTelemetry",
    "TelemetryCollector",
    # Benchmark
    "run_benchmark",
    "save_results",
    "generate_markdown_report",
]
