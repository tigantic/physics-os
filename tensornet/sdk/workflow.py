"""
SDK Workflow Builder — high-level, composable simulation pipeline.

``WorkflowBuilder`` is the primary user-facing entry point for constructing
simulation + analysis pipelines without touching platform internals.

Usage
-----
::

    from tensornet.sdk import WorkflowBuilder

    wf = (
        WorkflowBuilder("lid_driven_cavity")
        .domain(shape=(64, 64), extent=((0, 1), (0, 1)))
        .field("u", ic="uniform", value=0.0)
        .field("v", ic="uniform", value=0.0)
        .solver("PHY-II.1")
        .time(0.0, 1.0, dt=1e-3)
        .observe("energy")
        .export("vtu", path="results/cavity")
        .build()
    )
    result = wf.run()

Thread-safety: ``WorkflowBuilder`` is not thread-safe.  Build one per thread.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch
from torch import Tensor

from tensornet.platform.data_model import (
    BCType,
    BoundaryCondition,
    FieldData,
    InitialCondition,
    Mesh,
    SimulationState,
    StructuredMesh,
)
from tensornet.platform.domain_pack import DomainRegistry, get_registry
from tensornet.platform.export import ExportBundle, export_csv, export_vtu
from tensornet.platform.lineage import LineageDAG, LineageEvent, LineageTracker
from tensornet.platform.protocols import Observable, Solver, SolveResult, WorkflowResult
from tensornet.platform.reproduce import ReproducibilityContext, hash_tensor

logger = logging.getLogger(__name__)

__all__ = ["WorkflowBuilder", "WorkflowConfig", "ExecutedWorkflow"]


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FieldSpec:
    """Specification for a single field."""

    name: str
    ic_type: str = "uniform"
    ic_value: Union[float, Tensor, None] = 0.0
    ic_function: Optional[Callable[[Tensor], Tensor]] = None
    components: int = 1
    units: str = "1"


@dataclass
class BCSpec:
    """Specification for a boundary condition."""

    field_name: str
    region: str
    bc_type: str  # "dirichlet", "neumann", "periodic"
    value: Union[float, None] = None


@dataclass
class ExportSpec:
    """Specification for result export."""

    format: str  # "vtu", "csv", "json", "xdmf"
    path: str = "."
    fields: Optional[List[str]] = None


@dataclass
class WorkflowConfig:
    """Complete workflow configuration built by WorkflowBuilder."""

    name: str
    mesh_shape: Optional[Tuple[int, ...]] = None
    mesh_domain: Optional[Tuple[Tuple[float, float], ...]] = None
    mesh: Optional[Mesh] = None
    fields: List[FieldSpec] = dc_field(default_factory=list)
    bcs: List[BCSpec] = dc_field(default_factory=list)
    solver_id: Optional[str] = None
    solver_kwargs: Dict[str, Any] = dc_field(default_factory=dict)
    t_start: float = 0.0
    t_end: float = 1.0
    dt: float = 0.01
    observe_names: List[str] = dc_field(default_factory=list)
    exports: List[ExportSpec] = dc_field(default_factory=list)
    seed: int = 42
    max_steps: Optional[int] = None
    metadata: Dict[str, Any] = dc_field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════════════


class WorkflowBuilder:
    """
    Fluent builder for simulation workflows.

    Chain calls to configure domain, fields, solver, time, export, then
    call ``.build()`` to get an ``ExecutedWorkflow`` that can be ``.run()``'d.
    """

    def __init__(self, name: str) -> None:
        self._config = WorkflowConfig(name=name)

    def domain(
        self,
        shape: Tuple[int, ...],
        extent: Tuple[Tuple[float, float], ...],
    ) -> "WorkflowBuilder":
        """Define a structured-mesh domain."""
        self._config.mesh_shape = shape
        self._config.mesh_domain = extent
        return self

    def mesh(self, mesh: Mesh) -> "WorkflowBuilder":
        """Provide a pre-built mesh (structured or unstructured)."""
        self._config.mesh = mesh
        return self

    def field(
        self,
        name: str,
        *,
        ic: str = "uniform",
        value: Union[float, Tensor, None] = 0.0,
        function: Optional[Callable[[Tensor], Tensor]] = None,
        components: int = 1,
        units: str = "1",
    ) -> "WorkflowBuilder":
        """Add a field with initial condition."""
        self._config.fields.append(FieldSpec(
            name=name,
            ic_type=ic,
            ic_value=value,
            ic_function=function,
            components=components,
            units=units,
        ))
        return self

    def bc(
        self,
        field_name: str,
        region: str,
        bc_type: str = "dirichlet",
        value: Optional[float] = None,
    ) -> "WorkflowBuilder":
        """Add a boundary condition."""
        self._config.bcs.append(BCSpec(
            field_name=field_name,
            region=region,
            bc_type=bc_type,
            value=value,
        ))
        return self

    def solver(
        self,
        taxonomy_id: str,
        **kwargs: Any,
    ) -> "WorkflowBuilder":
        """
        Set the solver by taxonomy ID (e.g. ``'PHY-II.1'``).

        Extra kwargs are passed to the solver constructor.
        """
        self._config.solver_id = taxonomy_id
        self._config.solver_kwargs = kwargs
        return self

    def time(
        self,
        t_start: float,
        t_end: float,
        dt: float,
        *,
        max_steps: Optional[int] = None,
    ) -> "WorkflowBuilder":
        """Define the time window."""
        self._config.t_start = t_start
        self._config.t_end = t_end
        self._config.dt = dt
        self._config.max_steps = max_steps
        return self

    def observe(self, *names: str) -> "WorkflowBuilder":
        """Add named observables to record."""
        self._config.observe_names.extend(names)
        return self

    def export(
        self,
        format: str,
        *,
        path: str = ".",
        fields: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add an export action (``'vtu'``, ``'csv'``, ``'json'``, ``'xdmf'``)."""
        self._config.exports.append(ExportSpec(
            format=format, path=path, fields=fields
        ))
        return self

    def seed(self, seed: int) -> "WorkflowBuilder":
        """Set the reproducibility seed."""
        self._config.seed = seed
        return self

    def meta(self, **kwargs: Any) -> "WorkflowBuilder":
        """Attach arbitrary metadata."""
        self._config.metadata.update(kwargs)
        return self

    def build(self) -> "ExecutedWorkflow":
        """Validate and return an executable workflow."""
        cfg = self._config
        if cfg.mesh is None and cfg.mesh_shape is None:
            raise ValueError("No mesh defined.  Call .domain() or .mesh().")
        if cfg.solver_id is None:
            raise ValueError("No solver specified.  Call .solver().")
        if not cfg.fields:
            raise ValueError("No fields defined.  Call .field().")
        return ExecutedWorkflow(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Executed Workflow
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RunResult:
    """Result of executing a workflow."""

    solve_result: SolveResult
    final_state: SimulationState
    provenance: Dict[str, Any]
    exported_files: List[Path]
    wall_time: float
    lineage: LineageDAG


class ExecutedWorkflow:
    """
    A fully-configured workflow ready to run.

    Call ``.run()`` to execute the simulation + post-processing pipeline.
    """

    def __init__(self, config: WorkflowConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return self._config.name

    def run(self, **overrides: Any) -> RunResult:
        """Execute the full pipeline: mesh → IC → solve → export → provenance."""
        cfg = self._config
        lineage = LineageDAG()
        tracker = LineageTracker(lineage)
        wall_start = time.monotonic()

        with ReproducibilityContext(seed=cfg.seed) as repro:

            # ── Mesh ──
            if cfg.mesh is not None:
                mesh = cfg.mesh
            else:
                mesh = StructuredMesh(
                    shape=cfg.mesh_shape,  # type: ignore[arg-type]
                    domain=cfg.mesh_domain,  # type: ignore[arg-type]
                )

            # ── Initial conditions ──
            fields: Dict[str, FieldData] = {}
            for fspec in cfg.fields:
                ic = InitialCondition(
                    field_name=fspec.name,
                    ic_type=fspec.ic_type,
                    value=fspec.ic_value,
                    function=fspec.ic_function,
                    metadata={"units": fspec.units},
                )
                fdata = ic.generate(mesh)
                if fspec.components > 1:
                    # Expand scalar IC to vector
                    fdata = FieldData(
                        name=fspec.name,
                        data=fdata.data.unsqueeze(-1).expand(-1, fspec.components).clone().reshape(mesh.n_cells, fspec.components),
                        mesh=mesh,
                        components=fspec.components,
                        units=fspec.units,
                    )
                fields[fspec.name] = fdata

            state = SimulationState(
                t=cfg.t_start,
                fields=fields,
                mesh=mesh,
                metadata=dict(cfg.metadata),
            )

            # Record lineage
            tracker.record_instant(
                event=LineageEvent.CHECKPOINT,
                metadata={"phase": "initial_condition", "n_fields": len(fields)},
            )

            # ── Solver ──
            registry = get_registry()
            solver_cls = registry.get_solver(cfg.solver_id)
            solver = solver_cls(**cfg.solver_kwargs)

            # ── Solve ──
            with tracker.record(
                event=LineageEvent.FORWARD_SOLVE,
                metadata={"solver": solver.name, "dt": cfg.dt},
            ):
                if hasattr(solver, "solve") and callable(solver.solve):
                    solve_result = solver.solve(
                        state,
                        (cfg.t_start, cfg.t_end),
                        cfg.dt,
                        max_steps=cfg.max_steps,
                    )
                else:
                    # Step-by-step fallback
                    t = cfg.t_start
                    steps = 0
                    limit = cfg.max_steps or int(1e9)
                    while t < cfg.t_end - 1e-14 and steps < limit:
                        actual_dt = min(cfg.dt, cfg.t_end - t)
                        state = solver.step(state, actual_dt)
                        t += actual_dt
                        steps += 1
                    solve_result = SolveResult(
                        final_state=state,
                        t_final=t,
                        steps_taken=steps,
                    )

            final_state = solve_result.final_state

            # ── Record artifact hashes ──
            for fname, fdata in final_state.fields.items():
                repro.record(fname, hash_tensor(fdata.data))

            # ── Export ──
            exported: List[Path] = []
            for espec in cfg.exports:
                if espec.format == "vtu":
                    p = export_vtu(
                        final_state,
                        Path(espec.path) / f"{cfg.name}.vtu",
                        fields=espec.fields,
                    )
                    exported.append(p)
                elif espec.format == "csv":
                    if solve_result.observable_history:
                        csv_data: Dict[str, List[float]] = {}
                        for obs_name, vals in solve_result.observable_history.items():
                            csv_data[obs_name] = [
                                v.item() if isinstance(v, Tensor) else float(v)
                                for v in vals
                            ]
                        p = export_csv(csv_data, Path(espec.path) / f"{cfg.name}.csv")
                        exported.append(p)
                elif espec.format == "json":
                    from tensornet.platform.export import export_json
                    meta = {
                        "name": cfg.name,
                        "t_final": solve_result.t_final,
                        "steps": solve_result.steps_taken,
                        "converged": solve_result.converged,
                    }
                    p = export_json(meta, Path(espec.path) / f"{cfg.name}_meta.json")
                    exported.append(p)
                elif espec.format == "xdmf":
                    from tensornet.platform.export import export_xdmf_hdf5
                    xp, hp = export_xdmf_hdf5(
                        final_state,
                        Path(espec.path) / cfg.name,
                        fields=espec.fields,
                    )
                    exported.extend([xp, hp])

        wall_time = time.monotonic() - wall_start

        return RunResult(
            solve_result=solve_result,
            final_state=final_state,
            provenance=repro.provenance(),
            exported_files=exported,
            wall_time=wall_time,
            lineage=lineage,
        )
