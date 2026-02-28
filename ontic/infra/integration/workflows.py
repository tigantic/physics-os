"""
Workflow Orchestration for Project The Ontic Engine.

Provides end-to-end workflow definitions and execution for:
- CFD simulations
- Guidance computations
- Digital twin synchronization
- Validation campaigns
"""

import copy
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class WorkflowStatus(Enum):
    """Status of a workflow or step."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    SKIPPED = auto()


@dataclass
class WorkflowStep:
    """
    Single step in a workflow.

    Attributes:
        name: Step identifier
        executor: Callable that performs the step
        description: Human-readable description
        required_inputs: Input keys this step needs
        outputs: Output keys this step produces
        timeout: Maximum execution time in seconds
        retries: Number of retry attempts
        skip_on_failure: Whether to continue workflow if step fails
    """

    name: str
    executor: Callable[[dict[str, Any]], dict[str, Any]]
    description: str = ""
    required_inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    timeout: float | None = None
    retries: int = 0
    skip_on_failure: bool = False

    def validate_inputs(self, context: dict[str, Any]) -> bool:
        """Check if all required inputs are available."""
        return all(key in context for key in self.required_inputs)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the step.

        Args:
            context: Workflow context with inputs

        Returns:
            Step outputs to merge into context
        """
        return self.executor(context)


@dataclass
class WorkflowStage:
    """
    Collection of steps that can run in parallel.

    Attributes:
        name: Stage identifier
        steps: Steps in this stage
        description: Stage description
        parallel: Whether steps can run in parallel
    """

    name: str
    steps: list[WorkflowStep] = field(default_factory=list)
    description: str = ""
    parallel: bool = False

    def add_step(self, step: WorkflowStep):
        """Add a step to this stage."""
        self.steps.append(step)


@dataclass
class WorkflowResult:
    """
    Result of workflow execution.

    Attributes:
        workflow_name: Name of the workflow
        status: Final status
        context: Final workflow context
        step_results: Results per step
        duration: Total execution time
        error: Error message if failed
    """

    workflow_name: str
    status: WorkflowStatus
    context: dict[str, Any]
    step_results: dict[str, dict] = field(default_factory=dict)
    duration: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether workflow completed successfully."""
        return self.status == WorkflowStatus.COMPLETED

    def get_output(self, key: str, default: Any = None) -> Any:
        """Get an output from the context."""
        return self.context.get(key, default)

    def summary(self) -> str:
        """Generate a summary of the workflow result."""
        lines = [
            f"Workflow: {self.workflow_name}",
            f"Status: {self.status.name}",
            f"Duration: {self.duration:.3f}s",
            f"Steps: {len(self.step_results)}",
        ]

        passed = sum(
            1 for r in self.step_results.values() if r.get("status") == "completed"
        )
        failed = sum(
            1 for r in self.step_results.values() if r.get("status") == "failed"
        )
        lines.append(f"  Passed: {passed}, Failed: {failed}")

        if self.error:
            lines.append(f"Error: {self.error}")

        return "\n".join(lines)


@dataclass
class Workflow:
    """
    Complete workflow definition.

    Attributes:
        name: Workflow identifier
        description: Workflow description
        stages: Ordered list of stages
        initial_context: Default context values
        version: Workflow version string
    """

    name: str
    description: str = ""
    stages: list[WorkflowStage] = field(default_factory=list)
    initial_context: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def add_stage(self, stage: WorkflowStage):
        """Add a stage to the workflow."""
        self.stages.append(stage)

    def add_step(self, step: WorkflowStep, stage_name: str | None = None):
        """
        Add a step to a stage.

        If stage_name is None, creates a new stage for the step.
        """
        if stage_name is None:
            stage = WorkflowStage(name=step.name)
            stage.add_step(step)
            self.add_stage(stage)
        else:
            for stage in self.stages:
                if stage.name == stage_name:
                    stage.add_step(step)
                    return
            # Stage not found, create it
            stage = WorkflowStage(name=stage_name)
            stage.add_step(step)
            self.add_stage(stage)


class WorkflowEngine:
    """
    Engine for executing workflows.

    Handles step execution, context management, retries, and error handling.
    """

    def __init__(
        self,
        max_parallel: int = 4,
        default_timeout: float = 300.0,
        verbose: bool = True,
    ):
        """
        Initialize workflow engine.

        Args:
            max_parallel: Maximum parallel steps
            default_timeout: Default step timeout
            verbose: Whether to print progress
        """
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout
        self.verbose = verbose
        self._hooks: dict[str, list[Callable]] = {
            "before_step": [],
            "after_step": [],
            "on_error": [],
        }

    def add_hook(self, event: str, callback: Callable):
        """Add a hook callback for an event."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    def _call_hooks(self, event: str, **kwargs):
        """Call all hooks for an event."""
        for hook in self._hooks.get(event, []):
            try:
                hook(**kwargs)
            except Exception:
                pass  # Don't let hooks break the workflow

    def _execute_step(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single step with retries."""
        attempts = 0
        last_error = None

        while attempts <= step.retries:
            try:
                self._call_hooks("before_step", step=step, context=context)

                start_time = time.time()
                outputs = step.execute(context)
                elapsed = time.time() - start_time

                result = {
                    "status": "completed",
                    "duration": elapsed,
                    "outputs": outputs or {},
                    "attempts": attempts + 1,
                }

                self._call_hooks("after_step", step=step, result=result)
                return result

            except Exception as e:
                last_error = e
                attempts += 1
                self._call_hooks("on_error", step=step, error=e, attempt=attempts)

                if attempts <= step.retries:
                    if self.verbose:
                        print(
                            f"  Step '{step.name}' failed, retrying ({attempts}/{step.retries})"
                        )

        return {
            "status": "failed",
            "error": str(last_error),
            "traceback": traceback.format_exc(),
            "attempts": attempts,
        }

    def run(
        self,
        workflow: Workflow,
        context: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """
        Execute a workflow.

        Args:
            workflow: Workflow to execute
            context: Initial context overrides

        Returns:
            WorkflowResult with outcomes
        """
        start_time = time.time()

        # Initialize context
        ctx = copy.deepcopy(workflow.initial_context)
        if context:
            ctx.update(context)

        step_results = {}
        final_status = WorkflowStatus.COMPLETED
        error_message = None

        if self.verbose:
            print(f"Starting workflow: {workflow.name}")

        for stage in workflow.stages:
            if self.verbose:
                print(f"  Stage: {stage.name}")

            for step in stage.steps:
                # Validate inputs
                if not step.validate_inputs(ctx):
                    missing = [k for k in step.required_inputs if k not in ctx]
                    step_results[step.name] = {
                        "status": "skipped",
                        "reason": f"Missing inputs: {missing}",
                    }

                    if not step.skip_on_failure:
                        final_status = WorkflowStatus.FAILED
                        error_message = f"Step '{step.name}' missing inputs: {missing}"
                        break
                    continue

                if self.verbose:
                    print(f"    Running: {step.name}")

                result = self._execute_step(step, ctx)
                step_results[step.name] = result

                if result["status"] == "completed":
                    # Merge outputs into context
                    ctx.update(result.get("outputs", {}))
                else:
                    if step.skip_on_failure:
                        if self.verbose:
                            print(f"    Skipping failure for: {step.name}")
                    else:
                        final_status = WorkflowStatus.FAILED
                        error_message = (
                            f"Step '{step.name}' failed: {result.get('error')}"
                        )
                        break

            if final_status == WorkflowStatus.FAILED:
                break

        duration = time.time() - start_time

        if self.verbose:
            status_str = (
                "SUCCESS" if final_status == WorkflowStatus.COMPLETED else "FAILED"
            )
            print(f"Workflow {status_str} in {duration:.2f}s")

        return WorkflowResult(
            workflow_name=workflow.name,
            status=final_status,
            context=ctx,
            step_results=step_results,
            duration=duration,
            error=error_message,
        )


# =============================================================================
# Pre-built Workflow Definitions
# =============================================================================


class CFDSimulationWorkflow(Workflow):
    """
    Standard CFD simulation workflow.

    Stages:
        1. Initialization - Load mesh, set ICs/BCs
        2. Preprocessing - Compute metrics, initialize solver
        3. Solving - Time stepping loop
        4. Postprocessing - Extract results, visualize
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize CFD workflow.

        Args:
            config: Simulation configuration
        """
        super().__init__(
            name="CFD_Simulation",
            description="End-to-end CFD simulation workflow",
            version="1.0",
        )

        config = config or {}
        self.initial_context.update(config)

        # Stage 1: Initialization
        init_stage = WorkflowStage(
            name="initialization",
            description="Load geometry and initial conditions",
        )

        init_stage.add_step(
            WorkflowStep(
                name="load_mesh",
                executor=self._load_mesh,
                description="Load computational mesh",
                outputs=["mesh", "dx", "dy"],
            )
        )

        init_stage.add_step(
            WorkflowStep(
                name="set_initial_conditions",
                executor=self._set_initial_conditions,
                description="Set initial flow conditions",
                required_inputs=["mesh"],
                outputs=["state"],
            )
        )

        init_stage.add_step(
            WorkflowStep(
                name="set_boundary_conditions",
                executor=self._set_boundary_conditions,
                description="Configure boundary conditions",
                required_inputs=["mesh"],
                outputs=["boundary_conditions"],
            )
        )

        self.add_stage(init_stage)

        # Stage 2: Preprocessing
        preproc_stage = WorkflowStage(
            name="preprocessing",
            description="Prepare solver",
        )

        preproc_stage.add_step(
            WorkflowStep(
                name="initialize_solver",
                executor=self._initialize_solver,
                description="Initialize CFD solver",
                required_inputs=["state", "boundary_conditions"],
                outputs=["solver"],
            )
        )

        self.add_stage(preproc_stage)

        # Stage 3: Solving
        solve_stage = WorkflowStage(
            name="solving",
            description="Time integration",
        )

        solve_stage.add_step(
            WorkflowStep(
                name="time_stepping",
                executor=self._time_stepping,
                description="Advance solution in time",
                required_inputs=["solver", "state"],
                outputs=["final_state", "history"],
                timeout=3600.0,
            )
        )

        self.add_stage(solve_stage)

        # Stage 4: Postprocessing
        post_stage = WorkflowStage(
            name="postprocessing",
            description="Extract and save results",
        )

        post_stage.add_step(
            WorkflowStep(
                name="extract_results",
                executor=self._extract_results,
                description="Compute derived quantities",
                required_inputs=["final_state"],
                outputs=["results"],
            )
        )

        self.add_stage(post_stage)

    def _load_mesh(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Load mesh step."""
        import torch

        nx = ctx.get("nx", 100)
        ny = ctx.get("ny", 50)
        Lx = ctx.get("Lx", 1.0)
        Ly = ctx.get("Ly", 0.5)

        x = torch.linspace(0, Lx, nx)
        y = torch.linspace(0, Ly, ny)
        mesh = {"x": x, "y": y, "nx": nx, "ny": ny}

        return {
            "mesh": mesh,
            "dx": Lx / (nx - 1),
            "dy": Ly / (ny - 1),
        }

    def _set_initial_conditions(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Set initial conditions step."""
        import torch

        mesh = ctx["mesh"]
        nx, ny = mesh["nx"], mesh["ny"]

        # Default: uniform flow
        rho = torch.ones(nx, ny)
        u = torch.full((nx, ny), ctx.get("u_inf", 1.0))
        v = torch.zeros(nx, ny)
        p = torch.full((nx, ny), ctx.get("p_inf", 1.0 / 1.4))

        state = torch.stack(
            [rho, rho * u, rho * v, p / 0.4 + 0.5 * rho * (u**2 + v**2)]
        )

        return {"state": state}

    def _set_boundary_conditions(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Set boundary conditions step."""
        bc = {
            "left": ctx.get("bc_left", "supersonic_inlet"),
            "right": ctx.get("bc_right", "supersonic_outlet"),
            "top": ctx.get("bc_top", "slip_wall"),
            "bottom": ctx.get("bc_bottom", "slip_wall"),
        }
        return {"boundary_conditions": bc}

    def _initialize_solver(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Initialize solver step."""
        solver_config = {
            "cfl": ctx.get("cfl", 0.5),
            "flux_scheme": ctx.get("flux_scheme", "roe"),
            "limiter": ctx.get("limiter", "minmod"),
            "gamma": ctx.get("gamma", 1.4),
        }
        return {"solver": solver_config}

    def _time_stepping(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Time stepping step."""
        state = ctx["state"]
        n_steps = ctx.get("n_steps", 100)

        history = []
        current_state = state.clone()

        for step in range(n_steps):
            # Simplified: just track time
            history.append({"step": step, "residual": 1e-3})

        return {
            "final_state": current_state,
            "history": history,
        }

    def _extract_results(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Extract results step."""
        state = ctx["final_state"]

        rho = state[0]
        u = state[1] / rho
        v = state[2] / rho
        E = state[3]
        p = 0.4 * (E - 0.5 * rho * (u**2 + v**2))

        gamma = ctx.get("gamma", 1.4)
        mach = (u**2 + v**2).sqrt() / (gamma * p / rho).sqrt()

        results = {
            "density": rho,
            "velocity_x": u,
            "velocity_y": v,
            "pressure": p,
            "mach": mach,
            "max_mach": mach.max().item(),
        }

        return {"results": results}


class GuidanceWorkflow(Workflow):
    """
    Trajectory guidance workflow.

    Stages:
        1. State estimation - Current vehicle state
        2. CFD query - Aerodynamic coefficients
        3. Trajectory optimization - Optimal path
        4. Control command - Actuator commands
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize guidance workflow."""
        super().__init__(
            name="Guidance_Loop",
            description="Real-time trajectory guidance workflow",
            version="1.0",
        )

        config = config or {}
        self.initial_context.update(config)

        # Stage 1: State estimation
        state_stage = WorkflowStage(name="state_estimation")
        state_stage.add_step(
            WorkflowStep(
                name="estimate_state",
                executor=self._estimate_state,
                description="Estimate current vehicle state",
                outputs=["vehicle_state"],
            )
        )
        self.add_stage(state_stage)

        # Stage 2: CFD query
        cfd_stage = WorkflowStage(name="aerodynamics")
        cfd_stage.add_step(
            WorkflowStep(
                name="query_aero",
                executor=self._query_aerodynamics,
                description="Get aerodynamic coefficients",
                required_inputs=["vehicle_state"],
                outputs=["aero_coeffs"],
            )
        )
        self.add_stage(cfd_stage)

        # Stage 3: Trajectory optimization
        traj_stage = WorkflowStage(name="trajectory")
        traj_stage.add_step(
            WorkflowStep(
                name="optimize_trajectory",
                executor=self._optimize_trajectory,
                description="Compute optimal trajectory",
                required_inputs=["vehicle_state", "aero_coeffs"],
                outputs=["trajectory"],
            )
        )
        self.add_stage(traj_stage)

        # Stage 4: Control
        control_stage = WorkflowStage(name="control")
        control_stage.add_step(
            WorkflowStep(
                name="compute_control",
                executor=self._compute_control,
                description="Compute control commands",
                required_inputs=["trajectory", "vehicle_state"],
                outputs=["control_command"],
            )
        )
        self.add_stage(control_stage)

    def _estimate_state(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Estimate vehicle state."""
        import torch

        state = {
            "position": torch.tensor(ctx.get("position", [0.0, 0.0, 10000.0])),
            "velocity": torch.tensor(ctx.get("velocity", [1000.0, 0.0, -50.0])),
            "attitude": torch.tensor(ctx.get("attitude", [0.0, 0.05, 0.0])),
            "angular_rate": torch.tensor(ctx.get("angular_rate", [0.0, 0.01, 0.0])),
        }
        return {"vehicle_state": state}

    def _query_aerodynamics(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Query aerodynamic database."""

        state = ctx["vehicle_state"]
        velocity = state["velocity"]
        mach = velocity.norm().item() / 340.0  # Approximate

        # Simplified aero model
        coeffs = {
            "CL": 0.5 * mach / 5.0,
            "CD": 0.02 + 0.01 * mach,
            "Cm": -0.01,
        }
        return {"aero_coeffs": coeffs}

    def _optimize_trajectory(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Compute optimal trajectory."""
        import torch

        state = ctx["vehicle_state"]

        # Simplified: straight line to target
        target = torch.tensor(ctx.get("target", [50000.0, 0.0, 0.0]))
        current = state["position"]

        trajectory = {
            "waypoints": [current, target],
            "time_to_target": (target - current).norm().item() / 1000.0,
        }
        return {"trajectory": trajectory}

    def _compute_control(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Compute control commands."""

        trajectory = ctx["trajectory"]
        state = ctx["vehicle_state"]

        # Simplified proportional control
        command = {
            "thrust": 0.9,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
        }
        return {"control_command": command}


class DigitalTwinWorkflow(Workflow):
    """
    Digital twin synchronization workflow.

    Stages:
        1. Data ingestion - Receive telemetry
        2. State sync - Update digital twin
        3. Health check - Monitor anomalies
        4. Prediction - Forecast future states
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize digital twin workflow."""
        super().__init__(
            name="Digital_Twin_Sync",
            description="Digital twin synchronization workflow",
            version="1.0",
        )

        config = config or {}
        self.initial_context.update(config)

        # Stage 1: Data ingestion
        ingest_stage = WorkflowStage(name="ingestion")
        ingest_stage.add_step(
            WorkflowStep(
                name="receive_telemetry",
                executor=self._receive_telemetry,
                description="Receive sensor data",
                outputs=["telemetry"],
            )
        )
        self.add_stage(ingest_stage)

        # Stage 2: State sync
        sync_stage = WorkflowStage(name="synchronization")
        sync_stage.add_step(
            WorkflowStep(
                name="update_twin",
                executor=self._update_twin,
                description="Update digital twin state",
                required_inputs=["telemetry"],
                outputs=["twin_state"],
            )
        )
        self.add_stage(sync_stage)

        # Stage 3: Health check
        health_stage = WorkflowStage(name="health_monitoring")
        health_stage.add_step(
            WorkflowStep(
                name="check_health",
                executor=self._check_health,
                description="Monitor system health",
                required_inputs=["twin_state"],
                outputs=["health_status"],
            )
        )
        self.add_stage(health_stage)

        # Stage 4: Prediction
        predict_stage = WorkflowStage(name="prediction")
        predict_stage.add_step(
            WorkflowStep(
                name="forecast",
                executor=self._forecast,
                description="Forecast future states",
                required_inputs=["twin_state"],
                outputs=["forecast"],
                skip_on_failure=True,
            )
        )
        self.add_stage(predict_stage)

    def _receive_telemetry(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Receive telemetry data."""
        import time

        telemetry = ctx.get(
            "incoming_telemetry",
            {
                "timestamp": time.time(),
                "temperature": 300.0,
                "pressure": 101325.0,
                "strain": [0.001, 0.002, 0.001],
            },
        )
        return {"telemetry": telemetry}

    def _update_twin(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Update digital twin."""
        telemetry = ctx["telemetry"]

        twin_state = {
            "last_update": telemetry.get("timestamp", 0),
            "thermal_state": telemetry.get("temperature", 300.0),
            "structural_state": telemetry.get("strain", [0.0, 0.0, 0.0]),
            "sync_quality": 0.95,
        }
        return {"twin_state": twin_state}

    def _check_health(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Check system health."""
        twin_state = ctx["twin_state"]

        issues = []
        if twin_state.get("thermal_state", 0) > 500:
            issues.append("HIGH_TEMPERATURE")

        health_status = {
            "healthy": len(issues) == 0,
            "issues": issues,
            "confidence": twin_state.get("sync_quality", 0.0),
        }
        return {"health_status": health_status}

    def _forecast(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Forecast future states."""

        twin_state = ctx["twin_state"]

        # Simplified linear forecast
        forecast = {
            "horizon": 60.0,  # seconds
            "temperature_trend": 0.1,  # K/s
            "rul_estimate": 1000.0,  # seconds
        }
        return {"forecast": forecast}


class ValidationWorkflow(Workflow):
    """
    Validation campaign workflow.

    Stages:
        1. Setup - Configure validation suite
        2. Execution - Run validation tests
        3. Analysis - Analyze results
        4. Reporting - Generate reports
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize validation workflow."""
        super().__init__(
            name="Validation_Campaign",
            description="End-to-end validation workflow",
            version="1.0",
        )

        config = config or {}
        self.initial_context.update(config)

        # Stages
        self.add_stage(WorkflowStage(name="setup"))
        self.add_stage(WorkflowStage(name="execution"))
        self.add_stage(WorkflowStage(name="analysis"))
        self.add_stage(WorkflowStage(name="reporting"))


# =============================================================================
# Convenience Functions
# =============================================================================


def run_workflow(
    workflow: Workflow,
    context: dict[str, Any] | None = None,
    verbose: bool = True,
) -> WorkflowResult:
    """
    Execute a workflow.

    Args:
        workflow: Workflow to run
        context: Initial context
        verbose: Print progress

    Returns:
        WorkflowResult with outcomes
    """
    engine = WorkflowEngine(verbose=verbose)
    return engine.run(workflow, context)


def create_cfd_workflow(
    nx: int = 100,
    ny: int = 50,
    n_steps: int = 100,
    cfl: float = 0.5,
    **kwargs,
) -> CFDSimulationWorkflow:
    """
    Create a CFD simulation workflow.

    Args:
        nx: Grid points in x
        ny: Grid points in y
        n_steps: Number of time steps
        cfl: CFL number
        **kwargs: Additional configuration

    Returns:
        Configured CFDSimulationWorkflow
    """
    config = {
        "nx": nx,
        "ny": ny,
        "n_steps": n_steps,
        "cfl": cfl,
        **kwargs,
    }
    return CFDSimulationWorkflow(config)


def create_guidance_workflow(
    target: list[float] | None = None,
    **kwargs,
) -> GuidanceWorkflow:
    """
    Create a guidance workflow.

    Args:
        target: Target position [x, y, z]
        **kwargs: Additional configuration

    Returns:
        Configured GuidanceWorkflow
    """
    config = kwargs.copy()
    if target:
        config["target"] = target
    return GuidanceWorkflow(config)
