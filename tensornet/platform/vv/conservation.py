"""
Conservation Monitors
======================

Track conserved quantities (mass, energy, momentum, divergence-free constraints)
throughout a simulation and flag violations.

Every well-posed PDE system has conservation laws.  For example:

- **Mass conservation**:  ∫ ρ dV = const  (compressible flow)
- **Energy conservation**: ∫ E dV = const  (Hamiltonian / closed systems)
- **Divergence-free**:     ∇ · B = 0       (Maxwell's equations)

The ``ConservationMonitor`` hooks into the solver loop via the ``Observable``
protocol and accumulates time-series of conserved quantities, then produces
a ``ConservationReport`` with drift metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor

from tensornet.platform.data_model import (
    SimulationState,
    StructuredMesh,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ConservedQuantity base class
# ═══════════════════════════════════════════════════════════════════════════════


class ConservedQuantity(ABC):
    """
    Abstract base for any quantity that should be conserved.

    Implements the ``Observable`` protocol so it can be passed directly
    to ``TimeIntegrator.solve(observables=[...])``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name (e.g. 'mass', 'total_energy')."""
        ...

    @property
    def units(self) -> str:
        return "1"

    @abstractmethod
    def compute(self, state: SimulationState) -> Tensor:
        """Evaluate the conserved quantity for the current state."""
        ...

    def expected_value(self, state: SimulationState) -> Optional[float]:
        """
        The exact expected value, if known analytically.

        Returns ``None`` if only relative drift can be checked
        (compare to initial value).
        """
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Concrete conserved-quantity implementations
# ═══════════════════════════════════════════════════════════════════════════════


class MassIntegral(ConservedQuantity):
    """
    ∫ u dV  computed via midpoint rule on the cell volumes.

    For a density field *rho* on a structured mesh:
    ``mass = sum(rho * cell_volume)``.
    """

    def __init__(self, field_name: str = "u") -> None:
        self._field_name = field_name

    @property
    def name(self) -> str:
        return f"mass({self._field_name})"

    @property
    def units(self) -> str:
        return "kg"

    def compute(self, state: SimulationState) -> Tensor:
        field = state.get_field(self._field_name)
        vols = state.mesh.cell_volumes()
        return (field.data * vols).sum().unsqueeze(0)


class EnergyIntegral(ConservedQuantity):
    """
    Generic energy integral:  ∫ energy_density(state) dV.

    The user provides a callable ``energy_density(state) → Tensor(n_cells,)``
    that computes the energy density at each cell.
    """

    def __init__(
        self,
        energy_density_fn: Callable[[SimulationState], Tensor],
        label: str = "total_energy",
        units: str = "J",
    ) -> None:
        self._fn = energy_density_fn
        self._label = label
        self._units = units

    @property
    def name(self) -> str:
        return self._label

    @property
    def units(self) -> str:  # type: ignore[override]
        return self._units

    def compute(self, state: SimulationState) -> Tensor:
        density = self._fn(state)
        vols = state.mesh.cell_volumes()
        return (density * vols).sum().unsqueeze(0)


class LpNormQuantity(ConservedQuantity):
    """
    L^p norm of a field: (∫ |u|^p dV)^{1/p}.

    Not strictly conserved in general, but useful for tracking
    monotone decay (e.g. diffusion problems).
    """

    def __init__(self, field_name: str = "u", p: int = 2) -> None:
        self._field_name = field_name
        self._p = p

    @property
    def name(self) -> str:
        return f"L{self._p}({self._field_name})"

    def compute(self, state: SimulationState) -> Tensor:
        field = state.get_field(self._field_name)
        vols = state.mesh.cell_volumes()
        integral = (field.data.abs() ** self._p * vols).sum()
        return integral.pow(1.0 / self._p).unsqueeze(0)


class DivergenceFree(ConservedQuantity):
    """
    Discrete divergence ∇ · F for a vector field.

    For a 1-D structured mesh this is just dF/dx via central differences.
    For higher dimensions, sums of face flux differences.
    """

    def __init__(self, field_name: str = "B") -> None:
        self._field_name = field_name

    @property
    def name(self) -> str:
        return f"div({self._field_name})"

    def compute(self, state: SimulationState) -> Tensor:
        field = state.get_field(self._field_name)
        mesh = state.mesh
        if not isinstance(mesh, StructuredMesh):
            raise NotImplementedError(
                "DivergenceFree only supports StructuredMesh"
            )
        data = field.data
        if mesh.ndim == 1:
            # Central differences: (u_{i+1} - u_{i-1}) / (2 dx)
            dx = mesh.dx[0]
            div = torch.zeros_like(data)
            if data.ndim == 1:
                div[1:-1] = (data[2:] - data[:-2]) / (2.0 * dx)
                div[0] = (data[1] - data[0]) / dx
                div[-1] = (data[-1] - data[-2]) / dx
            else:
                # Vector field: sum of component divergences
                for d in range(data.shape[1]):
                    comp = data[:, d]
                    div[1:-1] += (comp[2:] - comp[:-2]) / (2.0 * dx)
                    div[0] += (comp[1] - comp[0]) / dx
                    div[-1] += (comp[-1] - comp[-2]) / dx
            # Return max absolute divergence
            return div.abs().max().unsqueeze(0)
        raise NotImplementedError(
            f"DivergenceFree not yet implemented for ndim={mesh.ndim}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Conservation Report
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ConservationReport:
    """
    Summary of conservation-law adherence across a simulation run.

    Attributes
    ----------
    quantity_name : str
    initial_value : float
    final_value : float
    max_absolute_drift : float
    max_relative_drift : float
    time_series : List[float]
    passed : bool
    threshold : float
    """

    quantity_name: str
    initial_value: float
    final_value: float
    max_absolute_drift: float
    max_relative_drift: float
    time_series: List[float]
    passed: bool
    threshold: float

    def summary(self) -> str:
        return (
            f"Conservation: {self.quantity_name}\n"
            f"  Initial:          {self.initial_value:.6e}\n"
            f"  Final:            {self.final_value:.6e}\n"
            f"  Max abs drift:    {self.max_absolute_drift:.6e}\n"
            f"  Max rel drift:    {self.max_relative_drift:.6e}\n"
            f"  Threshold:        {self.threshold:.6e}\n"
            f"  Verdict:          {'PASS' if self.passed else 'FAIL'}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Conservation Monitor
# ═══════════════════════════════════════════════════════════════════════════════


class ConservationMonitor:
    """
    Monitors one or more conserved quantities during a simulation.

    Usage::

        monitor = ConservationMonitor(
            quantities=[MassIntegral("rho"), EnergyIntegral(energy_fn)],
            threshold=1e-8,
        )

        # Option A: use as observable collector
        for step in simulation:
            state = integrator.step(state, rhs, dt)
            monitor.record(state)

        # Option B: post-hoc analysis from observable history
        monitor.record_from_history("mass(rho)", values)

        reports = monitor.reports()
    """

    def __init__(
        self,
        quantities: Sequence[ConservedQuantity],
        threshold: float = 1e-8,
    ) -> None:
        self._quantities = list(quantities)
        self._threshold = threshold
        self._history: Dict[str, List[float]] = {
            q.name: [] for q in self._quantities
        }
        self._recorded_steps = 0

    @property
    def quantities(self) -> List[ConservedQuantity]:
        return list(self._quantities)

    def record(self, state: SimulationState) -> None:
        """Evaluate all quantities and append to history."""
        for q in self._quantities:
            val = q.compute(state)
            self._history[q.name].append(val.item())
        self._recorded_steps += 1

    def record_from_history(
        self, quantity_name: str, values: Sequence[float]
    ) -> None:
        """Directly load a pre-computed time series."""
        if quantity_name not in self._history:
            self._history[quantity_name] = []
        self._history[quantity_name] = list(values)

    def report(
        self, quantity_name: str, threshold: Optional[float] = None
    ) -> ConservationReport:
        """
        Generate a conservation report for one quantity.

        Parameters
        ----------
        quantity_name : str
        threshold : float, optional
            Override default threshold.
        """
        thr = threshold if threshold is not None else self._threshold
        series = self._history.get(quantity_name, [])
        if len(series) < 2:
            return ConservationReport(
                quantity_name=quantity_name,
                initial_value=series[0] if series else 0.0,
                final_value=series[-1] if series else 0.0,
                max_absolute_drift=0.0,
                max_relative_drift=0.0,
                time_series=series,
                passed=True,
                threshold=thr,
            )

        initial = series[0]
        abs_drifts = [abs(v - initial) for v in series]
        max_abs = max(abs_drifts)
        ref = abs(initial) if abs(initial) > 1e-30 else 1.0
        max_rel = max_abs / ref

        return ConservationReport(
            quantity_name=quantity_name,
            initial_value=initial,
            final_value=series[-1],
            max_absolute_drift=max_abs,
            max_relative_drift=max_rel,
            time_series=series,
            passed=max_rel <= thr,
            threshold=thr,
        )

    def reports(
        self, threshold: Optional[float] = None
    ) -> List[ConservationReport]:
        """Generate conservation reports for all tracked quantities."""
        return [
            self.report(q.name, threshold)
            for q in self._quantities
            if q.name in self._history and self._history[q.name]
        ]

    def all_passed(self, threshold: Optional[float] = None) -> bool:
        """True if every tracked quantity is within threshold."""
        return all(r.passed for r in self.reports(threshold))
