"""QTT Physics VM — Parametric antenna geometry engine.

Defines parameterised antenna families with design variables that can be
swept, optimised, and scored.  Each design maps its parameters to the
MaxwellAntenna3DCompiler's geometry dataclasses.

Architecture:
    DesignVariable  — one knob (continuous, discrete, or categorical)
    DesignSpace     — collection of variables + constraints
    *AntennaDesign  — concrete parameterised family (patch, dipole, etc.)

Each design class implements:
    .design_space   → DesignSpace (advertises tuneable parameters)
    .to_compiler()  → MaxwellAntenna3DCompiler (builds a runnable sim)
    .to_dict()      → serialisable parameter snapshot

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..compilers.maxwell_antenna_3d import (
    DipoleGeometry,
    MaxwellAntenna3DCompiler,
    PatchGeometry,
    WavePort,
)
from .materials import Material, MaterialLibrary


# ─────────────────────────────────────────────────────────────────────
# Design variable primitives
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DesignVariable:
    """One tuneable knob in a parametric antenna design.

    Parameters
    ----------
    name : str
        Unique identifier (e.g. ``"patch_width"``).
    low : float
        Lower bound.
    high : float
        Upper bound.
    default : float
        Nominal / starting value.
    unit : str
        Physical unit string (e.g. ``"mm"``, ``"MHz"``).
    description : str
        Human-readable description for reports.
    n_steps : int
        Default number of sweep steps (for uniform grid).
    is_integer : bool
        If True, values are rounded to integers during sweep.
    """

    name: str
    low: float
    high: float
    default: float
    unit: str = ""
    description: str = ""
    n_steps: int = 5
    is_integer: bool = False

    def linspace(self, n: int | None = None) -> list[float]:
        """Generate uniformly spaced values within bounds."""
        steps = n if n is not None else self.n_steps
        vals = np.linspace(self.low, self.high, steps).tolist()
        if self.is_integer:
            vals = [float(round(v)) for v in vals]
        return vals

    def clamp(self, value: float) -> float:
        """Clamp a value to within bounds."""
        v = max(self.low, min(self.high, value))
        if self.is_integer:
            v = float(round(v))
        return v


@dataclass
class DesignSpace:
    """Collection of design variables defining a parameter space.

    The design space also carries constraints (callables that return
    True if a parameter combination is valid).  Invalid combinations
    are skipped during sweeps.
    """

    variables: list[DesignVariable] = field(default_factory=list)
    constraints: list[Any] = field(default_factory=list)

    @property
    def n_dims(self) -> int:
        """Number of tuneable dimensions."""
        return len(self.variables)

    @property
    def names(self) -> list[str]:
        """Variable names."""
        return [v.name for v in self.variables]

    def defaults(self) -> dict[str, float]:
        """Return the default parameter vector as a dict."""
        return {v.name: v.default for v in self.variables}

    def grid_points(
        self, n_per_dim: int | None = None
    ) -> list[dict[str, float]]:
        """Generate full grid of parameter combinations.

        Filters out invalid combinations using constraints.
        """
        import itertools

        axes = []
        for v in self.variables:
            axes.append(v.linspace(n_per_dim))

        points: list[dict[str, float]] = []
        for combo in itertools.product(*axes):
            params = {
                self.variables[i].name: combo[i]
                for i in range(len(self.variables))
            }
            if self._check_constraints(params):
                points.append(params)
        return points

    def random_points(
        self,
        n: int,
        seed: int | None = None,
    ) -> list[dict[str, float]]:
        """Generate n random parameter combinations (Latin hypercube)."""
        rng = np.random.default_rng(seed)
        points: list[dict[str, float]] = []
        attempts = 0
        max_attempts = n * 10

        while len(points) < n and attempts < max_attempts:
            params: dict[str, float] = {}
            for v in self.variables:
                val = rng.uniform(v.low, v.high)
                if v.is_integer:
                    val = float(round(val))
                params[v.name] = val

            if self._check_constraints(params):
                points.append(params)
            attempts += 1

        return points

    def _check_constraints(self, params: dict[str, float]) -> bool:
        """Check all constraints against a parameter dict."""
        for constraint in self.constraints:
            try:
                if not constraint(params):
                    return False
            except Exception:
                return False
        return True


# ─────────────────────────────────────────────────────────────────────
# Parameterised antenna designs
# ─────────────────────────────────────────────────────────────────────


class _BaseAntennaDesign:
    """Abstract base for parametric antenna designs."""

    @property
    def family_name(self) -> str:
        """Human-readable name of the antenna family."""
        raise NotImplementedError

    @property
    def design_space(self) -> DesignSpace:
        """Advertise the tuneable parameter space."""
        raise NotImplementedError

    def to_compiler(
        self,
        params: dict[str, float],
        n_bits: int = 10,
        n_steps: int = 300,
        dft_all_components: bool = True,
    ) -> MaxwellAntenna3DCompiler:
        """Build a compiler instance from design parameters."""
        raise NotImplementedError

    def to_dict(self, params: dict[str, float]) -> dict[str, Any]:
        """Serialise a parameter set for attestation."""
        return {
            "family": self.family_name,
            "params": dict(params),
        }


class PatchAntennaDesign(_BaseAntennaDesign):
    """Parametric rectangular microstrip patch antenna.

    Tuneable parameters:
        - patch_width  (normalised to domain)
        - patch_length (normalised to domain)
        - substrate_eps_r
        - substrate_height (normalised)
        - feed_offset_y (normalised, for impedance matching)

    Fixed: ground plane at z=0, probe feed from ground to patch.
    """

    def __init__(
        self,
        freq_center: float = 1.0,
        freq_bandwidth: float = 0.5,
        substrate: Material | None = None,
    ) -> None:
        self._freq_center = freq_center
        self._freq_bandwidth = freq_bandwidth
        self._substrate = substrate

    @property
    def family_name(self) -> str:
        return "rectangular_patch"

    @property
    def design_space(self) -> DesignSpace:
        return DesignSpace(
            variables=[
                DesignVariable(
                    name="patch_width",
                    low=0.15, high=0.50, default=0.38,
                    unit="norm", description="Patch width (x-dir)",
                    n_steps=5,
                ),
                DesignVariable(
                    name="patch_length",
                    low=0.10, high=0.45, default=0.30,
                    unit="norm", description="Patch length (y-dir)",
                    n_steps=5,
                ),
                DesignVariable(
                    name="substrate_eps_r",
                    low=2.0, high=10.0, default=4.4,
                    unit="", description="Substrate permittivity",
                    n_steps=4,
                ),
                DesignVariable(
                    name="substrate_height",
                    low=0.02, high=0.12, default=0.06,
                    unit="norm", description="Substrate thickness",
                    n_steps=4,
                ),
                DesignVariable(
                    name="feed_offset_y",
                    low=-0.15, high=0.0, default=-0.10,
                    unit="norm", description="Feed y-offset from centre",
                    n_steps=4,
                ),
            ],
            constraints=[
                # Patch must fit in domain with margin
                lambda p: p["patch_width"] < 0.8,
                lambda p: p["patch_length"] < 0.8,
                # Substrate must be thinner than 20% of domain
                lambda p: p["substrate_height"] < 0.2,
                # Feed must be within patch footprint
                lambda p: abs(p["feed_offset_y"]) < p["patch_length"] / 2.0,
            ],
        )

    def to_compiler(
        self,
        params: dict[str, float],
        n_bits: int = 10,
        n_steps: int = 300,
        dft_all_components: bool = True,
    ) -> MaxwellAntenna3DCompiler:
        eps_r = params.get("substrate_eps_r", 4.4)
        if self._substrate is not None:
            eps_r = self._substrate.eps_r

        geo = PatchGeometry(
            substrate_eps_r=eps_r,
            substrate_height=params.get("substrate_height", 0.06),
            patch_width=params.get("patch_width", 0.38),
            patch_length=params.get("patch_length", 0.30),
            feed_offset_y=params.get("feed_offset_y", -0.10),
        )

        N = 2 ** n_bits
        dx = 1.0 / N
        h_loop = max(0.03, 4.0 * dx)

        # Feed position: centre of patch footprint vertically
        feed_z = geo.substrate_height * 0.5
        feed_x = 0.5 + geo.feed_offset_x
        feed_y = 0.5 + geo.feed_offset_y

        return MaxwellAntenna3DCompiler(
            n_bits=n_bits,
            n_steps=n_steps,
            geometry="patch",
            geometry_params=geo,
            freq_center=self._freq_center,
            freq_bandwidth=self._freq_bandwidth,
            source_position=(feed_x, feed_y, feed_z),
            source_polarization=2,
            source_width=0.02,
            n_dft_bins=1,
            port=WavePort(
                impedance=1.0,
                gap_size=geo.substrate_height,
                h_loop_half_side=h_loop,
            ),
            dft_all_components=dft_all_components,
        )

    def to_dict(self, params: dict[str, float]) -> dict[str, Any]:
        result = super().to_dict(params)
        if self._substrate is not None:
            result["substrate"] = self._substrate.to_dict()
        return result


class DipoleAntennaDesign(_BaseAntennaDesign):
    """Parametric half-wave dipole antenna.

    Tuneable parameters:
        - arm_length (normalised)
        - wire_radius (normalised)
        - gap_half (normalised)
    """

    def __init__(
        self,
        freq_center: float = 1.0,
        freq_bandwidth: float = 0.5,
    ) -> None:
        self._freq_center = freq_center
        self._freq_bandwidth = freq_bandwidth

    @property
    def family_name(self) -> str:
        return "half_wave_dipole"

    @property
    def design_space(self) -> DesignSpace:
        return DesignSpace(
            variables=[
                DesignVariable(
                    name="arm_length",
                    low=0.10, high=0.40, default=0.25,
                    unit="norm", description="Single arm length",
                    n_steps=7,
                ),
                DesignVariable(
                    name="wire_radius",
                    low=0.005, high=0.04, default=0.015,
                    unit="norm", description="Wire radius",
                    n_steps=4,
                ),
                DesignVariable(
                    name="gap_half",
                    low=0.005, high=0.03, default=0.01,
                    unit="norm", description="Half feed-gap width",
                    n_steps=3,
                ),
            ],
            constraints=[
                # Both arms must fit in domain
                lambda p: 2.0 * (p["arm_length"] + p["gap_half"]) < 0.9,
                # Gap must be at least wire diameter
                lambda p: p["gap_half"] >= p["wire_radius"],
            ],
        )

    def to_compiler(
        self,
        params: dict[str, float],
        n_bits: int = 10,
        n_steps: int = 300,
        dft_all_components: bool = True,
    ) -> MaxwellAntenna3DCompiler:
        geo = DipoleGeometry(
            arm_length=params.get("arm_length", 0.25),
            wire_radius=params.get("wire_radius", 0.015),
            gap_half=params.get("gap_half", 0.01),
        )

        N = 2 ** n_bits
        dx = 1.0 / N
        h_loop = max(0.03, 4.0 * dx)

        return MaxwellAntenna3DCompiler(
            n_bits=n_bits,
            n_steps=n_steps,
            geometry="dipole",
            geometry_params=geo,
            freq_center=self._freq_center,
            freq_bandwidth=self._freq_bandwidth,
            source_position=(0.5, 0.5, 0.5),
            source_polarization=2,
            source_width=0.02,
            n_dft_bins=1,
            port=WavePort(
                impedance=1.0,
                gap_size=2.0 * geo.gap_half,
                h_loop_half_side=h_loop,
            ),
            dft_all_components=dft_all_components,
        )


class EShapedPatchDesign(_BaseAntennaDesign):
    """Parametric E-shaped patch antenna for wideband operation.

    The E-shaped patch is a rectangular patch with two parallel slots
    cut symmetrically about the centre line.  The slots introduce
    additional resonances, widening the impedance bandwidth.

    Tuneable parameters:
        - patch_width, patch_length
        - slot_length (length of each slot)
        - slot_width  (width of each slot)
        - slot_offset (distance from centre to slot centre)
        - substrate_eps_r, substrate_height
        - feed_offset_y

    The geometry is implemented by modifying the conductor mask:
    the two slots are cut from the rectangular patch.
    """

    def __init__(
        self,
        freq_center: float = 1.0,
        freq_bandwidth: float = 0.5,
        substrate: Material | None = None,
    ) -> None:
        self._freq_center = freq_center
        self._freq_bandwidth = freq_bandwidth
        self._substrate = substrate

    @property
    def family_name(self) -> str:
        return "e_shaped_patch"

    @property
    def design_space(self) -> DesignSpace:
        return DesignSpace(
            variables=[
                DesignVariable(
                    name="patch_width",
                    low=0.20, high=0.50, default=0.40,
                    unit="norm", description="Patch width (x-dir)",
                    n_steps=4,
                ),
                DesignVariable(
                    name="patch_length",
                    low=0.15, high=0.45, default=0.32,
                    unit="norm", description="Patch length (y-dir)",
                    n_steps=4,
                ),
                DesignVariable(
                    name="slot_length",
                    low=0.05, high=0.30, default=0.18,
                    unit="norm", description="Slot length (y-dir)",
                    n_steps=4,
                ),
                DesignVariable(
                    name="slot_width",
                    low=0.01, high=0.06, default=0.03,
                    unit="norm", description="Slot width (x-dir)",
                    n_steps=3,
                ),
                DesignVariable(
                    name="slot_offset",
                    low=0.03, high=0.15, default=0.08,
                    unit="norm", description="Centre-to-slot offset (x-dir)",
                    n_steps=4,
                ),
                DesignVariable(
                    name="substrate_eps_r",
                    low=2.0, high=6.0, default=3.55,
                    unit="", description="Substrate permittivity",
                    n_steps=3,
                ),
                DesignVariable(
                    name="substrate_height",
                    low=0.03, high=0.10, default=0.06,
                    unit="norm", description="Substrate thickness",
                    n_steps=3,
                ),
                DesignVariable(
                    name="feed_offset_y",
                    low=-0.12, high=0.0, default=-0.08,
                    unit="norm", description="Feed y-offset from centre",
                    n_steps=3,
                ),
            ],
            constraints=[
                lambda p: p["patch_width"] < 0.8,
                lambda p: p["patch_length"] < 0.8,
                # Slots must fit inside the patch
                lambda p: p["slot_length"] < p["patch_length"] * 0.9,
                lambda p: (p["slot_offset"] + p["slot_width"] / 2.0)
                < p["patch_width"] / 2.0,
                lambda p: abs(p["feed_offset_y"]) < p["patch_length"] / 2.0,
                lambda p: p["substrate_height"] < 0.2,
            ],
        )

    def to_compiler(
        self,
        params: dict[str, float],
        n_bits: int = 10,
        n_steps: int = 300,
        dft_all_components: bool = True,
    ) -> MaxwellAntenna3DCompiler:
        """Build compiler for E-shaped patch.

        Uses a standard PatchGeometry as the base, with slots handled
        by modifying the conductor_mask init function.
        """
        eps_r = params.get("substrate_eps_r", 3.55)
        if self._substrate is not None:
            eps_r = self._substrate.eps_r

        h_sub = params.get("substrate_height", 0.06)

        # Base patch geometry (slots are added via conductor mask override)
        geo = PatchGeometry(
            substrate_eps_r=eps_r,
            substrate_height=h_sub,
            patch_width=params.get("patch_width", 0.40),
            patch_length=params.get("patch_length", 0.32),
            feed_offset_y=params.get("feed_offset_y", -0.08),
        )

        N = 2 ** n_bits
        dx = 1.0 / N
        h_loop = max(0.03, 4.0 * dx)
        feed_z = h_sub * 0.5
        feed_x = 0.5
        feed_y = 0.5 + geo.feed_offset_y

        compiler = MaxwellAntenna3DCompiler(
            n_bits=n_bits,
            n_steps=n_steps,
            geometry="patch",
            geometry_params=geo,
            freq_center=self._freq_center,
            freq_bandwidth=self._freq_bandwidth,
            source_position=(feed_x, feed_y, feed_z),
            source_polarization=2,
            source_width=0.02,
            n_dft_bins=1,
            port=WavePort(
                impedance=1.0,
                gap_size=h_sub,
                h_loop_half_side=h_loop,
            ),
            dft_all_components=dft_all_components,
        )

        # Override the conductor mask to include E-slots
        slot_length = params.get("slot_length", 0.18)
        slot_width = params.get("slot_width", 0.03)
        slot_offset = params.get("slot_offset", 0.08)
        pw = geo.patch_width
        pl = geo.patch_length
        pt = geo.patch_thickness
        gt = geo.ground_thickness

        def _e_shaped_cond_mask(
            x: np.ndarray, y: np.ndarray, z: np.ndarray
        ) -> np.ndarray:
            mask = np.ones_like(x)
            # Ground plane
            mask[z < gt] = 0.0
            # Base patch
            in_patch_x = np.abs(x - 0.5) < pw / 2.0
            in_patch_y = np.abs(y - 0.5) < pl / 2.0
            in_patch_z = np.abs(z - h_sub) < pt / 2.0
            patch = in_patch_x & in_patch_y & in_patch_z
            mask[patch] = 0.0

            # Two E-slots (symmetric about x=0.5)
            for sign in (+1, -1):
                slot_cx = 0.5 + sign * slot_offset
                in_slot_x = np.abs(x - slot_cx) < slot_width / 2.0
                in_slot_y = np.abs(y - 0.5) < slot_length / 2.0
                in_slot_z = np.abs(z - h_sub) < pt / 2.0
                slot_region = in_slot_x & in_slot_y & in_slot_z
                # Remove conductor in slot → mask = 1 (air)
                mask[slot_region] = 1.0
            return mask

        # Inject the custom conductor mask into program metadata
        # This requires monkey-patching the compiled program's metadata
        original_compile = compiler.compile

        def _compile_with_eslots() -> Any:
            prog = original_compile()
            prog.metadata["init_conductor_mask"] = _e_shaped_cond_mask
            # Force dense init for conductor mask (slots not separable)
            if "init_conductor_mask_separable" in prog.metadata:
                del prog.metadata["init_conductor_mask_separable"]
            return prog

        compiler.compile = _compile_with_eslots  # type: ignore[assignment]
        return compiler

    def to_dict(self, params: dict[str, float]) -> dict[str, Any]:
        result = super().to_dict(params)
        if self._substrate is not None:
            result["substrate"] = self._substrate.to_dict()
        return result


class USlotsDesign(_BaseAntennaDesign):
    """Parametric U-slot patch antenna for wideband operation.

    A rectangular patch with a U-shaped slot centred on the radiating
    element.  The U-slot creates a second resonance close to the
    dominant mode, yielding ~30% fractional bandwidth.

    Tuneable parameters:
        - patch_width, patch_length
        - slot_length_y (depth of the U arms)
        - slot_width_x  (overall U opening width)
        - slot_arm_width (width of each U arm)
        - substrate_eps_r, substrate_height
        - feed_offset_y
    """

    def __init__(
        self,
        freq_center: float = 1.0,
        freq_bandwidth: float = 0.5,
        substrate: Material | None = None,
    ) -> None:
        self._freq_center = freq_center
        self._freq_bandwidth = freq_bandwidth
        self._substrate = substrate

    @property
    def family_name(self) -> str:
        return "u_slot_patch"

    @property
    def design_space(self) -> DesignSpace:
        return DesignSpace(
            variables=[
                DesignVariable(
                    name="patch_width",
                    low=0.20, high=0.50, default=0.42,
                    unit="norm", description="Patch width (x-dir)",
                    n_steps=4,
                ),
                DesignVariable(
                    name="patch_length",
                    low=0.15, high=0.45, default=0.34,
                    unit="norm", description="Patch length (y-dir)",
                    n_steps=4,
                ),
                DesignVariable(
                    name="slot_length_y",
                    low=0.05, high=0.25, default=0.16,
                    unit="norm", description="U-arm depth (y-dir)",
                    n_steps=4,
                ),
                DesignVariable(
                    name="slot_width_x",
                    low=0.06, high=0.30, default=0.18,
                    unit="norm", description="U opening width (x-dir)",
                    n_steps=4,
                ),
                DesignVariable(
                    name="slot_arm_width",
                    low=0.008, high=0.04, default=0.02,
                    unit="norm", description="Width of each U arm",
                    n_steps=3,
                ),
                DesignVariable(
                    name="substrate_eps_r",
                    low=2.0, high=6.0, default=3.55,
                    unit="", description="Substrate permittivity",
                    n_steps=3,
                ),
                DesignVariable(
                    name="substrate_height",
                    low=0.03, high=0.10, default=0.06,
                    unit="norm", description="Substrate thickness",
                    n_steps=3,
                ),
                DesignVariable(
                    name="feed_offset_y",
                    low=-0.12, high=0.0, default=-0.08,
                    unit="norm", description="Feed y-offset from centre",
                    n_steps=3,
                ),
            ],
            constraints=[
                lambda p: p["patch_width"] < 0.8,
                lambda p: p["patch_length"] < 0.8,
                # U-slot must fit within patch
                lambda p: p["slot_width_x"] < p["patch_width"] * 0.85,
                lambda p: p["slot_length_y"] < p["patch_length"] * 0.85,
                # Arms must be positive width and fit
                lambda p: 2.0 * p["slot_arm_width"] < p["slot_width_x"],
                lambda p: abs(p["feed_offset_y"]) < p["patch_length"] / 2.0,
                lambda p: p["substrate_height"] < 0.2,
            ],
        )

    def to_compiler(
        self,
        params: dict[str, float],
        n_bits: int = 10,
        n_steps: int = 300,
        dft_all_components: bool = True,
    ) -> MaxwellAntenna3DCompiler:
        eps_r = params.get("substrate_eps_r", 3.55)
        if self._substrate is not None:
            eps_r = self._substrate.eps_r

        h_sub = params.get("substrate_height", 0.06)
        geo = PatchGeometry(
            substrate_eps_r=eps_r,
            substrate_height=h_sub,
            patch_width=params.get("patch_width", 0.42),
            patch_length=params.get("patch_length", 0.34),
            feed_offset_y=params.get("feed_offset_y", -0.08),
        )

        N = 2 ** n_bits
        dx = 1.0 / N
        h_loop = max(0.03, 4.0 * dx)
        feed_z = h_sub * 0.5
        feed_x = 0.5
        feed_y = 0.5 + geo.feed_offset_y

        compiler = MaxwellAntenna3DCompiler(
            n_bits=n_bits,
            n_steps=n_steps,
            geometry="patch",
            geometry_params=geo,
            freq_center=self._freq_center,
            freq_bandwidth=self._freq_bandwidth,
            source_position=(feed_x, feed_y, feed_z),
            source_polarization=2,
            source_width=0.02,
            n_dft_bins=1,
            port=WavePort(impedance=1.0, gap_size=h_sub, h_loop_half_side=h_loop),
            dft_all_components=dft_all_components,
        )

        slot_length_y = params.get("slot_length_y", 0.16)
        slot_width_x = params.get("slot_width_x", 0.18)
        slot_arm_w = params.get("slot_arm_width", 0.02)
        pw = geo.patch_width
        pl = geo.patch_length
        pt = geo.patch_thickness
        gt = geo.ground_thickness

        def _u_slot_cond_mask(
            x: np.ndarray, y: np.ndarray, z: np.ndarray
        ) -> np.ndarray:
            mask = np.ones_like(x)
            # Ground plane
            mask[z < gt] = 0.0
            # Base patch
            in_patch_x = np.abs(x - 0.5) < pw / 2.0
            in_patch_y = np.abs(y - 0.5) < pl / 2.0
            in_patch_z = np.abs(z - h_sub) < pt / 2.0
            patch = in_patch_x & in_patch_y & in_patch_z
            mask[patch] = 0.0

            # U-slot: bottom bar + two vertical arms
            # Centre the U at patch centre
            cx, cy = 0.5, 0.5
            half_w = slot_width_x / 2.0

            # Bottom bar of U: spans full slot_width_x at the bottom
            bar_y_lo = cy - slot_length_y / 2.0
            bar_y_hi = bar_y_lo + slot_arm_w
            in_bar_x = np.abs(x - cx) < half_w
            in_bar_y = (y >= bar_y_lo) & (y <= bar_y_hi)
            in_bar_z = np.abs(z - h_sub) < pt / 2.0
            bar = in_bar_x & in_bar_y & in_bar_z
            mask[bar] = 1.0  # remove conductor

            # Left arm: from bottom bar up
            arm_y_lo = bar_y_lo
            arm_y_hi = cy + slot_length_y / 2.0
            left_x_lo = cx - half_w
            left_x_hi = left_x_lo + slot_arm_w
            in_left_x = (x >= left_x_lo) & (x <= left_x_hi)
            in_left_y = (y >= arm_y_lo) & (y <= arm_y_hi)
            left_arm = in_left_x & in_left_y & in_bar_z
            mask[left_arm] = 1.0

            # Right arm
            right_x_hi = cx + half_w
            right_x_lo = right_x_hi - slot_arm_w
            in_right_x = (x >= right_x_lo) & (x <= right_x_hi)
            in_right_y = (y >= arm_y_lo) & (y <= arm_y_hi)
            right_arm = in_right_x & in_right_y & in_bar_z
            mask[right_arm] = 1.0

            return mask

        original_compile = compiler.compile

        def _compile_with_uslot() -> Any:
            prog = original_compile()
            prog.metadata["init_conductor_mask"] = _u_slot_cond_mask
            if "init_conductor_mask_separable" in prog.metadata:
                del prog.metadata["init_conductor_mask_separable"]
            return prog

        compiler.compile = _compile_with_uslot  # type: ignore[assignment]
        return compiler

    def to_dict(self, params: dict[str, float]) -> dict[str, Any]:
        result = super().to_dict(params)
        if self._substrate is not None:
            result["substrate"] = self._substrate.to_dict()
        return result
