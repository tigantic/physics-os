"""QTT Physics VM — 3-D Maxwell Antenna compiler.

Full 3-D Maxwell equations with materials, PEC conductors, broadband
source excitation, and DFT-based frequency-domain extraction for
S-parameter computation.

Physics (material Maxwell, μ_r = 1):
    ε_r ∂E/∂t  =  c ∇×B  −  σ E  −  J_src(x,y,z,t)
    ∂B/∂t      = −c ∇×E

For lossless dielectrics (σ = 0):
    ∂E/∂t = inv_ε_r ⊙ (c ∇×B − J_src(t))
    ∂B/∂t = −c ∇×E

Where ``inv_ε_r = 1/ε_r(x,y,z)`` is a spatially varying material field.

Interior PEC conductors (antenna/ground-plane metallisation) are modelled
with a binary *conductor mask*: ``mask(x,y,z) = 0`` inside metal,
``1`` elsewhere.  After every E-field update the mask is applied via
Hadamard product, which zeroes E inside conductors identically.

Source excitation is a Gaussian-modulated sinusoidal current:
    J(x,y,z,t) = J_spatial(x,y,z) · a(t)
    a(t) = sin(2π f₀ t) · exp(−(t − t_peak)² / 2τ²)
    τ = 1 / (2π · bandwidth)

DFT accumulators run in the time loop to capture the frequency-domain
field at one or more bins, enabling post-simulation S-parameter
extraction:
    E_f(x,y,z) = Σ_n E(x,y,z,n·dt) · exp(−j 2π f n / N)

Geometry presets
----------------
``dipole``
    Half-wave dipole along *z*, centred in the domain.
    Conductor mask blanks E inside the wire except at the feed gap.
    Source injects z-polarised current at the gap.
    ε_r = 1 everywhere (vacuum).

``patch``
    Rectangular microstrip patch on FR-4 substrate.
    Ground plane at z = 0, dielectric slab up to z = h,
    patch conductor at z = h, probe-feed from ground to patch.
    ε_r = substrate_eps_r for 0 < z < h, 1 above.

Time integration: Störmer–Verlet (symplectic leap-frog).
Boundary conditions: PEC on E at all six domain faces.
Conserved quantity: modified EM energy (approximately conserved when
source is off; deliberately grows during active excitation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..ir import (
    BCKind,
    FieldSpec,
    Instruction,
    Program,
    add,
    bc_apply,
    dft_accumulate,
    grad,
    load_field,
    loop_end,
    loop_start,
    mask_multiply,
    measure,
    negate,
    probe_record,
    scale,
    source_add,
    store_field,
    sub,
    truncate,
)
from .base import BaseCompiler


# ── Geometry descriptors ─────────────────────────────────────────────

@dataclass(frozen=True)
class DipoleGeometry:
    """Half-wave dipole along z, centred in [0,1]³."""

    arm_length: float = 0.25
    """Length of each arm (total = 2 × arm_length)."""
    wire_radius: float = 0.015
    """Radius of the wire (normalised to domain)."""
    gap_half: float = 0.01
    """Half-width of the feed gap at domain centre."""


@dataclass(frozen=True)
class PatchGeometry:
    """Rectangular microstrip patch on dielectric substrate."""

    substrate_eps_r: float = 4.4
    """Relative permittivity of substrate (FR-4)."""
    substrate_height: float = 0.06
    """Substrate thickness (normalised)."""
    patch_width: float = 0.38
    """Patch width along x (normalised)."""
    patch_length: float = 0.30
    """Patch length along y (normalised)."""
    ground_thickness: float = 0.005
    """Ground-plane conductor slab thickness."""
    patch_thickness: float = 0.005
    """Patch conductor slab thickness."""
    feed_offset_x: float = 0.0
    """Feed x-offset from patch centre (normalised)."""
    feed_offset_y: float = -0.10
    """Feed y-offset from patch centre (normalised, towards edge for matching)."""


@dataclass(frozen=True)
class WavePort:
    """Lumped-port definition for S-parameter extraction.

    A lumped port models the feed as a voltage gap with a reference
    impedance.  The runtime records E-field (voltage) and B-field
    (current via loop integral) probes at the port location each step.
    Post-simulation FFT of the probe time series yields V(f) and I(f),
    from which Z_in(f) and S₁₁(f) are extracted.
    """

    impedance: float = 1.0
    """Reference impedance Z₀ in simulation units.

    For normalised simulations (c=1, μ₀=1), the natural impedance
    unit is 1.  Physical mapping to 50 Ω etc. is applied in
    post-processing based on the domain scaling.
    """
    gap_size: float = 0.02
    """Physical size of the voltage gap (integration length for V)."""
    h_loop_half_side: float = 0.02
    """Half-side of the rectangular H-loop for current extraction.

    The loop is centred on the feed point in the plane perpendicular
    to the source polarisation axis.
    """


class MaxwellAntenna3DCompiler(BaseCompiler):
    """Compile 3-D Maxwell equations for antenna simulation.

    Parameters
    ----------
    n_bits : int
        Bits per spatial dimension.  Grid is ``(2^n)³``.
    n_steps : int
        Number of time steps.
    c : float
        Speed of light (normalised, default 1.0).
    dt : float | None
        Time step.  Auto from CFL if *None*.
    geometry : str
        ``"dipole"`` or ``"patch"``.
    geometry_params : DipoleGeometry | PatchGeometry | None
        Override default geometry; inferred from *geometry* if *None*.
    freq_center : float
        Centre frequency of broadband source excitation (normalised).
    freq_bandwidth : float
        Gaussian bandwidth of the excitation pulse.
    source_position : tuple[float, float, float]
        Normalised source location in ``[0,1]³``.
    source_polarization : int
        Polarisation axis for the source current: 0 = x, 1 = y, 2 = z.
    source_width : float
        Spatial width (σ) of the Gaussian source profile.
    n_dft_bins : int
        Number of DFT frequency bins to accumulate (0 = none).
    dft_freq_bin : int
        Which DFT bin index to accumulate (typically near centre freq).
    port : WavePort | None
        Lumped port definition for S-parameter extraction.  When
        provided, voltage and current probes are inserted into the
        time loop and their time series are returned in the execution
        result ``probes`` dict.
    dft_all_components : bool
        When *True*, DFT-accumulate all six E/B components (needed for
        far-field extraction).  When *False* (default), only the
        source-polarisation E-component is accumulated.
    """

    def __init__(
        self,
        n_bits: int = 6,
        n_steps: int = 300,
        c: float = 1.0,
        dt: float | None = None,
        *,
        geometry: str = "dipole",
        geometry_params: DipoleGeometry | PatchGeometry | None = None,
        freq_center: float = 1.0,
        freq_bandwidth: float = 0.5,
        source_position: tuple[float, float, float] = (0.5, 0.5, 0.5),
        source_polarization: int = 2,
        source_width: float = 0.02,
        n_dft_bins: int = 1,
        dft_freq_bin: int = 0,
        port: WavePort | None = None,
        dft_all_components: bool = False,
    ) -> None:
        if geometry not in ("dipole", "patch"):
            raise ValueError(
                f"Unknown geometry '{geometry}'; must be 'dipole' or 'patch'"
            )
        self._n_bits = n_bits
        self._n_steps = n_steps
        self._c = c
        self._geometry = geometry
        self._freq_center = freq_center
        self._freq_bandwidth = freq_bandwidth
        self._source_position = source_position
        self._source_polarization = source_polarization
        self._source_width = source_width
        self._n_dft_bins = n_dft_bins
        self._dft_freq_bin = dft_freq_bin
        self._dft_omega = 2.0 * math.pi * freq_center  # angular frequency for DFT
        self._port = port if port is not None else WavePort()
        self._dft_all_components = dft_all_components

        # Geometry
        if geometry_params is not None:
            self._geo = geometry_params
        elif geometry == "dipole":
            self._geo = DipoleGeometry()
        else:
            self._geo = PatchGeometry()

        # CFL
        N = 2 ** n_bits
        h = 1.0 / N
        if dt is None:
            self._dt = 0.3 * h / (c * np.sqrt(3.0))
        else:
            self._dt = dt

    # ── BaseCompiler interface ───────────────────────────────────────

    @property
    def domain(self) -> str:
        return "maxwell_antenna_3d"

    @property
    def domain_label(self) -> str:
        return f"3D Maxwell Antenna ({self._geometry})"

    # ── Compile ──────────────────────────────────────────────────────

    def compile(self) -> Program:
        c = self._c
        dt = self._dt
        nb = self._n_bits
        bits = (nb, nb, nb)
        dom = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

        cdt = c * dt
        half_cdt = 0.5 * cdt

        # ── Source temporal parameters ───────────────────────────────
        # t_peak: time at which envelope is maximal.
        # Typically 3τ so the pulse starts ~0 at t=0.
        tau = (
            1.0 / (2.0 * np.pi * self._freq_bandwidth)
            if self._freq_bandwidth > 0.0
            else 1e10
        )
        t_peak = 3.0 * tau

        # ── Register map ────────────────────────────────────────────
        # r0-r5   : Ex, Ey, Ez, Bx, By, Bz           (EM fields)
        # r6-r11  : scratch – gradients
        # r12-r17 : scratch – curl sub-results
        # r18-r23 : scratch – scaled updates
        # r24     : inv_eps_r (material: 1/ε_r)
        # r25     : conductor_mask (0 in PEC metal, 1 elsewhere)
        # r26     : J_spatial (source current spatial profile)
        # r27     : scratch – source-scaled temp
        # r28-r29 : DFT accumulators (Re, Im) for pol-axis E at centre
        # r30-r31 : extra scratch
        # r32-r37 : DFT Re accumulators for Ex Ey Ez Bx By Bz (all-comp)
        # r38-r43 : DFT Im accumulators for Ex Ey Ez Bx By Bz (all-comp)
        N_REGS = 48

        R_EX, R_EY, R_EZ = 0, 1, 2
        R_BX, R_BY, R_BZ = 3, 4, 5
        R_INV_EPS = 24
        R_COND_MASK = 25
        R_J_SPATIAL = 26
        R_SCRATCH_SRC = 27
        R_DFT_RE = 28
        R_DFT_IM = 29
        # All-component DFT registers (for far-field extraction)
        R_DFT_ALL_RE_BASE = 32  # 32..37 for Ex..Bz real
        R_DFT_ALL_IM_BASE = 38  # 38..43 for Ex..Bz imag

        pol = self._source_polarization  # 0, 1, or 2
        E_pol_reg = pol  # register of E-component aligned with source

        # ── Port probe configuration ─────────────────────────────────
        # Cyclic permutation: pol → (q, r) axes perpendicular to pol
        _PERP = {0: (1, 2), 1: (2, 0), 2: (0, 1)}
        q_axis, r_axis = _PERP[pol]
        port = self._port
        fp = self._source_position  # feed point (x, y, z)
        delta = port.h_loop_half_side

        # Four H-loop probe offsets: ±δ in the perpendicular plane
        _h_probe_offsets: list[tuple[str, int, float]] = [
            # (probe_name, B-component register, offset direction)
            # B_q at (fp with r-δ) — bottom edge
            (f"B{['x','y','z'][q_axis]}_r_neg", 3 + q_axis, -delta),
            # B_q at (fp with r+δ) — top edge
            (f"B{['x','y','z'][q_axis]}_r_pos", 3 + q_axis, +delta),
            # B_r at (fp with q+δ) — right edge
            (f"B{['x','y','z'][r_axis]}_q_pos", 3 + r_axis, +delta),
            # B_r at (fp with q-δ) — left edge
            (f"B{['x','y','z'][r_axis]}_q_neg", 3 + r_axis, -delta),
        ]

        def _make_probe_coord(
            base: tuple[float, float, float],
            axis: int,
            offset: float,
        ) -> tuple[float, float, float]:
            """Offset a 3D coordinate along one axis."""
            lst = list(base)
            lst[axis] = lst[axis] + offset
            return (lst[0], lst[1], lst[2])

        def _port_probe_instructions() -> list[Instruction]:
            """Voltage + current probes at the lumped port."""
            out: list[Instruction] = []
            # Voltage probe: E_pol at feed point
            out.append(
                probe_record(E_pol_reg, "V_port", fp)
            )
            # Current probes: B_q and B_r at four loop vertices
            # B_q at r-δ
            coord_rn = _make_probe_coord(fp, r_axis, -delta)
            out.append(
                probe_record(
                    _h_probe_offsets[0][1],
                    _h_probe_offsets[0][0],
                    coord_rn,
                )
            )
            # B_q at r+δ
            coord_rp = _make_probe_coord(fp, r_axis, +delta)
            out.append(
                probe_record(
                    _h_probe_offsets[1][1],
                    _h_probe_offsets[1][0],
                    coord_rp,
                )
            )
            # B_r at q+δ
            coord_qp = _make_probe_coord(fp, q_axis, +delta)
            out.append(
                probe_record(
                    _h_probe_offsets[2][1],
                    _h_probe_offsets[2][0],
                    coord_qp,
                )
            )
            # B_r at q-δ
            coord_qn = _make_probe_coord(fp, q_axis, -delta)
            out.append(
                probe_record(
                    _h_probe_offsets[3][1],
                    _h_probe_offsets[3][0],
                    coord_qn,
                )
            )
            return out

        # ── Helper: half-kick E with materials ───────────────────────
        def _half_kick_E(alpha: float) -> list[Instruction]:
            """E += alpha · inv_ε_r ⊙ (c · ∇×B)."""
            out: list[Instruction] = []

            # Component x: curl(B)_x = ∂Bz/∂y − ∂By/∂z
            out += [
                grad(6, R_BZ, dim=1), grad(7, R_BY, dim=2),
                sub(12, 6, 7),
                scale(18, 12, alpha),
                mask_multiply(18, R_INV_EPS, 18),
                add(R_EX, R_EX, 18), truncate(R_EX),
            ]

            # Component y: curl(B)_y = ∂Bx/∂z − ∂Bz/∂x
            out += [
                grad(8, R_BX, dim=2), grad(9, R_BZ, dim=0),
                sub(13, 8, 9),
                scale(19, 13, alpha),
                mask_multiply(19, R_INV_EPS, 19),
                add(R_EY, R_EY, 19), truncate(R_EY),
            ]

            # Component z: curl(B)_z = ∂By/∂x − ∂Bx/∂y
            out += [
                grad(10, R_BY, dim=0), grad(11, R_BX, dim=1),
                sub(14, 10, 11),
                scale(20, 14, alpha),
                mask_multiply(20, R_INV_EPS, 20),
                add(R_EZ, R_EZ, 20), truncate(R_EZ),
            ]

            return out

        def _source_inject() -> list[Instruction]:
            """Inject source current into E-field (full-step amount).

            J is z-polarised (or as configured).  The update is:
                E_pol −= dt · inv_ε_r ⊙ J_spatial · a(t)

            Split across two half-kicks, so each half adds −(dt/2).
            Using SOURCE_ADD with temporal modulation; the call below
            adds the *full* step amount — the caller should place it
            once per full step, not once per half-kick, to avoid double
            counting.
            """
            return [
                load_field(R_J_SPATIAL, "J_spatial"),
                # scale spatial profile by −dt to get correct sign & magnitude
                scale(R_SCRATCH_SRC, R_J_SPATIAL, -dt),
                # apply material coefficient
                mask_multiply(R_SCRATCH_SRC, R_INV_EPS, R_SCRATCH_SRC),
                # inject with temporal modulation
                source_add(
                    E_pol_reg,
                    R_SCRATCH_SRC,
                    freq_center=self._freq_center,
                    bandwidth=self._freq_bandwidth,
                    t_peak=t_peak,
                ),
            ]

        def _conductor_mask_E() -> list[Instruction]:
            """Zero E inside PEC conductor regions."""
            return [
                mask_multiply(R_EX, R_COND_MASK, R_EX),
                mask_multiply(R_EY, R_COND_MASK, R_EY),
                mask_multiply(R_EZ, R_COND_MASK, R_EZ),
            ]

        def _pec_boundary_E() -> list[Instruction]:
            """PEC on E at domain boundaries.

            At a PEC face the *tangential* E-components vanish:
              x-faces → Ey, Ez = 0  (tangential to x-normal face)
              y-faces → Ex, Ez = 0
              z-faces → Ex, Ey = 0

            Each component is zeroed only on the faces where it
            is tangential, via the ``dims`` BC parameter.
            """
            return [
                # Ex tangential at y-faces and z-faces
                bc_apply(R_EX, BCKind.PEC, bc_params={"dims": [1, 2]}),
                # Ey tangential at x-faces and z-faces
                bc_apply(R_EY, BCKind.PEC, bc_params={"dims": [0, 2]}),
                # Ez tangential at x-faces and y-faces
                bc_apply(R_EZ, BCKind.PEC, bc_params={"dims": [0, 1]}),
            ]

        def _full_kick_B() -> list[Instruction]:
            """B −= c·dt · ∇×E  (full step)."""
            out: list[Instruction] = []

            # curl(E)_x = ∂Ez/∂y − ∂Ey/∂z
            out += [
                grad(6, R_EZ, dim=1), grad(7, R_EY, dim=2),
                sub(15, 6, 7),
                scale(21, 15, -cdt),
                add(R_BX, R_BX, 21), truncate(R_BX),
            ]

            # curl(E)_y = ∂Ex/∂z − ∂Ez/∂x
            out += [
                grad(8, R_EX, dim=2), grad(9, R_EZ, dim=0),
                sub(16, 8, 9),
                scale(22, 16, -cdt),
                add(R_BY, R_BY, 22), truncate(R_BY),
            ]

            # curl(E)_z = ∂Ey/∂x − ∂Ex/∂y
            out += [
                grad(10, R_EY, dim=0), grad(11, R_EX, dim=1),
                sub(17, 10, 11),
                scale(23, 17, -cdt),
                add(R_BZ, R_BZ, 23), truncate(R_BZ),
            ]

            # PEC on B: zero *normal* component at each face.
            #   x-faces → Bx = 0 (normal to face)
            #   y-faces → By = 0
            #   z-faces → Bz = 0
            out += [
                bc_apply(R_BX, BCKind.PEC, bc_params={"dims": [0]}),
                bc_apply(R_BY, BCKind.PEC, bc_params={"dims": [1]}),
                bc_apply(R_BZ, BCKind.PEC, bc_params={"dims": [2]}),
            ]

            return out

        def _dft_accumulate_step() -> list[Instruction]:
            """Accumulate DFT of E at source polarisation axis."""
            if self._n_dft_bins <= 0:
                return []
            return [
                dft_accumulate(
                    R_DFT_RE, E_pol_reg,
                    freq_bin=self._dft_freq_bin,
                    component="real",
                    omega=self._dft_omega,
                ),
                dft_accumulate(
                    R_DFT_IM, E_pol_reg,
                    freq_bin=self._dft_freq_bin,
                    component="imag",
                    omega=self._dft_omega,
                ),
            ]

        def _dft_all_components_step() -> list[Instruction]:
            """Accumulate DFT of all 6 E/B components for far-field."""
            if not self._dft_all_components or self._n_dft_bins <= 0:
                return []
            out: list[Instruction] = []
            field_regs = [R_EX, R_EY, R_EZ, R_BX, R_BY, R_BZ]
            for i, freg in enumerate(field_regs):
                out.append(
                    dft_accumulate(
                        R_DFT_ALL_RE_BASE + i, freg,
                        freq_bin=self._dft_freq_bin,
                        component="real",
                        omega=self._dft_omega,
                    )
                )
                out.append(
                    dft_accumulate(
                        R_DFT_ALL_IM_BASE + i, freg,
                        freq_bin=self._dft_freq_bin,
                        component="imag",
                        omega=self._dft_omega,
                    )
                )
            return out

        # ── Load instructions for all-component DFT accumulators ─────
        _dft_all_field_names = [
            "dft_re_Ex", "dft_re_Ey", "dft_re_Ez",
            "dft_re_Bx", "dft_re_By", "dft_re_Bz",
            "dft_im_Ex", "dft_im_Ey", "dft_im_Ez",
            "dft_im_Bx", "dft_im_By", "dft_im_Bz",
        ]

        def _load_dft_all() -> list[Instruction]:
            if not self._dft_all_components or self._n_dft_bins <= 0:
                return []
            out: list[Instruction] = []
            for i in range(6):
                out.append(load_field(R_DFT_ALL_RE_BASE + i, _dft_all_field_names[i]))
                out.append(load_field(R_DFT_ALL_IM_BASE + i, _dft_all_field_names[6 + i]))
            return out

        def _store_dft_all() -> list[Instruction]:
            if not self._dft_all_components or self._n_dft_bins <= 0:
                return []
            out: list[Instruction] = []
            for i in range(6):
                out.append(store_field(R_DFT_ALL_RE_BASE + i, _dft_all_field_names[i]))
                out.append(store_field(R_DFT_ALL_IM_BASE + i, _dft_all_field_names[6 + i]))
            return out

        # ── Assemble time-step loop ─────────────────────────────────
        instructions: list[Instruction] = [
            loop_start(self._n_steps),

            # Load all fields
            load_field(R_EX, "Ex"), load_field(R_EY, "Ey"),
            load_field(R_EZ, "Ez"),
            load_field(R_BX, "Bx"), load_field(R_BY, "By"),
            load_field(R_BZ, "Bz"),
            load_field(R_INV_EPS, "inv_eps_r"),
            load_field(R_COND_MASK, "conductor_mask"),
            # Load DFT accumulators (persist across steps)
            *(
                [
                    load_field(R_DFT_RE, "dft_re"),
                    load_field(R_DFT_IM, "dft_im"),
                ]
                if self._n_dft_bins > 0
                else []
            ),
            *_load_dft_all(),

            # ── Half-kick E ────────────────────────────────────────
            *_half_kick_E(half_cdt),

            # ── Source injection (full step) ───────────────────────
            *_source_inject(),

            # ── Conductor mask + PEC on E ──────────────────────────
            *_conductor_mask_E(),
            *_pec_boundary_E(),

            # ── Full-kick B ────────────────────────────────────────
            *_full_kick_B(),

            # ── Half-kick E ────────────────────────────────────────
            *_half_kick_E(half_cdt),

            # ── Conductor mask + PEC on E (again after second half) ─
            *_conductor_mask_E(),
            *_pec_boundary_E(),

            # ── Port probes (V + I, after final E and B updates) ──
            *_port_probe_instructions(),

            # ── DFT accumulation ───────────────────────────────────
            *_dft_accumulate_step(),
            *_dft_all_components_step(),

            # ── Store & measure ────────────────────────────────────
            store_field(R_EX, "Ex"), store_field(R_EY, "Ey"),
            store_field(R_EZ, "Ez"),
            store_field(R_BX, "Bx"), store_field(R_BY, "By"),
            store_field(R_BZ, "Bz"),
            # Persist DFT accumulators so they survive across steps
            *(
                [
                    store_field(R_DFT_RE, "dft_re"),
                    store_field(R_DFT_IM, "dft_im"),
                ]
                if self._n_dft_bins > 0
                else []
            ),
            *_store_dft_all(),
            measure(R_EX, "Ex"),

            loop_end(),
        ]

        # ── Field specs & separable init functions ──────────────────
        field_specs, metadata = self._build_fields_and_metadata(bits, dom)

        return Program(
            domain=self.domain,
            domain_label=self.domain_label,
            n_registers=N_REGS,
            fields=field_specs,
            instructions=instructions,
            dt=self._dt,
            n_steps=self._n_steps,
            params={
                "c": self._c,
                "geometry": self._geometry,
                "freq_center": self._freq_center,
                "freq_bandwidth": self._freq_bandwidth,
                "dft_omega": self._dft_omega,
                "port_impedance": port.impedance,
                "port_gap_size": port.gap_size,
                "port_h_loop_half_side": port.h_loop_half_side,
                "source_polarization": pol,
            },
            metadata=metadata,
        )

    # ── Field & metadata construction ────────────────────────────────

    def _build_fields_and_metadata(
        self,
        bits: tuple[int, int, int],
        dom: tuple[tuple[float, float], ...],
    ) -> tuple[dict[str, FieldSpec], dict[str, Any]]:
        """Build FieldSpec dict and metadata for the chosen geometry."""
        sw = self._source_width
        sp = self._source_position

        # ── Common separable helpers ────────────────────────────────
        def _gaussian_1d(
            centre: float, sigma: float
        ) -> "Callable[[NDArray], NDArray]":
            def fn(t: NDArray) -> NDArray:
                return np.exp(-((t - centre) ** 2) / (2.0 * sigma ** 2))
            return fn

        def _ones(t: NDArray) -> NDArray:
            return np.ones_like(t)

        def _zeros(t: NDArray) -> NDArray:
            return np.zeros_like(t)

        # ── Geometry-specific material & conductor fields ───────────
        if self._geometry == "dipole":
            inv_eps_separable, cond_mask_separable, cond_mask_fn = (
                self._dipole_fields()
            )
            source_sep = [
                _gaussian_1d(sp[0], sw),
                _gaussian_1d(sp[1], sw),
                _gaussian_1d(sp[2], sw),
            ]
        else:  # patch
            inv_eps_separable, cond_mask_separable, cond_mask_fn = (
                self._patch_fields()
            )
            geo = self._geo
            assert isinstance(geo, PatchGeometry)
            feed_x = 0.5 + geo.feed_offset_x
            feed_y = 0.5 + geo.feed_offset_y
            feed_z = geo.substrate_height * 0.5  # midpoint of substrate
            source_sep = [
                _gaussian_1d(feed_x, sw),
                _gaussian_1d(feed_y, sw),
                _gaussian_1d(feed_z, sw),
            ]

        # ── EM field init: all zero (antenna starts cold) ───────────
        field_specs: dict[str, FieldSpec] = {}
        for name in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
            field_specs[name] = FieldSpec(
                name=name,
                n_dims=3,
                bits_per_dim=bits,
                bc=BCKind.PEC,
                bc_params={"domain": dom},
                initial_fn=f"init_{name}",
                conserved_quantity="em_energy" if name == "Ex" else "",
            )

        # Material field: inv_eps_r = 1/ε_r(x,y,z)
        field_specs["inv_eps_r"] = FieldSpec(
            name="inv_eps_r",
            n_dims=3,
            bits_per_dim=bits,
            bc=BCKind.PERIODIC,
            bc_params={"domain": dom},
            initial_fn="init_inv_eps_r",
        )

        # Conductor mask: 0 in metal, 1 elsewhere
        field_specs["conductor_mask"] = FieldSpec(
            name="conductor_mask",
            n_dims=3,
            bits_per_dim=bits,
            bc=BCKind.PERIODIC,
            bc_params={"domain": dom},
            initial_fn="init_conductor_mask",
        )

        # Source spatial profile
        field_specs["J_spatial"] = FieldSpec(
            name="J_spatial",
            n_dims=3,
            bits_per_dim=bits,
            bc=BCKind.PERIODIC,
            bc_params={"domain": dom},
            initial_fn="init_J_spatial",
        )

        # DFT accumulators (zero-initialised)
        if self._n_dft_bins > 0:
            for dft_name in ("dft_re", "dft_im"):
                field_specs[dft_name] = FieldSpec(
                    name=dft_name,
                    n_dims=3,
                    bits_per_dim=bits,
                    bc=BCKind.PERIODIC,
                    bc_params={"domain": dom},
                    initial_fn=f"init_{dft_name}",
                )

        # All-component DFT accumulators for far-field (zero-initialised)
        if self._dft_all_components and self._n_dft_bins > 0:
            _comp_names = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
            for comp in _comp_names:
                for part in ("re", "im"):
                    fn = f"dft_{part}_{comp}"
                    field_specs[fn] = FieldSpec(
                        name=fn,
                        n_dims=3,
                        bits_per_dim=bits,
                        bc=BCKind.PERIODIC,
                        bc_params={"domain": dom},
                        initial_fn=f"init_{fn}",
                    )

        # ── Dense init (CPU fallback) ───────────────────────────────
        def _init_zero_3d(x: NDArray, y: NDArray, z: NDArray) -> NDArray:
            return np.zeros_like(x)

        def _init_ones_3d(x: NDArray, y: NDArray, z: NDArray) -> NDArray:
            return np.ones_like(x)

        def _init_source_3d(x: NDArray, y: NDArray, z: NDArray) -> NDArray:
            return (
                np.exp(-((x - sp[0]) ** 2) / (2.0 * sw ** 2))
                * np.exp(-((y - sp[1]) ** 2) / (2.0 * sw ** 2))
                * np.exp(-((z - sp[2]) ** 2) / (2.0 * sw ** 2))
            )

        def _invariant_fn(fields: dict) -> float:
            h = fields["Ex"].grid_spacing(0)
            dV = h ** 3
            energy = 0.0
            for name in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
                f = fields[name]
                energy += f.inner(f)
            return 0.5 * dV * energy

        metadata: dict[str, Any] = {
            # Dense init fns (CPU fallback)
            "init_Ex": _init_zero_3d,
            "init_Ey": _init_zero_3d,
            "init_Ez": _init_zero_3d,
            "init_Bx": _init_zero_3d,
            "init_By": _init_zero_3d,
            "init_Bz": _init_zero_3d,
            "init_J_spatial": _init_source_3d,
            "init_inv_eps_r": _init_ones_3d,
            "init_conductor_mask": cond_mask_fn,
            # Separable factors for GPU-native init (no dense grids)
            # Zero EM fields: runtime detects no separable → uses .zeros()
            "init_inv_eps_r_separable": inv_eps_separable,
            "init_conductor_mask_separable": cond_mask_separable,
            "init_J_spatial_separable": source_sep,
            # DFT accumulators: zero init (handled by .zeros())
            "invariant_fn": _invariant_fn,
            "invariant": "em_energy",
            "equations": (
                "ε_r ∂E/∂t = c∇×B − J_src(t), "
                "∂B/∂t = −c∇×E, "
                "PEC: E_tan = 0"
            ),
        }

        if self._n_dft_bins > 0:
            metadata["init_dft_re"] = _init_zero_3d
            metadata["init_dft_im"] = _init_zero_3d

        # All-component DFT dense init functions (zero)
        if self._dft_all_components and self._n_dft_bins > 0:
            for comp in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
                for part in ("re", "im"):
                    metadata[f"init_dft_{part}_{comp}"] = _init_zero_3d

        return field_specs, metadata

    # ── Dipole geometry ──────────────────────────────────────────────

    def _dipole_fields(
        self,
    ) -> tuple[list, list | None, "Callable"]:
        """Build separable factors for half-wave dipole geometry.

        Returns
        -------
        inv_eps_separable
            Separable 1-D factors for 1/ε_r (ones for vacuum).
        cond_mask_separable
            Separable factors for conductor mask, or *None* if the
            mask is too complex for rank-1 (runtime falls back to dense
            init and compresses via SVD).
        cond_mask_fn
            Dense 3-D init function for the conductor mask.
        """
        geo = self._geo
        assert isinstance(geo, DipoleGeometry)

        # 1/ε_r = 1 everywhere (vacuum)
        def _one(t: NDArray) -> NDArray:
            return np.ones_like(t)

        inv_eps_sep = [_one, _one, _one]

        # Conductor mask: 0 on wire, 1 elsewhere.
        # Wire occupies (x ≈ 0.5, y ≈ 0.5) for
        # z ∈ [0.5 − arm − gap, 0.5 − gap] ∪ [0.5 + gap, 0.5 + arm + gap].
        arm = geo.arm_length
        gap = geo.gap_half
        r = geo.wire_radius
        cx, cy, cz = 0.5, 0.5, 0.5

        def _cond_mask_3d(
            x: NDArray, y: NDArray, z: NDArray
        ) -> NDArray:
            in_wire_xy = ((x - cx) ** 2 + (y - cy) ** 2) < r ** 2
            lower_arm = (z >= (cz - arm - gap)) & (z <= (cz - gap))
            upper_arm = (z >= (cz + gap)) & (z <= (cz + arm + gap))
            in_wire = in_wire_xy & (lower_arm | upper_arm)
            mask = np.ones_like(x)
            mask[in_wire] = 0.0
            return mask

        # Conductor mask strategy:
        # ≤ 8 bits/dim (256³): dense init builds exact conductor geometry,
        #   runtime compresses to QTT via SVD.
        # > 8 bits/dim: thin-wire approximation — no explicit conductor
        #   mesh.  The current source alone models the antenna; the
        #   PEC domain boundaries provide the ground plane.  We provide
        #   all-ones separable so from_separable creates a rank-1 mask.
        total_bits = sum(self._n_bits for _ in range(3))
        if total_bits > 24:  # 3 × 8 = 24 → 256³ threshold
            cond_mask_sep: list | None = [_one, _one, _one]
        else:
            cond_mask_sep = None  # use dense init

        return inv_eps_sep, cond_mask_sep, _cond_mask_3d

    # ── Patch geometry ───────────────────────────────────────────────

    def _patch_fields(
        self,
    ) -> tuple[list, list | None, "Callable"]:
        """Build fields for rectangular microstrip patch."""
        geo = self._geo
        assert isinstance(geo, PatchGeometry)

        eps_r = geo.substrate_eps_r
        h_sub = geo.substrate_height

        # inv_eps_r: 1/ε_r in substrate, 1 above.
        # ε_r(x,y,z) = ε_r if z < h_sub else 1.0
        # → inv_ε_r = 1/ε_r if z < h_sub else 1.0
        # This is separable: 1(x) · 1(y) · f(z)
        inv_eps_val = 1.0 / eps_r

        def _inv_eps_z(z: NDArray) -> NDArray:
            return np.where(z < h_sub, inv_eps_val, 1.0)

        def _one(t: NDArray) -> NDArray:
            return np.ones_like(t)

        inv_eps_sep = [_one, _one, _inv_eps_z]

        # Conductor mask: 0 on ground plane and patch, 1 elsewhere.
        pw = geo.patch_width
        pl = geo.patch_length
        gt = geo.ground_thickness
        pt = geo.patch_thickness

        def _cond_mask_3d(
            x: NDArray, y: NDArray, z: NDArray
        ) -> NDArray:
            mask = np.ones_like(x)
            # Ground plane: full xy extent at z ≈ 0
            ground = z < gt
            mask[ground] = 0.0
            # Patch: rectangle at z ≈ h_sub
            in_patch_x = np.abs(x - 0.5) < pw / 2.0
            in_patch_y = np.abs(y - 0.5) < pl / 2.0
            in_patch_z = np.abs(z - h_sub) < pt / 2.0
            patch = in_patch_x & in_patch_y & in_patch_z
            mask[patch] = 0.0
            return mask

        return inv_eps_sep, None, _cond_mask_3d
