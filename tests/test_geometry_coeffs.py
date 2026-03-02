"""Phase C: Geometry Coefficient Compilation — Tests.

Verifies:
  C1) GeometryCompiler produces valid QTT coefficient fields
  C2) Procedural geometry primitives (rectangle, circle, step)
  C3) Penalization, distance proxy, and material fields
  C4) Rank stays bounded for typical shapes
  C5) All geometry internals are QTT — never dense in execution path
  C6) Convenience functions for standard CFD geometries
"""

from __future__ import annotations

import numpy as np
import pytest

from ontic.engine.vm.qtt_tensor import QTTTensor
from ontic.engine.vm.compilers.geometry_coeffs import (
    GeometryCompiler,
    GeometryScene,
    GeometrySpec,
    GeometryPrimitive,
    CompiledGeometry,
    compile_cylinder_in_channel,
    compile_backward_facing_step,
    compile_lid_driven_cavity,
)


# ══════════════════════════════════════════════════════════════════════
# C1: Basic compilation
# ══════════════════════════════════════════════════════════════════════

class TestGeometryCompilation:
    """Tests for GeometryCompiler producing valid QTT fields."""

    def test_compile_produces_qtt_tensors(self) -> None:
        """All compiled fields should be QTTTensor instances."""
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.RECTANGLE,
                    params={
                        "center": [0.5, 0.5],
                        "half_extents": [0.1, 0.1],
                    },
                ),
            ],
            bits_per_dim=(7, 7),
            domain=((0.0, 1.0), (0.0, 1.0)),
        )
        compiler = GeometryCompiler(max_rank=32)
        result = compiler.compile(scene)

        assert isinstance(result.solid_mask, QTTTensor)
        assert isinstance(result.penalization, QTTTensor)
        assert isinstance(result.distance_proxy, QTTTensor)

    def test_solid_mask_is_binary(self) -> None:
        """Solid mask should be approximately 0 or 1."""
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.RECTANGLE,
                    params={
                        "center": [0.5, 0.5],
                        "half_extents": [0.2, 0.2],
                    },
                ),
            ],
            bits_per_dim=(6, 6),
            domain=((0.0, 1.0), (0.0, 1.0)),
        )
        compiler = GeometryCompiler(max_rank=32)
        result = compiler.compile(scene)
        dense = result.solid_mask.to_dense()
        # Values should be near 0 or 1 (QTT compression adds small noise)
        assert np.all(dense >= -0.1)
        assert np.all(dense <= 1.1)

    def test_penalization_proportional_to_mask(self) -> None:
        """Penalization field = β × mask."""
        beta = 1e4
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.CIRCLE,
                    params={"center": [0.5, 0.5], "radius": 0.1},
                ),
            ],
            bits_per_dim=(6, 6),
            domain=((0.0, 1.0), (0.0, 1.0)),
            penalization_strength=beta,
        )
        compiler = GeometryCompiler(max_rank=32)
        result = compiler.compile(scene)
        mask = result.solid_mask.to_dense()
        pen = result.penalization.to_dense()
        # Inside solid: pen ≈ β, outside: pen ≈ 0
        expected = beta * mask
        np.testing.assert_allclose(pen, expected, atol=beta * 0.1)

    def test_field_specs_populated(self) -> None:
        """Compiled geometry should include IR-compatible field specs."""
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.RECTANGLE,
                    params={
                        "center": [0.5],
                        "half_extents": [0.2],
                    },
                ),
            ],
            bits_per_dim=(8,),
            domain=((0.0, 1.0),),
        )
        compiler = GeometryCompiler()
        result = compiler.compile(scene)
        assert "chi_solid" in result.field_specs
        assert "beta_penalization" in result.field_specs
        assert "phi_distance" in result.field_specs


# ══════════════════════════════════════════════════════════════════════
# C2: Procedural geometry primitives
# ══════════════════════════════════════════════════════════════════════

class TestGeometryPrimitives:
    """Tests for individual geometry primitives."""

    def test_rectangle_1d(self) -> None:
        """Rectangle indicator in 1D should be 1 inside [0.3, 0.7]."""
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.RECTANGLE,
                    params={
                        "center": [0.5],
                        "half_extents": [0.2],
                    },
                ),
            ],
            bits_per_dim=(8,),
            domain=((0.0, 1.0),),
        )
        compiler = GeometryCompiler(max_rank=32)
        result = compiler.compile(scene)
        dense = result.solid_mask.to_dense()
        N = 2 ** 8
        x = np.linspace(0, 1, N, endpoint=False)
        # Interior points should be ~1, exterior ~0
        inside = (x >= 0.3) & (x <= 0.7)
        assert np.mean(dense[inside]) > 0.8
        assert np.mean(dense[~inside]) < 0.2

    def test_circle_2d(self) -> None:
        """Circle at (0.5, 0.5) radius 0.2."""
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.CIRCLE,
                    params={"center": [0.5, 0.5], "radius": 0.2},
                ),
            ],
            bits_per_dim=(6, 6),
            domain=((0.0, 1.0), (0.0, 1.0)),
        )
        compiler = GeometryCompiler(max_rank=32)
        result = compiler.compile(scene)
        N = 2 ** 6
        dense = result.solid_mask.to_dense().reshape(N, N)
        # Center should be solid
        center_val = dense[N // 2, N // 2]
        assert center_val > 0.5
        # Far corner should be fluid
        corner_val = dense[0, 0]
        assert corner_val < 0.5

    def test_step_geometry(self) -> None:
        """Backward-facing step should have solid in lower-left."""
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.STEP,
                    params={"step_x": 0.3, "step_y": 0.4},
                ),
            ],
            bits_per_dim=(6, 6),
            domain=((0.0, 1.0), (0.0, 1.0)),
        )
        compiler = GeometryCompiler(max_rank=32)
        result = compiler.compile(scene)
        N = 2 ** 6
        dense = result.solid_mask.to_dense().reshape(N, N)
        # Point in solid region (x < 0.3, y < 0.4) → index ~(4, 6)
        assert dense[4, 6] > 0.5
        # Point far downstream (x > 0.3) → fluid
        assert dense[N - 2, N // 2] < 0.5

    def test_custom_geometry(self) -> None:
        """Custom indicator function should be accepted."""
        def my_indicator(x: np.ndarray) -> np.ndarray:
            return (x < 0.5).astype(np.float64)

        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.CUSTOM,
                    params={"indicator_fn": my_indicator},
                ),
            ],
            bits_per_dim=(8,),
            domain=((0.0, 1.0),),
        )
        compiler = GeometryCompiler(max_rank=16)
        result = compiler.compile(scene)
        dense = result.solid_mask.to_dense()
        N = 2 ** 8
        x = np.linspace(0, 1, N, endpoint=False)
        # Left half should be solid
        assert np.mean(dense[x < 0.4]) > 0.8
        assert np.mean(dense[x > 0.6]) < 0.2


# ══════════════════════════════════════════════════════════════════════
# C3: Distance proxy and material fields
# ══════════════════════════════════════════════════════════════════════

class TestDerivedFields:
    """Tests for distance proxy and material coefficient fields."""

    def test_distance_proxy_sign(self) -> None:
        """Distance proxy: negative inside solid, positive outside."""
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.RECTANGLE,
                    params={
                        "center": [0.5],
                        "half_extents": [0.2],
                    },
                ),
            ],
            bits_per_dim=(8,),
            domain=((0.0, 1.0),),
        )
        compiler = GeometryCompiler(max_rank=32)
        result = compiler.compile(scene)
        dense = result.distance_proxy.to_dense()
        N = 2 ** 8
        x = np.linspace(0, 1, N, endpoint=False)
        # Deep inside solid (x=0.5) → negative
        mid_idx = N // 2
        assert dense[mid_idx] < 0, f"Inside solid should be negative, got {dense[mid_idx]}"
        # Well outside (x=0.1) → positive
        far_idx = int(0.1 * N)
        assert dense[far_idx] > 0, f"Outside solid should be positive, got {dense[far_idx]}"

    def test_multi_material(self) -> None:
        """Multiple materials should produce a material coefficient field."""
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.RECTANGLE,
                    params={"center": [0.3], "half_extents": [0.1]},
                    material_id=1,
                ),
                GeometrySpec(
                    primitive=GeometryPrimitive.RECTANGLE,
                    params={"center": [0.7], "half_extents": [0.1]},
                    material_id=2,
                ),
            ],
            bits_per_dim=(8,),
            domain=((0.0, 1.0),),
        )
        compiler = GeometryCompiler(max_rank=32)
        result = compiler.compile(scene)
        assert result.material_coeff is not None
        assert isinstance(result.material_coeff, QTTTensor)

    def test_single_material_no_coeff(self) -> None:
        """Single material → no material coefficient field."""
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.RECTANGLE,
                    params={"center": [0.5], "half_extents": [0.2]},
                    material_id=0,
                ),
            ],
            bits_per_dim=(8,),
            domain=((0.0, 1.0),),
        )
        compiler = GeometryCompiler(max_rank=32)
        result = compiler.compile(scene)
        assert result.material_coeff is None


# ══════════════════════════════════════════════════════════════════════
# C4: Rank stability
# ══════════════════════════════════════════════════════════════════════

class TestRankStability:
    """Tests that rank stays bounded for typical geometries."""

    def test_rank_stats_recorded(self) -> None:
        """Compiled geometry should record rank stats (PRIVATE)."""
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.CIRCLE,
                    params={"center": [0.5, 0.5], "radius": 0.15},
                ),
            ],
            bits_per_dim=(7, 7),
            domain=((0.0, 1.0), (0.0, 1.0)),
        )
        compiler = GeometryCompiler(max_rank=32)
        result = compiler.compile(scene)
        assert "solid_mask_max_rank" in result.rank_stats
        assert "penalization_max_rank" in result.rank_stats
        assert "distance_proxy_max_rank" in result.rank_stats

    def test_rank_within_budget(self) -> None:
        """All fields should respect the max_rank budget."""
        max_rank = 24
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.CIRCLE,
                    params={"center": [0.5, 0.5], "radius": 0.15},
                ),
            ],
            bits_per_dim=(7, 7),
            domain=((0.0, 1.0), (0.0, 1.0)),
        )
        compiler = GeometryCompiler(max_rank=max_rank)
        result = compiler.compile(scene)
        for name, rank in result.rank_stats.items():
            assert rank <= max_rank, (
                f"{name} rank {rank} exceeds budget {max_rank}"
            )


# ══════════════════════════════════════════════════════════════════════
# C6: Convenience geometry functions
# ══════════════════════════════════════════════════════════════════════

class TestConvenienceGeometry:
    """Tests for pre-built CFD geometry patterns."""

    def test_cylinder_in_channel(self) -> None:
        """Cylinder-in-channel compiles without error."""
        result = compile_cylinder_in_channel(
            bits_per_dim=(6, 6), max_rank=24,
        )
        assert isinstance(result, CompiledGeometry)
        assert result.solid_mask.max_rank > 0

    def test_backward_facing_step(self) -> None:
        """Backward-facing step compiles without error."""
        result = compile_backward_facing_step(
            bits_per_dim=(6, 6), max_rank=24,
        )
        assert isinstance(result, CompiledGeometry)

    def test_lid_driven_cavity(self) -> None:
        """Lid-driven cavity compiles without error."""
        result = compile_lid_driven_cavity(
            bits_per_dim=(6, 6), max_rank=24,
        )
        assert isinstance(result, CompiledGeometry)

    def test_custom_geometry_missing_fn_raises(self) -> None:
        """CUSTOM without indicator_fn should raise ValueError."""
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.CUSTOM,
                    params={},  # missing indicator_fn
                ),
            ],
            bits_per_dim=(6,),
            domain=((0.0, 1.0),),
        )
        compiler = GeometryCompiler()
        with pytest.raises(ValueError, match="CUSTOM.*indicator_fn"):
            compiler.compile(scene)
