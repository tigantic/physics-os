"""Test suite for the SDF geometry library and generalized ImmersedBoundary.

Covers all 12 SDF implementations:
  1. CircleSDF            7. FlatPlateSDF
  2. EllipseSDF           8. FinArraySDF
  3. RectangleSDF         9. PipeBendSDF
  4. RoundedRectSDF      10. ConcentricAnnulusSDF
  5. WedgeSDF            11. MultiBodySDF
  6. NACA4DigitSDF       12. StepSDF

Plus: SDFGeometry protocol tests, ImmersedBoundary generalization tests.
"""

from __future__ import annotations

import math

import pytest
import torch

from ontic.cfd.sdf import (
    CircleSDF,
    ConcentricAnnulusSDF,
    EllipseSDF,
    FinArraySDF,
    FlatPlateSDF,
    MultiBodySDF,
    NACA4DigitSDF,
    PipeBendSDF,
    RectangleSDF,
    RoundedRectSDF,
    SDFGeometry,
    StepSDF,
    WedgeSDF,
)
from ontic.cfd.geometry import ImmersedBoundary


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _grid(
    xlo: float, xhi: float, ylo: float, yhi: float, n: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(xlo, xhi, n, dtype=torch.float64)
    y = torch.linspace(ylo, yhi, n, dtype=torch.float64)
    return torch.meshgrid(x, y, indexing="ij")


def _assert_protocol(sdf: SDFGeometry) -> None:
    """Verify the full SDFGeometry protocol for any instance."""
    bb = sdf.bounding_box
    assert len(bb) == 4
    assert bb[1] > bb[0], "x_max > x_min"
    assert bb[3] > bb[2], "y_max > y_min"
    assert sdf.characteristic_length > 0

    X, Y = _grid(bb[0] - 0.5, bb[1] + 0.5, bb[2] - 0.5, bb[3] + 0.5, n=32)
    d = sdf.sdf(X, Y)
    assert d.shape == X.shape

    mask = sdf.is_inside(X, Y)
    assert mask.shape == X.shape
    assert mask.dtype == torch.bool

    nx, ny = sdf.normal(X, Y)
    assert nx.shape == X.shape
    assert ny.shape == X.shape
    mag = torch.sqrt(nx**2 + ny**2)
    assert torch.allclose(mag, torch.ones_like(mag), atol=1e-4)

    sx, sy = sdf.surface_points(n=32)
    assert sx.shape[0] == 32 or sx.shape[0] <= 32  # may be fewer
    assert sy.shape[0] == sx.shape[0]


# ──────────────────────────────────────────────────────────────────
# 1. CircleSDF
# ──────────────────────────────────────────────────────────────────

class TestCircleSDF:
    def test_protocol(self) -> None:
        _assert_protocol(CircleSDF(0.0, 0.0, 1.0))

    def test_sdf_center_negative(self) -> None:
        c = CircleSDF(0.0, 0.0, 1.0)
        d = c.sdf(torch.tensor([0.0]), torch.tensor([0.0]))
        assert d.item() == pytest.approx(-1.0)

    def test_sdf_on_surface_zero(self) -> None:
        c = CircleSDF(0.0, 0.0, 1.0)
        d = c.sdf(torch.tensor([1.0]), torch.tensor([0.0]))
        assert abs(d.item()) < 1e-12

    def test_sdf_outside_positive(self) -> None:
        c = CircleSDF(0.0, 0.0, 1.0)
        d = c.sdf(torch.tensor([2.0]), torch.tensor([0.0]))
        assert d.item() == pytest.approx(1.0)

    def test_is_inside(self) -> None:
        c = CircleSDF(0.0, 0.0, 1.0)
        assert c.is_inside(torch.tensor([0.5]), torch.tensor([0.0])).item()
        assert not c.is_inside(torch.tensor([1.5]), torch.tensor([0.0])).item()

    def test_characteristic_length(self) -> None:
        c = CircleSDF(0.0, 0.0, 0.5)
        assert c.characteristic_length == pytest.approx(1.0)  # diameter

    def test_surface_points_on_circle(self) -> None:
        c = CircleSDF(0.0, 0.0, 1.0)
        sx, sy = c.surface_points(n=100)
        dist = torch.sqrt(sx**2 + sy**2)
        assert torch.allclose(dist, torch.ones_like(dist), atol=1e-10)

    def test_invalid_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            CircleSDF(0.0, 0.0, -1.0)


# ──────────────────────────────────────────────────────────────────
# 2. EllipseSDF
# ──────────────────────────────────────────────────────────────────

class TestEllipseSDF:
    def test_protocol(self) -> None:
        _assert_protocol(EllipseSDF(0.0, 0.0, 2.0, 1.0))

    def test_center_inside(self) -> None:
        e = EllipseSDF(0.0, 0.0, 2.0, 1.0)
        d = e.sdf(torch.tensor([0.0]), torch.tensor([0.0]))
        assert d.item() < 0

    def test_far_outside(self) -> None:
        e = EllipseSDF(0.0, 0.0, 2.0, 1.0)
        d = e.sdf(torch.tensor([10.0]), torch.tensor([0.0]))
        assert d.item() > 0

    def test_on_tip(self) -> None:
        e = EllipseSDF(0.0, 0.0, 2.0, 1.0)
        d = e.sdf(torch.tensor([2.0]), torch.tensor([0.0]))
        assert abs(d.item()) < 0.05  # approximate SDF

    def test_invalid_axes_raises(self) -> None:
        with pytest.raises(ValueError, match="semi-axes"):
            EllipseSDF(0, 0, -1, 1)


# ──────────────────────────────────────────────────────────────────
# 3. RectangleSDF
# ──────────────────────────────────────────────────────────────────

class TestRectangleSDF:
    def test_protocol(self) -> None:
        _assert_protocol(RectangleSDF(0.0, 0.0, 2.0, 1.0))

    def test_center_negative(self) -> None:
        r = RectangleSDF(0.0, 0.0, 2.0, 1.0)
        d = r.sdf(torch.tensor([0.0]), torch.tensor([0.0]))
        assert d.item() < 0

    def test_corner_distance(self) -> None:
        r = RectangleSDF(0.0, 0.0, 2.0, 2.0)
        d = r.sdf(torch.tensor([2.0], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64))
        expected = math.sqrt(2.0)  # dist from (2,2) to corner at (1,1)
        assert d.item() == pytest.approx(expected, abs=1e-6)

    def test_edge_distance(self) -> None:
        r = RectangleSDF(0.0, 0.0, 2.0, 2.0)
        d = r.sdf(torch.tensor([1.5], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64))
        assert d.item() == pytest.approx(0.5, abs=1e-6)


# ──────────────────────────────────────────────────────────────────
# 4. RoundedRectSDF
# ──────────────────────────────────────────────────────────────────

class TestRoundedRectSDF:
    def test_protocol(self) -> None:
        _assert_protocol(RoundedRectSDF(0.0, 0.0, 4.0, 2.0, 0.3))

    def test_center_inside(self) -> None:
        rr = RoundedRectSDF(0.0, 0.0, 4.0, 2.0, 0.3)
        d = rr.sdf(torch.tensor([0.0]), torch.tensor([0.0]))
        assert d.item() < 0

    def test_far_outside(self) -> None:
        rr = RoundedRectSDF(0.0, 0.0, 4.0, 2.0, 0.3)
        d = rr.sdf(torch.tensor([10.0]), torch.tensor([10.0]))
        assert d.item() > 0

    def test_zero_radius_is_rectangle(self) -> None:
        rr = RoundedRectSDF(0.0, 0.0, 2.0, 2.0, 0.0)
        r = RectangleSDF(0.0, 0.0, 2.0, 2.0)
        x = torch.tensor([1.5, 0.5, 0.0])
        y = torch.tensor([0.0, 0.0, 1.5])
        assert torch.allclose(rr.sdf(x, y), r.sdf(x, y), atol=1e-10)


# ──────────────────────────────────────────────────────────────────
# 5. WedgeSDF
# ──────────────────────────────────────────────────────────────────

class TestWedgeSDF:
    def test_protocol(self) -> None:
        _assert_protocol(WedgeSDF(0.0, 0.0, math.radians(15.0), 1.0))

    def test_ahead_of_tip_positive(self) -> None:
        w = WedgeSDF(0.0, 0.0, math.radians(15.0), 1.0)
        d = w.sdf(torch.tensor([-0.5]), torch.tensor([0.0]))
        assert d.item() > 0

    def test_on_centerline_inside(self) -> None:
        w = WedgeSDF(0.0, 0.0, math.radians(15.0), 1.0)
        d = w.sdf(torch.tensor([0.5]), torch.tensor([0.0]))
        assert d.item() < 0

    def test_invalid_angle_raises(self) -> None:
        with pytest.raises(ValueError, match="half_angle"):
            WedgeSDF(0, 0, 0, 1.0)


# ──────────────────────────────────────────────────────────────────
# 6. NACA4DigitSDF
# ──────────────────────────────────────────────────────────────────

class TestNACA4DigitSDF:
    def test_protocol(self) -> None:
        _assert_protocol(NACA4DigitSDF(chord=1.0, naca_code="0012"))

    def test_center_inside(self) -> None:
        n = NACA4DigitSDF(chord=1.0, naca_code="0012")
        d = n.sdf(torch.tensor([0.5]), torch.tensor([0.0]))
        assert d.item() < 0  # mid-chord inside body

    def test_far_outside(self) -> None:
        n = NACA4DigitSDF(chord=1.0, naca_code="0012")
        d = n.sdf(torch.tensor([0.5]), torch.tensor([1.0]))
        assert d.item() > 0

    def test_cambered_profile(self) -> None:
        n = NACA4DigitSDF(chord=1.0, naca_code="2412")
        assert n.characteristic_length == 1.0
        bb = n.bounding_box
        assert bb[3] > abs(bb[2])  # upper surface higher for cambered

    def test_invalid_code_raises(self) -> None:
        with pytest.raises(ValueError, match="4 digits"):
            NACA4DigitSDF(chord=1.0, naca_code="012")


# ──────────────────────────────────────────────────────────────────
# 7. FlatPlateSDF
# ──────────────────────────────────────────────────────────────────

class TestFlatPlateSDF:
    def test_protocol(self) -> None:
        _assert_protocol(FlatPlateSDF(0.0, 0.0, 1.0))

    def test_characteristic_is_length(self) -> None:
        fp = FlatPlateSDF(0.0, 0.0, 2.5)
        assert fp.characteristic_length == pytest.approx(2.5)


# ──────────────────────────────────────────────────────────────────
# 8. FinArraySDF
# ──────────────────────────────────────────────────────────────────

class TestFinArraySDF:
    def test_protocol(self) -> None:
        _assert_protocol(FinArraySDF(
            base_y=0.0, n_fins=5, fin_height=0.05,
            fin_thickness=0.002, fin_spacing=0.01,
        ))

    def test_inside_fin(self) -> None:
        fa = FinArraySDF(
            base_y=0.0, n_fins=3, fin_height=0.1,
            fin_thickness=0.01, fin_spacing=0.05,
        )
        # Center fin at x=0, extends from base_y+base_thickness to +fin_height
        d = fa.sdf(torch.tensor([0.0]), torch.tensor([0.06]))
        assert d.item() < 0

    def test_invalid_nfins_raises(self) -> None:
        with pytest.raises(ValueError, match="n_fins"):
            FinArraySDF(0.0, 0, 0.05, 0.01, 0.01)


# ──────────────────────────────────────────────────────────────────
# 9. PipeBendSDF
# ──────────────────────────────────────────────────────────────────

class TestPipeBendSDF:
    def test_protocol(self) -> None:
        _assert_protocol(PipeBendSDF(0.1, 0.2, 1.0))

    def test_invalid_radii_raises(self) -> None:
        with pytest.raises(ValueError, match="inner_radius"):
            PipeBendSDF(0.5, 0.2, 1.0)  # inner > outer


# ──────────────────────────────────────────────────────────────────
# 10. ConcentricAnnulusSDF
# ──────────────────────────────────────────────────────────────────

class TestConcentricAnnulusSDF:
    def test_protocol(self) -> None:
        _assert_protocol(ConcentricAnnulusSDF(0.0, 0.0, 0.5, 1.0))

    def test_center_inside_inner(self) -> None:
        ca = ConcentricAnnulusSDF(0.0, 0.0, 0.5, 1.0)
        d = ca.sdf(torch.tensor([0.0]), torch.tensor([0.0]))
        assert d.item() < 0  # inside inner cylinder (solid)

    def test_annular_gap_outside(self) -> None:
        ca = ConcentricAnnulusSDF(0.0, 0.0, 0.5, 1.0)
        d = ca.sdf(torch.tensor([0.75]), torch.tensor([0.0]))
        assert d.item() > 0  # in the fluid annulus


# ──────────────────────────────────────────────────────────────────
# 11. MultiBodySDF
# ──────────────────────────────────────────────────────────────────

class TestMultiBodySDF:
    def test_protocol(self) -> None:
        _assert_protocol(MultiBodySDF([
            CircleSDF(0.0, 0.0, 0.5),
            CircleSDF(2.0, 0.0, 0.3),
        ]))

    def test_union_inside_either(self) -> None:
        mb = MultiBodySDF([
            CircleSDF(0.0, 0.0, 0.5),
            CircleSDF(2.0, 0.0, 0.3),
        ])
        # Inside first body
        assert mb.is_inside(torch.tensor([0.0]), torch.tensor([0.0])).item()
        # Inside second body
        assert mb.is_inside(torch.tensor([2.0]), torch.tensor([0.0])).item()
        # Between bodies – outside
        assert not mb.is_inside(torch.tensor([1.0]), torch.tensor([0.0])).item()

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            MultiBodySDF([])


# ──────────────────────────────────────────────────────────────────
# 12. StepSDF
# ──────────────────────────────────────────────────────────────────

class TestStepSDF:
    def test_protocol(self) -> None:
        _assert_protocol(StepSDF(step_x=1.0, step_height=0.5, channel_height=1.0))

    def test_upstream_solid(self) -> None:
        s = StepSDF(step_x=1.0, step_height=0.5, channel_height=1.0)
        # Upper region upstream of step should be solid
        d = s.sdf(torch.tensor([0.5]), torch.tensor([0.8]))
        assert d.item() < 0

    def test_downstream_fluid(self) -> None:
        s = StepSDF(step_x=1.0, step_height=0.5, channel_height=1.0)
        # Same y-position but downstream of step should be fluid
        d = s.sdf(torch.tensor([1.5]), torch.tensor([0.8]))
        assert d.item() > 0

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="step_height"):
            StepSDF(1.0, 1.0, 1.0)  # step_height == channel_height


# ──────────────────────────────────────────────────────────────────
# ImmersedBoundary with SDFGeometry
# ──────────────────────────────────────────────────────────────────

class TestImmersedBoundaryWithSDF:
    def test_circle_ib_mask(self) -> None:
        """ImmersedBoundary with CircleSDF identifies interior cells."""
        c = CircleSDF(0.0, 0.0, 0.5)
        x = torch.linspace(-1, 1, 64, dtype=torch.float64)
        Y, X = torch.meshgrid(x, x, indexing="ij")
        ib = ImmersedBoundary(c, X, Y)
        # Mask should be True inside the circle
        expected = (X**2 + Y**2) < 0.5**2
        assert torch.all(ib.mask == expected)

    def test_circle_ib_ghost_cells_exist(self) -> None:
        """Ghost cells should be present around the surface."""
        c = CircleSDF(0.0, 0.0, 0.5)
        x = torch.linspace(-1, 1, 64, dtype=torch.float64)
        Y, X = torch.meshgrid(x, x, indexing="ij")
        ib = ImmersedBoundary(c, X, Y)
        assert ib.ghost_mask.any()

    def test_circle_ib_apply(self) -> None:
        """Apply ghost-cell BC on a synthetic conservative field."""
        c = CircleSDF(0.0, 0.0, 0.3)
        x = torch.linspace(-1, 1, 32, dtype=torch.float64)
        Y, X = torch.meshgrid(x, x, indexing="ij")
        ib = ImmersedBoundary(c, X, Y)

        # Synthetic conservative variables: (4, Ny, Nx)
        rho = torch.ones(32, 32, dtype=torch.float64)
        rhou = 0.5 * torch.ones(32, 32, dtype=torch.float64)
        rhov = torch.zeros(32, 32, dtype=torch.float64)
        E = 2.5 * torch.ones(32, 32, dtype=torch.float64)
        U = torch.stack([rho, rhou, rhov, E])

        U_bc = ib.apply(U)
        assert U_bc.shape == U.shape
        # Ghost cells should differ from original (velocity reflected)
        if ib.ghost_mask.any():
            assert not torch.allclose(
                U_bc[:, ib.ghost_mask],
                U[:, ib.ghost_mask],
            )

    def test_naca_ib(self) -> None:
        """ImmersedBoundary works with NACA airfoil."""
        naca = NACA4DigitSDF(chord=1.0, naca_code="0012")
        x = torch.linspace(-0.5, 1.5, 48, dtype=torch.float64)
        y = torch.linspace(-0.5, 0.5, 48, dtype=torch.float64)
        Y, X = torch.meshgrid(y, x, indexing="ij")
        ib = ImmersedBoundary(naca, X, Y)
        assert ib.mask.any()  # some cells inside airfoil
        assert (~ib.mask).any()  # some cells outside

    def test_rectangle_ib(self) -> None:
        """ImmersedBoundary works with RectangleSDF."""
        r = RectangleSDF(0.0, 0.0, 0.4, 0.2)
        x = torch.linspace(-1, 1, 32, dtype=torch.float64)
        Y, X = torch.meshgrid(x, x, indexing="ij")
        ib = ImmersedBoundary(r, X, Y)
        assert ib.mask.any()
        assert ib.ghost_mask.any()

    def test_multibody_ib(self) -> None:
        """ImmersedBoundary works with MultiBodySDF (tandem cylinders)."""
        mb = MultiBodySDF([
            CircleSDF(-0.5, 0.0, 0.2),
            CircleSDF(0.5, 0.0, 0.2),
        ])
        x = torch.linspace(-1.5, 1.5, 48, dtype=torch.float64)
        Y, X = torch.meshgrid(x, x, indexing="ij")
        ib = ImmersedBoundary(mb, X, Y)
        assert ib.mask.any()
