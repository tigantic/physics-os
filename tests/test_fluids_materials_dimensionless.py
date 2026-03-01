"""Tests for fluids, materials, and dimensionless number libraries."""

from __future__ import annotations

import math

import pytest

from physics_os.core.fluids import (
    AIR,
    WATER,
    SEAWATER,
    GLYCEROL,
    ENGINE_OIL,
    MERCURY,
    ETHANOL,
    HYDROGEN,
    NITROGEN,
    HELIUM,
    CARBON_DIOXIDE,
    LIQUID_SODIUM,
    FluidProperties,
)
from physics_os.core.materials import (
    ALUMINUM_6061,
    COPPER_C110,
    STAINLESS_STEEL_304,
    CARBON_STEEL_1020,
    TITANIUM_TI6AL4V,
    SILICON,
    GLASS_PYREX,
    MaterialProperties,
)
from physics_os.core.dimensionless import (
    reynolds_number,
    mach_number,
    prandtl_number,
    rayleigh_number,
    grashof_number,
    peclet_number,
    strouhal_number,
    richardson_number,
    dean_number,
    nusselt_flat_plate_laminar,
    nusselt_flat_plate_turbulent,
    nusselt_cylinder_crossflow,
    nusselt_natural_convection_vertical_plate,
    strouhal_cylinder,
    drag_cylinder,
    drag_flat_plate_friction,
    drag_sphere,
    classify_flow,
)


# ──────────────────────────────────────────────────────────────────
# Fluids
# ──────────────────────────────────────────────────────────────────

class TestFluidProperties:
    def test_all_12_registered(self) -> None:
        keys = FluidProperties.list_all()
        assert len(keys) == 12

    def test_get_by_name(self) -> None:
        assert FluidProperties.get("air") is AIR
        assert FluidProperties.get("Water") is WATER

    def test_unknown_fluid_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown fluid"):
            FluidProperties.get("kryptonite")

    @pytest.mark.parametrize("fluid", [
        AIR, WATER, SEAWATER, GLYCEROL, ENGINE_OIL, MERCURY,
        ETHANOL, HYDROGEN, NITROGEN, HELIUM, CARBON_DIOXIDE, LIQUID_SODIUM,
    ])
    def test_positive_properties(self, fluid: FluidProperties) -> None:
        assert fluid.density > 0
        assert fluid.dynamic_viscosity > 0
        assert fluid.kinematic_viscosity > 0
        assert fluid.thermal_conductivity > 0
        assert fluid.specific_heat_cp > 0
        assert fluid.prandtl_number > 0
        assert fluid.speed_of_sound > 0

    def test_air_sanity(self) -> None:
        assert 1.0 < AIR.density < 1.5
        assert 0.7 < AIR.prandtl_number < 0.8
        assert 340 < AIR.speed_of_sound < 350

    def test_water_sanity(self) -> None:
        assert 990 < WATER.density < 1010
        assert 6.5 < WATER.prandtl_number < 7.5

    def test_mercury_low_prandtl(self) -> None:
        assert MERCURY.prandtl_number < 0.05  # liquid metal


# ──────────────────────────────────────────────────────────────────
# Materials
# ──────────────────────────────────────────────────────────────────

class TestMaterialProperties:
    def test_all_7_registered(self) -> None:
        keys = MaterialProperties.list_all()
        assert len(keys) == 7

    def test_get_by_name(self) -> None:
        assert MaterialProperties.get("aluminum_6061") is ALUMINUM_6061
        assert MaterialProperties.get("Copper C110") is COPPER_C110

    def test_unknown_material_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown material"):
            MaterialProperties.get("unobtainium")

    @pytest.mark.parametrize("mat", [
        ALUMINUM_6061, COPPER_C110, STAINLESS_STEEL_304,
        CARBON_STEEL_1020, TITANIUM_TI6AL4V, SILICON, GLASS_PYREX,
    ])
    def test_positive_properties(self, mat: MaterialProperties) -> None:
        assert mat.density > 0
        assert mat.thermal_conductivity > 0
        assert mat.specific_heat_cp > 0
        assert mat.thermal_diffusivity > 0
        assert 0.0 <= mat.emissivity <= 1.0

    def test_copper_highest_conductivity(self) -> None:
        all_k = [m.thermal_conductivity for m in [
            ALUMINUM_6061, COPPER_C110, STAINLESS_STEEL_304,
            CARBON_STEEL_1020, TITANIUM_TI6AL4V, SILICON, GLASS_PYREX,
        ]]
        assert max(all_k) == COPPER_C110.thermal_conductivity


# ──────────────────────────────────────────────────────────────────
# Dimensionless numbers
# ──────────────────────────────────────────────────────────────────

class TestDimensionlessNumbers:
    def test_reynolds_number(self) -> None:
        re = reynolds_number(10.0, 1.0, 1e-5)
        assert re == pytest.approx(1e6)

    def test_reynolds_invalid(self) -> None:
        with pytest.raises(ValueError):
            reynolds_number(10, 1, 0)

    def test_mach_number(self) -> None:
        assert mach_number(170, 340) == pytest.approx(0.5)

    def test_prandtl_number(self) -> None:
        pr = prandtl_number(AIR.dynamic_viscosity, AIR.specific_heat_cp, AIR.thermal_conductivity)
        assert pr == pytest.approx(AIR.prandtl_number, rel=0.02)

    def test_rayleigh_number(self) -> None:
        ra = rayleigh_number(9.81, 3.43e-3, 10.0, 0.1, 1.516e-5, 2.12e-5)
        assert ra > 0

    def test_grashof_number(self) -> None:
        gr = grashof_number(9.81, 3.43e-3, 10.0, 0.1, 1.516e-5)
        assert gr > 0

    def test_peclet_number(self) -> None:
        pe = peclet_number(1.0, 0.1, 1e-5)
        assert pe == pytest.approx(1e4)

    def test_strouhal_number(self) -> None:
        st = strouhal_number(10.0, 0.01, 1.0)
        assert st == pytest.approx(0.1)

    def test_richardson_number(self) -> None:
        ri = richardson_number(9.81, 3.43e-3, 10.0, 0.1, 1.0)
        assert ri > 0

    def test_dean_number(self) -> None:
        de = dean_number(10000, 0.05, 0.5)
        assert de > 0


# ──────────────────────────────────────────────────────────────────
# Correlations
# ──────────────────────────────────────────────────────────────────

class TestCorrelations:
    def test_nusselt_flat_plate_laminar(self) -> None:
        res = nusselt_flat_plate_laminar(1e5, 0.72, 0.025, 1.0)
        assert 50 < res.nu < 250
        assert res.h > 0
        assert "Incropera" in res.source

    def test_nusselt_flat_plate_turbulent(self) -> None:
        res = nusselt_flat_plate_turbulent(1e7, 0.72, 0.025, 1.0)
        assert res.nu > 0
        assert res.h > 0

    def test_nusselt_cylinder_crossflow(self) -> None:
        res = nusselt_cylinder_crossflow(1e4, 0.72, 0.025, 0.01)
        assert res.nu > 0

    def test_nusselt_natural_convection(self) -> None:
        res = nusselt_natural_convection_vertical_plate(1e8, 0.72, 0.025, 0.1)
        assert res.nu > 0

    def test_strouhal_cylinder_subcritical(self) -> None:
        res = strouhal_cylinder(1e4, 0.01, 1.0)
        assert 0.19 < res.st < 0.23
        assert res.frequency_hz > 0

    def test_strouhal_cylinder_below_onset(self) -> None:
        res = strouhal_cylinder(10, 0.01, 0.1)
        assert res.st == 0.0

    def test_drag_cylinder(self) -> None:
        res = drag_cylinder(1e4)
        assert 1.0 < res.cd < 1.5

    def test_drag_flat_plate_laminar(self) -> None:
        res = drag_flat_plate_friction(1e4)
        assert res.cd == pytest.approx(1.328 / 1e4**0.5, rel=1e-6)

    def test_drag_sphere_newton(self) -> None:
        res = drag_sphere(1e4)
        assert res.cd == pytest.approx(0.44)


# ──────────────────────────────────────────────────────────────────
# Flow classification
# ──────────────────────────────────────────────────────────────────

class TestFlowClassification:
    def test_laminar_incompressible(self) -> None:
        fc = classify_flow(re=500, ma=0.1)
        assert "laminar" in fc.label
        assert "incompressible" in fc.label

    def test_turbulent_supersonic(self) -> None:
        fc = classify_flow(re=1e6, ma=2.0)
        assert "turbulent" in fc.label
        assert "supersonic" in fc.label

    def test_creeping(self) -> None:
        fc = classify_flow(re=0.1)
        assert "Stokes" in fc.label
