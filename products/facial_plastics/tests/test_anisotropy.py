"""Tests for sim/anisotropy.py — anisotropic tissue constitutive models."""

from __future__ import annotations

import math

import numpy as np
import pytest

from products.facial_plastics.sim.anisotropy import (
    AnisotropicModel,
    FiberArchitecture,
    FiberFamily,
    FiberField,
    LANGERS_LINE_DIRECTIONS,
    TISSUE_ANISOTROPY,
    _tensor_to_voigt_4th,
    _voigt_to_tensor,
    build_muscle_fiber_field,
    build_skin_fiber_field,
    build_smas_fiber_field,
    build_fiber_field_for_tissue,
    compute_effective_stiffness,
    compute_fiber_mooney_rivlin_stress,
    compute_hgo_stress,
    compute_transverse_iso_stress,
    evaluate_anisotropic_stress,
    get_anisotropy_params,
)
from products.facial_plastics.core.types import StructureType


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def identity_F() -> np.ndarray:
    """Identity deformation gradient (undeformed state)."""
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def uniaxial_stretch_F() -> np.ndarray:
    """10% uniaxial stretch in x-direction, incompressible."""
    lam = 1.10
    lam_t = 1.0 / math.sqrt(lam)
    return np.diag([lam, lam_t, lam_t])


@pytest.fixture
def simple_family() -> FiberFamily:
    return FiberFamily(
        direction=np.array([1.0, 0.0, 0.0]),
        k1=50.0e3,
        k2=10.0,
        kappa_dispersion=0.1,
    )


@pytest.fixture
def two_families() -> list:
    return [
        FiberFamily(direction=np.array([1.0, 0.0, 0.0]), k1=50.0e3, k2=10.0, kappa_dispersion=0.1),
        FiberFamily(direction=np.array([0.0, 1.0, 0.0]), k1=50.0e3, k2=10.0, kappa_dispersion=0.1),
    ]


# ── FiberFamily ───────────────────────────────────────────────────

class TestFiberFamily:
    def test_normalize_direction(self) -> None:
        ff = FiberFamily(direction=np.array([3.0, 0.0, 0.0]))
        np.testing.assert_allclose(ff.direction, [1.0, 0.0, 0.0], atol=1e-12)

    def test_unit_direction(self, simple_family: FiberFamily) -> None:
        norm = np.linalg.norm(simple_family.direction)
        assert norm == pytest.approx(1.0, abs=1e-12)

    def test_zero_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            FiberFamily(direction=np.array([0.0, 0.0, 0.0]))

    def test_default_params(self, simple_family: FiberFamily) -> None:
        assert simple_family.k1 == 50.0e3
        assert simple_family.k2 == 10.0
        assert simple_family.kappa_dispersion == 0.1


# ── FiberField ────────────────────────────────────────────────────

class TestFiberField:
    def test_empty_field(self) -> None:
        ff = FiberField()
        assert len(ff.families_for_element(0)) == 0

    def test_default_families(self, simple_family: FiberFamily) -> None:
        ff = FiberField(default_families=[simple_family])
        fams = ff.families_for_element(999)
        assert len(fams) == 1

    def test_per_element_families(self, simple_family: FiberFamily) -> None:
        other = FiberFamily(direction=np.array([0.0, 1.0, 0.0]))
        ff = FiberField(
            element_families={5: [other]},
            default_families=[simple_family],
        )
        assert len(ff.families_for_element(5)) == 1
        assert ff.families_for_element(5)[0].direction[1] == pytest.approx(1.0)
        assert ff.families_for_element(0)[0].direction[0] == pytest.approx(1.0)


# ── Fiber field builders ─────────────────────────────────────────

class TestFiberFieldBuilders:
    def test_skin_fiber_field(self) -> None:
        ff = build_skin_fiber_field(region="cheek")
        assert ff.architecture == FiberArchitecture.BIAXIAL_SYMMETRIC
        assert len(ff.default_families) == 2
        for fam in ff.default_families:
            assert np.linalg.norm(fam.direction) == pytest.approx(1.0, abs=1e-10)

    def test_skin_regions(self) -> None:
        for region in LANGERS_LINE_DIRECTIONS:
            ff = build_skin_fiber_field(region=region)
            assert len(ff.default_families) == 2

    def test_muscle_fiber_field(self) -> None:
        ff = build_muscle_fiber_field(np.array([0.0, 1.0, 0.0]))
        assert ff.architecture == FiberArchitecture.UNIAXIAL
        assert len(ff.default_families) == 1

    def test_smas_fiber_field(self) -> None:
        ff = build_smas_fiber_field()
        assert ff.architecture == FiberArchitecture.PLANAR_RANDOM
        assert len(ff.default_families) == 6  # default n_directions
        # Total k1 should equal original
        total_k1 = sum(f.k1 for f in ff.default_families)
        assert total_k1 == pytest.approx(30.0e3, rel=0.01)

    def test_build_for_skin_tissue(self) -> None:
        ff = build_fiber_field_for_tissue(StructureType.SKIN_ENVELOPE)
        assert ff is not None
        assert len(ff.default_families) == 2

    def test_build_for_isotropic_tissue(self) -> None:
        ff = build_fiber_field_for_tissue(StructureType.FAT_SUBCUTANEOUS)
        assert ff is None

    def test_build_for_muscle(self) -> None:
        ff = build_fiber_field_for_tissue(
            StructureType.MUSCLE_MIMETIC,
            muscle_direction=np.array([1.0, 0.0, 0.0]),
        )
        assert ff is not None
        assert ff.architecture == FiberArchitecture.UNIAXIAL


# ── Voigt utilities ───────────────────────────────────────────────

class TestVoigtUtilities:
    def test_voigt_to_tensor_diagonal(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        T = _voigt_to_tensor(v)
        assert T.shape == (3, 3)
        assert T[0, 0] == 1.0
        assert T[1, 1] == 2.0
        assert T[2, 2] == 3.0

    def test_voigt_to_tensor_symmetric(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 0.5, 0.3, 0.1])
        T = _voigt_to_tensor(v)
        np.testing.assert_allclose(T, T.T)

    def test_tensor_to_voigt_4th_shape(self) -> None:
        T = np.zeros((3, 3, 3, 3))
        V = _tensor_to_voigt_4th(T)
        assert V.shape == (6, 6)

    def test_tensor_to_voigt_4th_identity(self) -> None:
        # Identity 4th-order tensor: T_ijkl = delta_ij * delta_kl
        T = np.einsum("ij,kl->ijkl", np.eye(3), np.eye(3))
        V = _tensor_to_voigt_4th(T)
        assert V[0, 0] == pytest.approx(1.0)
        assert V[0, 1] == pytest.approx(1.0)
        assert V[3, 3] == pytest.approx(0.0)


# ── HGO stress ────────────────────────────────────────────────────

class TestHGOStress:
    def test_identity_gives_zero_stress(
        self, identity_F: np.ndarray, two_families: list,
    ) -> None:
        S, C = compute_hgo_stress(identity_F, 1e4, 1e5, two_families)
        assert S.shape == (6,)
        assert C.shape == (6, 6)
        # At identity, stress should be essentially zero
        assert np.max(np.abs(S)) < 1.0  # numerical tolerance

    def test_uniaxial_stretch_stress(
        self, uniaxial_stretch_F: np.ndarray, simple_family: FiberFamily,
    ) -> None:
        S, C = compute_hgo_stress(
            uniaxial_stretch_F, 30.0e3, 300.0e3, [simple_family],
        )
        # Stretching in fiber direction should produce positive S11
        assert S[0] > 0  # tension in x

    def test_tangent_symmetry(
        self, uniaxial_stretch_F: np.ndarray, two_families: list,
    ) -> None:
        _, C = compute_hgo_stress(
            uniaxial_stretch_F, 30.0e3, 300.0e3, two_families,
        )
        # Material tangent must be symmetric
        np.testing.assert_allclose(C, C.T, atol=1e-6)

    def test_tangent_positive_definite(
        self, identity_F: np.ndarray, simple_family: FiberFamily,
    ) -> None:
        _, C = compute_hgo_stress(identity_F, 30.0e3, 300.0e3, [simple_family])
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > -1e-3)  # allow small numerical negative


# ── Transversely isotropic ────────────────────────────────────────

class TestTransverseIso:
    def test_identity_near_zero(
        self, identity_F: np.ndarray, simple_family: FiberFamily,
    ) -> None:
        S, C = compute_transverse_iso_stress(identity_F, 1e4, 1e5, simple_family)
        assert np.max(np.abs(S)) < 1.0

    def test_stretch_stress(
        self, uniaxial_stretch_F: np.ndarray, simple_family: FiberFamily,
    ) -> None:
        S, _ = compute_transverse_iso_stress(
            uniaxial_stretch_F, 30.0e3, 300.0e3, simple_family,
        )
        assert S[0] > 0  # tension along fiber


# ── Fiber-reinforced Mooney-Rivlin ────────────────────────────────

class TestFiberMooneyRivlin:
    def test_identity_near_zero(
        self, identity_F: np.ndarray, two_families: list,
    ) -> None:
        S, C = compute_fiber_mooney_rivlin_stress(
            identity_F, 0.5e3, 0.05e3, 1e5, two_families,
        )
        assert np.max(np.abs(S)) < 1.0

    def test_tangent_symmetric(
        self, uniaxial_stretch_F: np.ndarray, two_families: list,
    ) -> None:
        _, C = compute_fiber_mooney_rivlin_stress(
            uniaxial_stretch_F, 0.5e3, 0.05e3, 1e5, two_families,
        )
        np.testing.assert_allclose(C, C.T, atol=1e-6)


# ── Dispatcher ────────────────────────────────────────────────────

class TestEvaluateAnisotropicStress:
    def test_hgo_dispatch(
        self, uniaxial_stretch_F: np.ndarray, two_families: list,
    ) -> None:
        S, C = evaluate_anisotropic_stress(
            uniaxial_stretch_F,
            AnisotropicModel.HGO,
            {"mu": 30e3, "kappa": 300e3},
            two_families,
        )
        assert S.shape == (6,)

    def test_transverse_iso_dispatch(
        self, uniaxial_stretch_F: np.ndarray, simple_family: FiberFamily,
    ) -> None:
        S, C = evaluate_anisotropic_stress(
            uniaxial_stretch_F,
            AnisotropicModel.TRANSVERSE_ISO_NEOHOOKEAN,
            {"mu": 30e3, "kappa": 300e3},
            [simple_family],
        )
        assert S.shape == (6,)

    def test_fiber_mr_dispatch(
        self, uniaxial_stretch_F: np.ndarray, two_families: list,
    ) -> None:
        S, C = evaluate_anisotropic_stress(
            uniaxial_stretch_F,
            AnisotropicModel.FIBER_MOONEY_RIVLIN,
            {"C1": 0.5e3, "C2": 0.05e3, "kappa": 1e5},
            two_families,
        )
        assert S.shape == (6,)

    def test_transverse_iso_requires_family(
        self, uniaxial_stretch_F: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="requires at least one"):
            evaluate_anisotropic_stress(
                uniaxial_stretch_F,
                AnisotropicModel.TRANSVERSE_ISO_NEOHOOKEAN,
                {"mu": 30e3, "kappa": 300e3},
                [],
            )


# ── Effective stiffness ──────────────────────────────────────────

class TestEffectiveStiffness:
    def test_isotropic_tangent(self) -> None:
        # Build an isotropic tangent
        C = 1e4 * np.eye(6)
        props = compute_effective_stiffness(C)
        assert props["E_1"] > 0
        assert props["anisotropy_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_anisotropic_tangent(
        self, uniaxial_stretch_F: np.ndarray, simple_family: FiberFamily,
    ) -> None:
        _, C = compute_hgo_stress(
            uniaxial_stretch_F, 30e3, 300e3, [simple_family],
        )
        props = compute_effective_stiffness(C)
        # With fiber-reinforcement in x, E_1 should be larger
        assert props["E_1"] > 0
        assert props["anisotropy_ratio"] >= 1.0


# ── Tissue anisotropy table ───────────────────────────────────────

class TestTissueAnisotropy:
    def test_skin_has_hgo(self) -> None:
        params = get_anisotropy_params(StructureType.SKIN_ENVELOPE)
        assert params is not None
        assert params["model"] == AnisotropicModel.HGO

    def test_fat_is_isotropic(self) -> None:
        assert get_anisotropy_params(StructureType.FAT_SUBCUTANEOUS) is None

    def test_muscle_is_transverse(self) -> None:
        params = get_anisotropy_params(StructureType.MUSCLE_MIMETIC)
        assert params is not None
        assert params["model"] == AnisotropicModel.TRANSVERSE_ISO_NEOHOOKEAN
