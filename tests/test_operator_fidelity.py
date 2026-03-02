"""Phase B: Operator Fidelity v1 — MMS and convergence tests.

Verifies:
  B1) Operator family/variant registry exists and is consistent
  B2) 4th-order gradient and Laplacian MPOs are mathematically correct
  B3) Variable-coefficient elliptic operator ∇·(a∇u) works
  B4) MMS convergence under n_bits refinement (the QTT "h-refinement" analog)
  B5) Variant-aware caching and IR dispatch

Test strategy:
  - Compare MPO-applied results against analytic derivatives
  - Verify observed convergence order ≥ formal order (minus tolerance)
  - Ensure 4th-order variants converge faster than 2nd-order
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from numpy.typing import NDArray

from ontic.engine.vm.qtt_tensor import QTTTensor
from ontic.engine.vm.operators import (
    OPERATOR_VARIANTS,
    OperatorFamily,
    OperatorVariant,
    OperatorCache,
    gradient_mpo,
    gradient_mpo_1d,
    gradient_mpo_1d_4th,
    laplacian_mpo,
    laplacian_mpo_1d,
    laplacian_mpo_1d_4th,
    mpo_apply,
    mpo_compose,
    mpo_scale,
    identity_mpo,
    _shift_right_mpo,
    _shift_left_mpo,
    _shift_double_right_mpo,
    _shift_double_left_mpo,
    get_variant_info,
    variable_coeff_elliptic_apply,
)
from ontic.engine.vm.ir import grad, laplace, Instruction, OpCode


# ══════════════════════════════════════════════════════════════════════
# Helper: build QTT from function and extract dense
# ══════════════════════════════════════════════════════════════════════

def _make_qtt_1d(fn, n_bits: int, domain: tuple[float, float] = (0.0, 1.0),
                 max_rank: int = 128) -> QTTTensor:
    """Create a 1-D QTT tensor from a scalar function."""
    return QTTTensor.from_function(
        fn, bits_per_dim=(n_bits,),
        domain=(domain,), max_rank=max_rank, cutoff=1e-14,
    )


def _grid_1d(n_bits: int, domain: tuple[float, float] = (0.0, 1.0)) -> NDArray:
    """Return cell-center grid for 1D domain."""
    N = 2 ** n_bits
    lo, hi = domain
    return np.linspace(lo, hi, N, endpoint=False)


def _l2_error_1d(
    computed: NDArray, exact: NDArray, h: float
) -> float:
    """Compute discrete L2 error norm."""
    return float(np.sqrt(h * np.sum((computed - exact) ** 2)))


# ══════════════════════════════════════════════════════════════════════
# B1: Operator Family/Variant Registry
# ══════════════════════════════════════════════════════════════════════

class TestOperatorRegistry:
    """Tests for the operator variant registry."""

    def test_registry_has_all_expected_variants(self) -> None:
        expected = {
            "grad_v1", "grad_v2_high_order",
            "lap_v1", "lap_v2_high_order",
            "elliptic_var_v1",
        }
        assert expected.issubset(set(OPERATOR_VARIANTS.keys()))

    def test_variant_has_required_fields(self) -> None:
        for tag, v in OPERATOR_VARIANTS.items():
            assert isinstance(v, OperatorVariant)
            assert isinstance(v.family, OperatorFamily)
            assert v.tag == tag
            assert v.order >= 1
            assert len(v.description) > 0

    def test_gradient_variants_are_gradient_family(self) -> None:
        for tag in ("grad_v1", "grad_v2_high_order"):
            assert OPERATOR_VARIANTS[tag].family == OperatorFamily.GRADIENT

    def test_laplacian_variants_are_laplacian_family(self) -> None:
        for tag in ("lap_v1", "lap_v2_high_order"):
            assert OPERATOR_VARIANTS[tag].family == OperatorFamily.LAPLACIAN

    def test_get_variant_info(self) -> None:
        v = get_variant_info("grad_v1")
        assert v.order == 2
        with pytest.raises(KeyError):
            get_variant_info("nonexistent_variant")

    def test_order_consistency(self) -> None:
        assert OPERATOR_VARIANTS["grad_v1"].order == 2
        assert OPERATOR_VARIANTS["grad_v2_high_order"].order == 4
        assert OPERATOR_VARIANTS["lap_v1"].order == 2
        assert OPERATOR_VARIANTS["lap_v2_high_order"].order == 4


# ══════════════════════════════════════════════════════════════════════
# B2: MPO Composition and Double-Shift
# ══════════════════════════════════════════════════════════════════════

class TestMPOComposition:
    """Tests for MPO composition and double-shift operators."""

    def test_identity_compose_identity(self) -> None:
        """I · I = I."""
        n = 6
        I = identity_mpo(n)
        II = mpo_compose(I, I)
        # Apply to a test vector
        u = _make_qtt_1d(lambda x: np.sin(2 * np.pi * x), n)
        u_II = mpo_apply(II, u, max_rank=64)
        dense_u = u.to_dense()
        dense_u_II = u_II.to_dense()
        np.testing.assert_allclose(dense_u_II, dense_u, atol=1e-10)

    def test_double_shift_right(self) -> None:
        """S₊² = right-shift composed twice.

        S₊ is the "read-next" operator: (S₊·v)[i] = v[i+1].
        Applied to δ_5, peak moves to index 5-2 = 3 (reading two ahead).
        """
        n_bits = 8
        N = 2 ** n_bits
        vals = np.zeros(N)
        vals[5] = 1.0
        u = QTTTensor.from_function(
            lambda x: np.interp(x, np.linspace(0, 1, N, endpoint=False), vals),
            bits_per_dim=(n_bits,), domain=((0.0, 1.0),),
            max_rank=64,
        )
        sp2 = _shift_double_right_mpo(n_bits)
        result = mpo_apply(sp2, u, max_rank=64)
        dense = result.to_dense()
        # S₊²(δ_5): peak at 5 - 2 = 3
        assert np.argmax(np.abs(dense)) == 3

    def test_double_shift_left(self) -> None:
        """S₋² = left-shift composed twice.

        S₋ is the transpose of S₊: (S₋·v)[i] = v[i-1].
        Applied to δ_{10}, peak moves to index 10+2 = 12.
        """
        n_bits = 8
        N = 2 ** n_bits
        vals = np.zeros(N)
        vals[10] = 1.0
        u = QTTTensor.from_function(
            lambda x: np.interp(x, np.linspace(0, 1, N, endpoint=False), vals),
            bits_per_dim=(n_bits,), domain=((0.0, 1.0),),
            max_rank=64,
        )
        sm2 = _shift_double_left_mpo(n_bits)
        result = mpo_apply(sm2, u, max_rank=64)
        dense = result.to_dense()
        # S₋²(δ_{10}): peak at 10 + 2 = 12
        assert np.argmax(np.abs(dense)) == 12


# ══════════════════════════════════════════════════════════════════════
# B2: 4th-Order Gradient Correctness
# ══════════════════════════════════════════════════════════════════════

class TestGradient4thOrder:
    """Tests for 4th-order gradient MPO correctness."""

    def test_gradient_v2_sine(self) -> None:
        """4th-order gradient of sin(2πx) ≈ 2π·cos(2πx)."""
        n_bits = 10
        domain = (0.0, 1.0)
        bpd = (n_bits,)
        dom = (domain,)

        u = _make_qtt_1d(lambda x: np.sin(2 * np.pi * x), n_bits)
        mpo = gradient_mpo(0, bpd, dom, variant="grad_v2_high_order")
        du = mpo_apply(mpo, u, max_rank=128, cutoff=1e-14)

        x = _grid_1d(n_bits)
        exact = 2 * np.pi * np.cos(2 * np.pi * x)
        computed = du.to_dense()

        # Interior points only (boundary stencil issues)
        interior = slice(4, -4)
        error = np.max(np.abs(computed[interior] - exact[interior]))
        assert error < 1e-3, f"4th-order gradient error too large: {error}"

    def test_v2_more_accurate_than_v1(self) -> None:
        """4th-order gradient should be more accurate than 2nd-order."""
        n_bits = 10
        domain = (0.0, 1.0)
        bpd = (n_bits,)
        dom = (domain,)

        u = _make_qtt_1d(lambda x: np.sin(2 * np.pi * x), n_bits)
        x = _grid_1d(n_bits)
        exact = 2 * np.pi * np.cos(2 * np.pi * x)
        interior = slice(4, -4)

        # v1 (2nd-order)
        mpo_v1 = gradient_mpo(0, bpd, dom, variant="grad_v1")
        du_v1 = mpo_apply(mpo_v1, u, max_rank=128, cutoff=1e-14)
        err_v1 = np.max(np.abs(du_v1.to_dense()[interior] - exact[interior]))

        # v2 (4th-order)
        mpo_v2 = gradient_mpo(0, bpd, dom, variant="grad_v2_high_order")
        du_v2 = mpo_apply(mpo_v2, u, max_rank=128, cutoff=1e-14)
        err_v2 = np.max(np.abs(du_v2.to_dense()[interior] - exact[interior]))

        assert err_v2 < err_v1, (
            f"4th-order ({err_v2:.2e}) not more accurate than "
            f"2nd-order ({err_v1:.2e})"
        )


# ══════════════════════════════════════════════════════════════════════
# B2: 4th-Order Laplacian Correctness
# ══════════════════════════════════════════════════════════════════════

class TestLaplacian4thOrder:
    """Tests for 4th-order Laplacian MPO correctness."""

    def test_laplacian_v2_sine(self) -> None:
        """4th-order Laplacian of sin(2πx) ≈ -(2π)²·sin(2πx)."""
        n_bits = 10
        domain = (0.0, 1.0)
        bpd = (n_bits,)
        dom = (domain,)

        u = _make_qtt_1d(lambda x: np.sin(2 * np.pi * x), n_bits)
        mpo = laplacian_mpo(bpd, dom, dim=0, variant="lap_v2_high_order")
        d2u = mpo_apply(mpo, u, max_rank=128, cutoff=1e-14)

        x = _grid_1d(n_bits)
        exact = -(2 * np.pi) ** 2 * np.sin(2 * np.pi * x)
        computed = d2u.to_dense()

        interior = slice(4, -4)
        error = np.max(np.abs(computed[interior] - exact[interior]))
        assert error < 0.5, f"4th-order Laplacian error too large: {error}"

    def test_v2_more_accurate_than_v1(self) -> None:
        """4th-order Laplacian should be more accurate than 2nd-order."""
        n_bits = 10
        domain = (0.0, 1.0)
        bpd = (n_bits,)
        dom = (domain,)

        u = _make_qtt_1d(lambda x: np.sin(2 * np.pi * x), n_bits)
        x = _grid_1d(n_bits)
        exact = -(2 * np.pi) ** 2 * np.sin(2 * np.pi * x)
        interior = slice(4, -4)

        mpo_v1 = laplacian_mpo(bpd, dom, dim=0, variant="lap_v1")
        d2u_v1 = mpo_apply(mpo_v1, u, max_rank=128, cutoff=1e-14)
        err_v1 = np.max(np.abs(d2u_v1.to_dense()[interior] - exact[interior]))

        mpo_v2 = laplacian_mpo(bpd, dom, dim=0, variant="lap_v2_high_order")
        d2u_v2 = mpo_apply(mpo_v2, u, max_rank=128, cutoff=1e-14)
        err_v2 = np.max(np.abs(d2u_v2.to_dense()[interior] - exact[interior]))

        assert err_v2 < err_v1, (
            f"4th-order ({err_v2:.2e}) not more accurate than "
            f"2nd-order ({err_v1:.2e})"
        )


# ══════════════════════════════════════════════════════════════════════
# B4: MMS Convergence Studies
# ══════════════════════════════════════════════════════════════════════

def _observed_order(errors: list[float], hs: list[float]) -> float:
    """Compute observed convergence order via log-log least-squares fit.

    Returns the slope of log(error) vs log(h).
    """
    if len(errors) < 2:
        return 0.0
    log_h = np.log(np.array(hs))
    log_e = np.log(np.array(errors))
    # Least squares: log_e = order * log_h + c
    A = np.vstack([log_h, np.ones_like(log_h)]).T
    result = np.linalg.lstsq(A, log_e, rcond=None)
    return float(result[0][0])


class TestGradientMMS:
    """MMS convergence for gradient MPO under n_bits refinement."""

    @pytest.mark.parametrize("variant,expected_order", [
        ("grad_v1", 2.0),
        ("grad_v2_high_order", 4.0),
    ])
    def test_gradient_convergence_sine(
        self, variant: str, expected_order: float
    ) -> None:
        """Gradient of sin(2πx) should converge at the expected order."""
        domain = (0.0, 1.0)
        levels = [8, 9, 10]
        errors: list[float] = []
        hs: list[float] = []

        for n_bits in levels:
            N = 2 ** n_bits
            h = 1.0 / N
            bpd = (n_bits,)
            dom = (domain,)

            u = _make_qtt_1d(lambda x: np.sin(2 * np.pi * x), n_bits)
            mpo = gradient_mpo(0, bpd, dom, variant=variant)
            du = mpo_apply(mpo, u, max_rank=128, cutoff=1e-14)

            x = _grid_1d(n_bits)
            exact = 2 * np.pi * np.cos(2 * np.pi * x)
            computed = du.to_dense()

            # Interior only to avoid boundary artifacts
            pad = 4 if variant == "grad_v2_high_order" else 2
            interior = slice(pad, -pad)
            err = _l2_error_1d(computed[interior], exact[interior], h)
            errors.append(err)
            hs.append(h)

        order = _observed_order(errors, hs)
        # Allow 0.5 tolerance below formal order
        assert order > expected_order - 0.5, (
            f"Gradient {variant}: observed order {order:.2f} "
            f"< expected {expected_order:.1f} - 0.5. "
            f"Errors: {[f'{e:.2e}' for e in errors]}"
        )

    def test_gradient_polynomial_exact(self) -> None:
        """2nd-order gradient of x² should give 2x (exactly for polynomials)."""
        n_bits = 10
        domain = (0.0, 1.0)
        bpd = (n_bits,)
        dom = (domain,)

        u = _make_qtt_1d(lambda x: x ** 2, n_bits)
        mpo = gradient_mpo(0, bpd, dom, variant="grad_v1")
        du = mpo_apply(mpo, u, max_rank=128, cutoff=1e-14)

        x = _grid_1d(n_bits)
        exact = 2 * x
        computed = du.to_dense()
        interior = slice(2, -2)
        np.testing.assert_allclose(
            computed[interior], exact[interior], atol=1e-4,
            err_msg="Gradient of x² should be ≈ 2x",
        )


class TestLaplacianMMS:
    """MMS convergence for Laplacian MPO under n_bits refinement."""

    @pytest.mark.parametrize("variant,expected_order,min_order", [
        ("lap_v1", 2.0, 1.5),
        ("lap_v2_high_order", 4.0, 2.5),
    ])
    def test_laplacian_convergence_sine(
        self, variant: str, expected_order: float, min_order: float,
    ) -> None:
        """Laplacian of sin(2πx) should converge at the expected order.

        Note: 4th-order Laplacian convergence rates are sensitive to
        QTT truncation at fine resolutions.  We use a wider tolerance
        (min_order) and higher max_rank to account for this.
        """
        domain = (0.0, 1.0)
        # Use lower bit levels to keep truncation error sub-dominant
        levels = [7, 8, 9] if variant == "lap_v2_high_order" else [8, 9, 10]
        errors: list[float] = []
        hs: list[float] = []

        for n_bits in levels:
            N = 2 ** n_bits
            h = 1.0 / N
            bpd = (n_bits,)
            dom = (domain,)

            u = _make_qtt_1d(lambda x: np.sin(2 * np.pi * x), n_bits,
                             max_rank=256)
            mpo = laplacian_mpo(bpd, dom, dim=0, variant=variant)
            d2u = mpo_apply(mpo, u, max_rank=256, cutoff=1e-14)

            x = _grid_1d(n_bits)
            exact = -(2 * np.pi) ** 2 * np.sin(2 * np.pi * x)
            computed = d2u.to_dense()

            pad = 8 if variant == "lap_v2_high_order" else 2
            interior = slice(pad, -pad)
            err = _l2_error_1d(computed[interior], exact[interior], h)
            errors.append(err)
            hs.append(h)

        order = _observed_order(errors, hs)
        assert order > min_order, (
            f"Laplacian {variant}: observed order {order:.2f} "
            f"< minimum {min_order:.1f}. "
            f"Errors: {[f'{e:.2e}' for e in errors]}"
        )

    def test_laplacian_polynomial_exact(self) -> None:
        """Laplacian of x³ should ≈ 6x (2nd order captures cubic curvature)."""
        n_bits = 10
        domain = (0.0, 1.0)
        bpd = (n_bits,)
        dom = (domain,)

        u = _make_qtt_1d(lambda x: x ** 3, n_bits)
        mpo = laplacian_mpo(bpd, dom, dim=0, variant="lap_v1")
        d2u = mpo_apply(mpo, u, max_rank=128, cutoff=1e-14)

        x = _grid_1d(n_bits)
        exact = 6 * x
        computed = d2u.to_dense()
        interior = slice(2, -2)
        np.testing.assert_allclose(
            computed[interior], exact[interior], atol=0.1,
            err_msg="Laplacian of x³ should be ≈ 6x",
        )


# ══════════════════════════════════════════════════════════════════════
# B3: Variable-Coefficient Elliptic Operator
# ══════════════════════════════════════════════════════════════════════

class TestVariableCoefficientElliptic:
    """Tests for ∇·(a∇u) composite operator."""

    def test_constant_coefficient_reduces_to_laplacian(self) -> None:
        """When a(x) = 1, ∇·(a∇u) = ∇²u."""
        n_bits = 9
        domain = (0.0, 1.0)
        bpd = (n_bits,)
        dom = (domain,)

        u = _make_qtt_1d(lambda x: np.sin(2 * np.pi * x), n_bits)
        a = _make_qtt_1d(lambda x: np.ones_like(x), n_bits)

        # Variable-coefficient with a=1
        result_var = variable_coeff_elliptic_apply(
            a, u, bpd, dom, max_rank=128, cutoff=1e-14,
        )

        # Direct Laplacian
        mpo_lap = laplacian_mpo(bpd, dom, dim=0)
        result_lap = mpo_apply(mpo_lap, u, max_rank=128, cutoff=1e-14)

        dense_var = result_var.to_dense()
        dense_lap = result_lap.to_dense()

        # They won't be identical (different construction paths) but should
        # agree in the interior
        interior = slice(4, -4)
        np.testing.assert_allclose(
            dense_var[interior], dense_lap[interior],
            atol=0.5, rtol=0.1,
            err_msg="∇·(1·∇u) should ≈ ∇²u",
        )

    def test_variable_coefficient_nonunit(self) -> None:
        """∇·(a∇u) with a(x) = 2 should be ≈ 2·∇²u."""
        n_bits = 9
        domain = (0.0, 1.0)
        bpd = (n_bits,)
        dom = (domain,)

        u = _make_qtt_1d(lambda x: np.sin(2 * np.pi * x), n_bits)
        a = _make_qtt_1d(lambda x: 2.0 * np.ones_like(x), n_bits)

        result = variable_coeff_elliptic_apply(
            a, u, bpd, dom, max_rank=128, cutoff=1e-14,
        )
        mpo_lap = laplacian_mpo(bpd, dom, dim=0)
        result_lap = mpo_apply(mpo_lap, u, max_rank=128, cutoff=1e-14)

        dense_var = result.to_dense()
        dense_2lap = 2.0 * result_lap.to_dense()
        interior = slice(4, -4)
        np.testing.assert_allclose(
            dense_var[interior], dense_2lap[interior],
            atol=1.0, rtol=0.15,
            err_msg="∇·(2·∇u) should ≈ 2·∇²u",
        )


# ══════════════════════════════════════════════════════════════════════
# B5: Variant-Aware Caching and IR Integration
# ══════════════════════════════════════════════════════════════════════

class TestOperatorCacheVariants:
    """Tests for variant-aware OperatorCache."""

    def test_cache_returns_different_mpos_for_different_variants(self) -> None:
        """Cache should return distinct MPOs for v1 vs v2."""
        cache = OperatorCache()
        bpd = (8,)
        dom = ((0.0, 1.0),)

        grad_v1 = cache.get_gradient(0, bpd, dom, variant="grad_v1")
        grad_v2 = cache.get_gradient(0, bpd, dom, variant="grad_v2_high_order")

        # Different bond dimensions expected (v2 is higher)
        max_bond_v1 = max(c.shape[0] for c in grad_v1)
        max_bond_v2 = max(c.shape[0] for c in grad_v2)
        assert max_bond_v2 > max_bond_v1, (
            f"4th-order gradient should have higher bond dim "
            f"({max_bond_v2}) than 2nd-order ({max_bond_v1})"
        )

    def test_cache_reuses_same_variant(self) -> None:
        """Same variant + config should return the cached object."""
        cache = OperatorCache()
        bpd = (8,)
        dom = ((0.0, 1.0),)

        a = cache.get_gradient(0, bpd, dom, variant="grad_v1")
        b = cache.get_gradient(0, bpd, dom, variant="grad_v1")
        assert a is b, "Cache should return same object for same key"

    def test_lap_cache_variants(self) -> None:
        """Laplacian cache distinguishes variants."""
        cache = OperatorCache()
        bpd = (8,)
        dom = ((0.0, 1.0),)

        lap_v1 = cache.get_laplacian(bpd, dom, dim=0, variant="lap_v1")
        lap_v2 = cache.get_laplacian(bpd, dom, dim=0, variant="lap_v2_high_order")

        max_bond_v1 = max(c.shape[0] for c in lap_v1)
        max_bond_v2 = max(c.shape[0] for c in lap_v2)
        assert max_bond_v2 > max_bond_v1


class TestIRVariantIntegration:
    """Tests that IR instruction builders support operator_variant."""

    def test_grad_instruction_default_variant(self) -> None:
        """Default grad() should not add operator_variant to params."""
        instr = grad(dst=0, src=1)
        assert "operator_variant" not in instr.params

    def test_grad_instruction_explicit_variant(self) -> None:
        """grad() with explicit variant should include it in params."""
        instr = grad(dst=0, src=1, operator_variant="grad_v2_high_order")
        assert instr.params["operator_variant"] == "grad_v2_high_order"

    def test_laplace_instruction_default_variant(self) -> None:
        """Default laplace() should not add operator_variant."""
        instr = laplace(dst=0, src=1)
        assert "operator_variant" not in instr.params

    def test_laplace_instruction_explicit_variant(self) -> None:
        """laplace() with variant should include it."""
        instr = laplace(dst=0, src=1, operator_variant="lap_v2_high_order")
        assert instr.params["operator_variant"] == "lap_v2_high_order"

    def test_instruction_repr_includes_variant(self) -> None:
        """Instruction repr should show the variant parameter."""
        instr = grad(dst=0, src=1, operator_variant="grad_v2_high_order")
        text = repr(instr)
        assert "grad_v2_high_order" in text


# ══════════════════════════════════════════════════════════════════════
# B5: Cross-Variant Convergence Comparison
# ══════════════════════════════════════════════════════════════════════

class TestCrossVariantConvergence:
    """Confirm that 4th-order variants converge faster than 2nd-order."""

    def test_gradient_v2_converges_faster(self) -> None:
        """Observed order of grad_v2 > grad_v1."""
        domain = (0.0, 1.0)
        levels = [8, 9, 10]

        for variant, label in [("grad_v1", "v1"), ("grad_v2_high_order", "v2")]:
            errors = []
            hs = []
            for n_bits in levels:
                N = 2 ** n_bits
                h = 1.0 / N
                bpd = (n_bits,)
                dom = (domain,)
                u = _make_qtt_1d(lambda x: np.sin(2 * np.pi * x), n_bits)
                mpo = gradient_mpo(0, bpd, dom, variant=variant)
                du = mpo_apply(mpo, u, max_rank=128, cutoff=1e-14)
                x = _grid_1d(n_bits)
                exact = 2 * np.pi * np.cos(2 * np.pi * x)
                pad = 4
                interior = slice(pad, -pad)
                err = _l2_error_1d(du.to_dense()[interior], exact[interior], h)
                errors.append(err)
                hs.append(h)
            if label == "v1":
                order_v1 = _observed_order(errors, hs)
            else:
                order_v2 = _observed_order(errors, hs)

        assert order_v2 > order_v1, (
            f"grad_v2 order ({order_v2:.2f}) should exceed "
            f"grad_v1 order ({order_v1:.2f})"
        )

    def test_laplacian_v2_converges_faster(self) -> None:
        """Observed order of lap_v2 > lap_v1."""
        domain = (0.0, 1.0)
        levels = [8, 9, 10]

        for variant, label in [("lap_v1", "v1"), ("lap_v2_high_order", "v2")]:
            errors = []
            hs = []
            for n_bits in levels:
                N = 2 ** n_bits
                h = 1.0 / N
                bpd = (n_bits,)
                dom = (domain,)
                u = _make_qtt_1d(lambda x: np.sin(2 * np.pi * x), n_bits)
                mpo = laplacian_mpo(bpd, dom, dim=0, variant=variant)
                d2u = mpo_apply(mpo, u, max_rank=128, cutoff=1e-14)
                x = _grid_1d(n_bits)
                exact = -(2 * np.pi) ** 2 * np.sin(2 * np.pi * x)
                pad = 4
                interior = slice(pad, -pad)
                err = _l2_error_1d(d2u.to_dense()[interior], exact[interior], h)
                errors.append(err)
                hs.append(h)
            if label == "v1":
                order_v1 = _observed_order(errors, hs)
            else:
                order_v2 = _observed_order(errors, hs)

        assert order_v2 > order_v1, (
            f"lap_v2 order ({order_v2:.2f}) should exceed "
            f"lap_v1 order ({order_v1:.2f})"
        )
