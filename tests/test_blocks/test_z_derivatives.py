"""
Unit tests for z-derivatives of conformal blocks.

Tests the computation of d^m/dz^m G_{Δ,l}(z) using the ₃F₂ representation
and chain rule (Faà di Bruno's formula with Bell polynomials).

Reference: arXiv:1203.6064, Section 4
"""

import pytest
from mpmath import mp, mpf, diff

from ising_bootstrap.blocks.z_derivatives import (
    block_z_derivatives,
    spin0_block_z_derivatives,
    spin1_block_z_derivatives,
    higher_spin_block_z_derivatives,
)
from ising_bootstrap.blocks.diagonal_blocks import spin0_block, spin1_block
from ising_bootstrap.blocks.spin_recursion import higher_spin_block
from ising_bootstrap.config import Z_POINT, MPMATH_PRECISION


# Set precision for tests
mp.dps = MPMATH_PRECISION


class TestSpin0ZDerivatives:
    """Tests for z-derivatives of spin-0 blocks."""

    def test_zeroth_derivative_matches_block(self):
        """0th derivative should match the block value."""
        delta = mpf('1.5')
        z = mpf(Z_POINT)
        derivs = spin0_block_z_derivatives(delta, z, max_order=0)
        expected = spin0_block(delta, z)
        assert abs(derivs[0] - expected) < mpf('1e-30')

    def test_derivatives_finite(self):
        """All derivatives should be finite."""
        delta = mpf('1.41')
        z = mpf(Z_POINT)
        derivs = spin0_block_z_derivatives(delta, z, max_order=10)
        for i, d in enumerate(derivs):
            assert mp.isfinite(d), f"Derivative {i} is not finite"

    def test_first_derivative_numerical(self):
        """Compare first derivative to numerical differentiation."""
        delta = mpf('2.0')
        z = mpf(Z_POINT)

        # Analytical derivative
        derivs = spin0_block_z_derivatives(delta, z, max_order=1)
        analytical = derivs[1]

        # Numerical derivative
        numerical = diff(lambda x: spin0_block(delta, x), z)

        # Should match to reasonable precision
        rel_error = abs((analytical - numerical) / numerical) if numerical != 0 else abs(analytical)
        assert rel_error < mpf('1e-10'), f"Relative error {rel_error} too large"


class TestSpin1ZDerivatives:
    """Tests for z-derivatives of spin-1 blocks."""

    def test_zeroth_derivative_matches_block(self):
        """0th derivative should match the block value."""
        delta = mpf('2.0')
        z = mpf(Z_POINT)
        derivs = spin1_block_z_derivatives(delta, z, max_order=0)
        expected = spin1_block(delta, z)
        assert abs(derivs[0] - expected) < mpf('1e-30')

    def test_derivatives_finite(self):
        """All derivatives should be finite."""
        delta = mpf('2.5')
        z = mpf(Z_POINT)
        derivs = spin1_block_z_derivatives(delta, z, max_order=10)
        for i, d in enumerate(derivs):
            assert mp.isfinite(d), f"Derivative {i} is not finite"


class TestHigherSpinZDerivatives:
    """Tests for z-derivatives of higher spin blocks."""

    def test_spin2_zeroth_derivative_matches(self):
        """0th derivative of spin-2 should match block value."""
        delta = mpf('3.0')
        z = mpf(Z_POINT)
        derivs = higher_spin_block_z_derivatives(delta, 2, z, max_order=0)
        expected = higher_spin_block(delta, 2, z)
        assert abs(derivs[0] - expected) < mpf('1e-25')

    def test_spin4_derivatives_finite(self):
        """Spin-4 derivatives should be finite."""
        delta = mpf('5.0')
        z = mpf(Z_POINT)
        derivs = higher_spin_block_z_derivatives(delta, 4, z, max_order=5)
        for i, d in enumerate(derivs):
            assert mp.isfinite(d), f"Derivative {i} is not finite"


class TestBlockZDerivatives:
    """Tests for the unified block_z_derivatives function."""

    def test_spin0_routing(self):
        """block_z_derivatives with l=0 should use spin0 formula."""
        delta = mpf('1.5')
        z = mpf(Z_POINT)
        result = block_z_derivatives(delta, 0, z, max_order=3)
        expected = spin0_block_z_derivatives(delta, z, max_order=3)
        for i in range(4):
            assert abs(result[i] - expected[i]) < mpf('1e-35')

    def test_spin1_routing(self):
        """block_z_derivatives with l=1 should use spin1 formula."""
        delta = mpf('2.0')
        z = mpf(Z_POINT)
        result = block_z_derivatives(delta, 1, z, max_order=3)
        expected = spin1_block_z_derivatives(delta, z, max_order=3)
        for i in range(4):
            assert abs(result[i] - expected[i]) < mpf('1e-35')

    def test_max_derivative_order(self):
        """Should compute up to max_order derivatives."""
        delta = mpf('1.5')
        z = mpf(Z_POINT)
        for max_order in [5, 10, 15]:
            derivs = block_z_derivatives(delta, 0, z, max_order=max_order)
            assert len(derivs) == max_order + 1


class TestDerivativeConsistency:
    """Cross-checks for derivative consistency."""

    def test_higher_order_numerical_check(self):
        """Spot check higher derivatives against numerical diff."""
        delta = mpf('1.5')
        z = mpf(Z_POINT)

        # Get analytical second derivative
        derivs = spin0_block_z_derivatives(delta, z, max_order=2)
        analytical_d2 = derivs[2]

        # Numerical second derivative
        numerical_d2 = diff(lambda x: spin0_block(delta, x), z, n=2)

        rel_error = abs((analytical_d2 - numerical_d2) / numerical_d2) if numerical_d2 != 0 else abs(analytical_d2)
        assert rel_error < mpf('1e-8'), f"Relative error {rel_error} too large for d²G/dz²"

    def test_derivatives_needed_for_bootstrap(self):
        """Should compute all derivatives needed for n_max=10 (order 21+buffer)."""
        delta = mpf('1.41')
        z = mpf(Z_POINT)
        # Need order up to 21 + buffer for recursion
        max_order = 25
        derivs = block_z_derivatives(delta, 0, z, max_order=max_order)
        assert len(derivs) == max_order + 1
        for d in derivs:
            assert mp.isfinite(d)
