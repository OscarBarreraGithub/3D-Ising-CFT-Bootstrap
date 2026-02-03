"""
Unit tests for spin recursion of conformal blocks.

Tests the computation of G_{Δ,l}(z) for l ≥ 2 using the spin
recursion relation (Eq. 4.9 from arXiv:1203.6064).

At z=z̄, the derivative term vanishes, giving a pure algebraic recursion.
"""

import pytest
from mpmath import mp, mpf

from ising_bootstrap.blocks.spin_recursion import (
    higher_spin_block,
    diagonal_block_any_spin,
    _spin_recursion_lhs_coeff,
    _spin_recursion_first_term_coeff,
    _spin_recursion_second_term_coeff,
    _spin_recursion_third_term_coeff,
)
from ising_bootstrap.blocks.diagonal_blocks import spin0_block, spin1_block
from ising_bootstrap.config import Z_POINT, MPMATH_PRECISION, D


# Set precision for tests
mp.dps = MPMATH_PRECISION


class TestHigherSpinBlock:
    """Tests for the spin recursion computation."""

    def test_spin0_uses_base_formula(self):
        """Spin-0 should use the direct ₃F₂ formula."""
        delta = mpf('1.5')
        result = higher_spin_block(delta, 0)
        expected = spin0_block(delta)
        assert abs(result - expected) < mpf('1e-40')

    def test_spin1_uses_base_formula(self):
        """Spin-1 should use the direct ₃F₂ formula."""
        delta = mpf('2.0')
        result = higher_spin_block(delta, 1)
        expected = spin1_block(delta)
        assert abs(result - expected) < mpf('1e-40')

    def test_spin2_finite(self):
        """Spin-2 block should be finite."""
        delta = mpf('3.0')  # Unitarity bound for l=2 is Δ ≥ l + D - 2 = 3
        result = higher_spin_block(delta, 2)
        assert mp.isfinite(result)

    def test_spin4_finite(self):
        """Spin-4 block should be finite."""
        delta = mpf('5.0')  # Unitarity bound for l=4 is Δ ≥ 5
        result = higher_spin_block(delta, 4)
        assert mp.isfinite(result)

    def test_higher_spin_positive(self):
        """Higher spin blocks should be positive for physical Δ."""
        for spin in [2, 4, 6]:
            # Take Δ = l + D - 2 + 0.1 (slightly above unitarity bound)
            delta = mpf(spin + D - 2 + 0.1)
            result = higher_spin_block(delta, spin)
            assert result > 0, f"Block negative for l={spin}, Δ={delta}"

    def test_negative_spin_raises(self):
        """Negative spin should raise ValueError."""
        with pytest.raises(ValueError):
            higher_spin_block(mpf('2.0'), -1)


class TestSpinRecursionCoefficients:
    """Tests for the spin recursion coefficient functions."""

    def test_lhs_coeff_d3(self):
        """Test LHS coefficient (l+D-3)(2Δ+2-D) for D=3."""
        # For D=3: (l)(2Δ - 1)
        delta = mpf('2.0')
        spin = 2
        result = _spin_recursion_lhs_coeff(delta, spin)
        expected = 2 * (2 * 2.0 - 1)  # l × (2Δ - 1) = 2 × 3 = 6
        assert abs(result - expected) < mpf('1e-30')

    def test_first_term_coeff_d3(self):
        """Test first term coefficient (D-2)(Δ+l-1) for D=3."""
        # For D=3: (Δ + l - 1)
        delta = mpf('3.0')
        spin = 2
        result = _spin_recursion_first_term_coeff(delta, spin)
        expected = 3.0 + 2 - 1  # 4
        assert abs(result - expected) < mpf('1e-30')


class TestDiagonalBlockAnySpin:
    """Tests for the unified diagonal_block_any_spin function."""

    def test_matches_higher_spin_block(self):
        """Should match higher_spin_block for all spins."""
        for spin in [0, 1, 2, 3, 4]:
            delta = mpf(max(0.5, spin + 1.0))  # Above unitarity bound
            result = diagonal_block_any_spin(delta, spin)
            expected = higher_spin_block(delta, spin)
            assert abs(result - expected) < mpf('1e-40')


class TestCacheConsistency:
    """Tests for caching behavior in spin recursion."""

    def test_cache_reuse(self):
        """Cache should be reused across calls with same delta."""
        cache = {}
        delta = mpf('3.0')

        # First call populates cache
        G2_first = higher_spin_block(delta, 2, cache=cache)

        # Second call should use cache
        G2_second = higher_spin_block(delta, 2, cache=cache)

        assert abs(G2_first - G2_second) < mpf('1e-45')
        # Check cache is populated
        assert len(cache) > 0


class TestStressTensorBlock:
    """Test the stress tensor (l=2, Δ=3) which is special."""

    def test_stress_tensor_block(self):
        """Stress tensor has Δ=D=3 for l=2 in D=3."""
        delta = mpf('3.0')
        spin = 2
        result = higher_spin_block(delta, spin)
        assert mp.isfinite(result)
        assert result > 0
