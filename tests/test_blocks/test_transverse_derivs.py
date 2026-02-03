"""
Unit tests for transverse (b) derivatives via Casimir recursion.

Tests the computation of h_{m,n} = ∂_a^m ∂_b^n G|_{a=1,b=0} using
the Casimir differential equation recursion (Eq. C.1).

Reference: arXiv:1203.6064, Appendix C
"""

import pytest
from mpmath import mp, mpf

from ising_bootstrap.blocks.transverse_derivs import (
    block_derivatives_full,
    block_derivatives_as_vector,
    compute_all_h_mn,
    casimir_eigenvalue,
    extract_index_set_h_values,
)
from ising_bootstrap.blocks.coordinate_transform import compute_h_m0_from_block_derivs
from ising_bootstrap.spectrum.index_set import generate_index_set, index_set_size
from ising_bootstrap.config import N_MAX, D, MPMATH_PRECISION


# Set precision for tests
mp.dps = MPMATH_PRECISION


class TestCasimirEigenvalue:
    """Tests for the quadratic Casimir eigenvalue."""

    def test_scalar_eigenvalue(self):
        """C_{Δ,0} = Δ(Δ-D) for scalars."""
        delta = mpf('1.5')
        C = casimir_eigenvalue(delta, 0)
        expected = delta * (delta - D)  # 1.5 × (1.5 - 3) = 1.5 × (-1.5) = -2.25
        assert abs(C - expected) < mpf('1e-30')

    def test_spin2_eigenvalue(self):
        """C_{Δ,l} = Δ(Δ-D) + l(l+D-2) for spin l."""
        delta = mpf('3.0')
        spin = 2
        C = casimir_eigenvalue(delta, spin)
        # 3×0 + 2×3 = 0 + 6 = 6
        expected = delta * (delta - D) + spin * (spin + D - 2)
        assert abs(C - expected) < mpf('1e-30')

    def test_stress_tensor_eigenvalue(self):
        """Stress tensor (Δ=D, l=2) has specific eigenvalue."""
        delta = mpf(D)  # Δ = D = 3
        spin = 2
        C = casimir_eigenvalue(delta, spin)
        # 3×0 + 2×3 = 6
        expected = mpf('6')
        assert abs(C - expected) < mpf('1e-30')


class TestBlockDerivativesFull:
    """Tests for the full block derivative computation."""

    def test_correct_number_of_derivatives(self):
        """Should return exactly index_set_size derivatives."""
        delta = mpf('1.5')
        spin = 0
        n_max = 5  # Smaller for faster tests
        h = block_derivatives_full(delta, spin, n_max=n_max)
        expected_size = index_set_size(n_max)
        assert len(h) == expected_size

    def test_all_derivatives_finite(self):
        """All h_{m,n} should be finite."""
        delta = mpf('1.41')
        spin = 0
        n_max = 5
        h = block_derivatives_full(delta, spin, n_max=n_max)
        for (m, n), value in h.items():
            assert mp.isfinite(value), f"h_{{{m},{n}}} is not finite"

    def test_diagonal_derivatives_present(self):
        """h_{m,0} values should be present (diagonal derivatives)."""
        delta = mpf('2.0')
        spin = 0
        n_max = 5
        h = block_derivatives_full(delta, spin, n_max=n_max)

        # Check odd m values for n=0
        for m in [1, 3, 5]:
            assert (m, 0) in h, f"h_{{{m},0}} missing"

    def test_transverse_derivatives_present(self):
        """h_{m,n} for n > 0 should be present."""
        delta = mpf('2.0')
        spin = 0
        n_max = 5
        h = block_derivatives_full(delta, spin, n_max=n_max)

        # Check some n > 0 values
        assert (1, 1) in h
        assert (1, 2) in h
        assert (3, 1) in h


class TestBlockDerivativesAsVector:
    """Tests for vector format output."""

    def test_correct_length(self):
        """Vector should have length equal to index_set_size."""
        delta = mpf('1.5')
        spin = 0
        n_max = 5
        vec = block_derivatives_as_vector(delta, spin, n_max=n_max)
        expected_size = index_set_size(n_max)
        assert len(vec) == expected_size

    def test_matches_dict_ordering(self):
        """Vector should match dict values in index set order."""
        delta = mpf('2.0')
        spin = 0
        n_max = 5

        h_dict = block_derivatives_full(delta, spin, n_max=n_max)
        h_vec = block_derivatives_as_vector(delta, spin, n_max=n_max)
        index_set = generate_index_set(n_max)

        for i, (m, n) in enumerate(index_set):
            assert abs(h_vec[i] - h_dict[(m, n)]) < mpf('1e-40')


class TestComputeAllHmn:
    """Tests for the full h_{m,n} computation."""

    def test_with_known_h_m0(self):
        """Test recursion with pre-computed h_{m,0} values."""
        delta = mpf('1.5')
        spin = 0
        n_max = 3

        # Get h_{m,0} from coordinate transform
        max_order = 2 * n_max + 1 + 4  # Buffer for recursion
        h_m0 = compute_h_m0_from_block_derivs(delta, spin, max_order)

        # Compute all h_{m,n}
        h_all = compute_all_h_mn(delta, spin, h_m0, n_max=n_max)

        # Should have h_{m,0} matching input
        for m in range(len(h_m0)):
            if (m, 0) in h_all:
                assert abs(h_all[(m, 0)] - h_m0[m]) < mpf('1e-40')


class TestDifferentSpins:
    """Tests for blocks with different spins."""

    def test_spin0_block(self):
        """Spin-0 block derivatives should be computable."""
        delta = mpf('1.41')  # Near 3D Ising Δ_ε
        h = block_derivatives_full(delta, 0, n_max=5)
        assert len(h) > 0
        assert all(mp.isfinite(v) for v in h.values())

    def test_spin2_block(self):
        """Spin-2 block derivatives should be computable."""
        delta = mpf('3.0')  # Unitarity bound for l=2
        h = block_derivatives_full(delta, 2, n_max=5)
        assert len(h) > 0
        assert all(mp.isfinite(v) for v in h.values())

    def test_spin4_block(self):
        """Spin-4 block derivatives should be computable."""
        delta = mpf('5.0')  # Unitarity bound for l=4
        h = block_derivatives_full(delta, 4, n_max=5)
        assert len(h) > 0
        assert all(mp.isfinite(v) for v in h.values())


class TestFullN_max10:
    """Tests with the full n_max=10 needed for the bootstrap."""

    @pytest.mark.slow
    def test_full_index_set_spin0(self):
        """Compute full 66 derivatives for spin-0."""
        delta = mpf('1.41')
        h = block_derivatives_full(delta, 0, n_max=N_MAX)
        assert len(h) == 66
        assert all(mp.isfinite(v) for v in h.values())

    @pytest.mark.slow
    def test_full_index_set_spin2(self):
        """Compute full 66 derivatives for spin-2."""
        delta = mpf('3.0')
        h = block_derivatives_full(delta, 2, n_max=N_MAX)
        assert len(h) == 66
        assert all(mp.isfinite(v) for v in h.values())
