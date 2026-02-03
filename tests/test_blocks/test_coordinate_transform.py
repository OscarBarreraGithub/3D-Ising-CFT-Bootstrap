"""
Unit tests for coordinate transformation between (z, z̄) and (a, b).

Tests the transformation:
    z = (a + √b)/2,  z̄ = (a - √b)/2

At the crossing-symmetric point z = z̄ = 1/2, we have a = 1, b = 0.

Reference: arXiv:1203.6064, Section 4 and Eq. 4.15
"""

import pytest
from mpmath import mp, mpf

from ising_bootstrap.blocks.coordinate_transform import (
    z_zbar_to_a_b,
    a_b_to_z_zbar,
    z_zbar_to_u_v,
    z_derivatives_to_h_m0,
    h_m0_to_z_derivatives,
    diagonal_a_derivative_to_z_derivative,
    compute_h_m0_from_block_derivs,
    crossing_point_values,
)
from ising_bootstrap.config import Z_POINT, MPMATH_PRECISION


# Set precision for tests
mp.dps = MPMATH_PRECISION


class TestZZbarToAB:
    """Tests for (z, z̄) → (a, b) transformation."""

    def test_crossing_point(self):
        """At z = z̄ = 1/2, should get a = 1, b = 0."""
        z = zbar = mpf('0.5')
        a, b = z_zbar_to_a_b(z, zbar)
        assert abs(a - mpf('1')) < mpf('1e-30')
        assert abs(b) < mpf('1e-30')

    def test_diagonal(self):
        """When z = z̄, should get b = 0."""
        for z_val in [0.3, 0.4, 0.6, 0.7]:
            z = zbar = mpf(z_val)
            a, b = z_zbar_to_a_b(z, zbar)
            assert abs(b) < mpf('1e-30'), f"b not zero for z = {z_val}"
            assert abs(a - 2 * z) < mpf('1e-30'), f"a ≠ 2z for z = {z_val}"

    def test_off_diagonal(self):
        """When z ≠ z̄, should get b > 0."""
        z = mpf('0.6')
        zbar = mpf('0.4')
        a, b = z_zbar_to_a_b(z, zbar)
        assert b > 0
        assert abs(a - (z + zbar)) < mpf('1e-30')
        assert abs(b - (z - zbar)**2) < mpf('1e-30')


class TestABToZZbar:
    """Tests for (a, b) → (z, z̄) transformation."""

    def test_crossing_point(self):
        """At a = 1, b = 0, should get z = z̄ = 1/2."""
        a = mpf('1')
        b = mpf('0')
        z, zbar = a_b_to_z_zbar(a, b)
        assert abs(z - mpf('0.5')) < mpf('1e-30')
        assert abs(zbar - mpf('0.5')) < mpf('1e-30')

    def test_b_zero_diagonal(self):
        """When b = 0, should get z = z̄ = a/2."""
        for a_val in [0.5, 1.0, 1.5]:
            a = mpf(a_val)
            b = mpf('0')
            z, zbar = a_b_to_z_zbar(a, b)
            expected = a / 2
            assert abs(z - expected) < mpf('1e-30')
            assert abs(zbar - expected) < mpf('1e-30')

    def test_positive_b(self):
        """When b > 0, z > z̄."""
        a = mpf('1')
        b = mpf('0.04')  # √b = 0.2
        z, zbar = a_b_to_z_zbar(a, b)
        assert z > zbar


class TestRoundTrip:
    """Tests for round-trip consistency."""

    def test_z_to_a_to_z(self):
        """(z, z̄) → (a, b) → (z, z̄) should be identity."""
        test_points = [
            (mpf('0.5'), mpf('0.5')),
            (mpf('0.6'), mpf('0.4')),
            (mpf('0.3'), mpf('0.3')),
            (mpf('0.7'), mpf('0.2')),
        ]
        for z_orig, zbar_orig in test_points:
            a, b = z_zbar_to_a_b(z_orig, zbar_orig)
            z_back, zbar_back = a_b_to_z_zbar(a, b)
            assert abs(z_back - z_orig) < mpf('1e-30')
            assert abs(zbar_back - zbar_orig) < mpf('1e-30')

    def test_a_to_z_to_a(self):
        """(a, b) → (z, z̄) → (a, b) should be identity."""
        test_points = [
            (mpf('1'), mpf('0')),
            (mpf('1'), mpf('0.01')),
            (mpf('0.8'), mpf('0')),
            (mpf('1.2'), mpf('0.04')),
        ]
        for a_orig, b_orig in test_points:
            z, zbar = a_b_to_z_zbar(a_orig, b_orig)
            a_back, b_back = z_zbar_to_a_b(z, zbar)
            assert abs(a_back - a_orig) < mpf('1e-30')
            assert abs(b_back - b_orig) < mpf('1e-30')


class TestZZbarToUV:
    """Tests for (z, z̄) → (u, v) cross-ratio conversion."""

    def test_crossing_point(self):
        """At z = z̄ = 1/2, u = v = 1/4."""
        z = zbar = mpf('0.5')
        u, v = z_zbar_to_u_v(z, zbar)
        assert abs(u - mpf('0.25')) < mpf('1e-30')
        assert abs(v - mpf('0.25')) < mpf('1e-30')

    def test_formulas(self):
        """u = z z̄ and v = (1-z)(1-z̄)."""
        z = mpf('0.6')
        zbar = mpf('0.4')
        u, v = z_zbar_to_u_v(z, zbar)
        assert abs(u - z * zbar) < mpf('1e-30')
        assert abs(v - (1 - z) * (1 - zbar)) < mpf('1e-30')


class TestDerivativeConversion:
    """Tests for converting between z-derivatives and a-derivatives."""

    def test_diagonal_factor(self):
        """Factor should be (1/2)^m."""
        for m in range(10):
            factor = diagonal_a_derivative_to_z_derivative(m)
            expected = mpf('0.5') ** m
            assert abs(factor - expected) < mpf('1e-30')

    def test_z_derivs_to_h_m0(self):
        """h_{m,0} = (1/2)^m × d_z^m G."""
        z_derivs = [mpf('1'), mpf('2'), mpf('4'), mpf('8')]
        h_m0 = z_derivatives_to_h_m0(z_derivs)

        # h_{0,0} = 1 × 1 = 1
        # h_{1,0} = 0.5 × 2 = 1
        # h_{2,0} = 0.25 × 4 = 1
        # h_{3,0} = 0.125 × 8 = 1
        for h in h_m0:
            assert abs(h - mpf('1')) < mpf('1e-30')

    def test_h_m0_to_z_derivs(self):
        """d_z^m G = 2^m × h_{m,0}."""
        h_m0 = [mpf('1'), mpf('1'), mpf('1'), mpf('1')]
        z_derivs = h_m0_to_z_derivatives(h_m0)

        # d_z^0 G = 1 × 1 = 1
        # d_z^1 G = 2 × 1 = 2
        # d_z^2 G = 4 × 1 = 4
        # d_z^3 G = 8 × 1 = 8
        expected = [mpf('1'), mpf('2'), mpf('4'), mpf('8')]
        for z_d, exp in zip(z_derivs, expected):
            assert abs(z_d - exp) < mpf('1e-30')

    def test_round_trip_conversions(self):
        """z → h → z should be identity."""
        z_derivs = [mpf(2**i) for i in range(5)]
        h_m0 = z_derivatives_to_h_m0(z_derivs)
        z_back = h_m0_to_z_derivatives(h_m0)

        for orig, back in zip(z_derivs, z_back):
            assert abs(orig - back) < mpf('1e-30')


class TestCrossingPointValues:
    """Tests for crossing_point_values function."""

    def test_all_keys_present(self):
        """Should return all coordinate values."""
        values = crossing_point_values()
        required_keys = ['z', 'zbar', 'a', 'b', 'u', 'v']
        for key in required_keys:
            assert key in values

    def test_values_correct(self):
        """Values should match known crossing point."""
        values = crossing_point_values()
        assert abs(values['z'] - mpf('0.5')) < mpf('1e-30')
        assert abs(values['zbar'] - mpf('0.5')) < mpf('1e-30')
        assert abs(values['a'] - mpf('1')) < mpf('1e-30')
        assert abs(values['b']) < mpf('1e-30')
        assert abs(values['u'] - mpf('0.25')) < mpf('1e-30')
        assert abs(values['v'] - mpf('0.25')) < mpf('1e-30')


class TestComputeHm0FromBlockDerivs:
    """Tests for computing h_{m,0} from block derivatives."""

    def test_correct_length(self):
        """Should return max_m + 1 values."""
        delta = mpf('1.5')
        spin = 0
        max_m = 10
        h_m0 = compute_h_m0_from_block_derivs(delta, spin, max_m)
        assert len(h_m0) == max_m + 1

    def test_all_finite(self):
        """All h_{m,0} should be finite."""
        delta = mpf('1.41')
        spin = 0
        max_m = 15
        h_m0 = compute_h_m0_from_block_derivs(delta, spin, max_m)
        for i, h in enumerate(h_m0):
            assert mp.isfinite(h), f"h_{{{i},0}} is not finite"

    def test_different_spins(self):
        """Should work for different spins."""
        for spin in [0, 1, 2, 4]:
            delta = mpf(max(1.5, spin + 1.0))
            h_m0 = compute_h_m0_from_block_derivs(delta, spin, max_m=10)
            assert len(h_m0) == 11
            assert all(mp.isfinite(h) for h in h_m0)
