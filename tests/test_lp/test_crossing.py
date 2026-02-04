"""
Tests for crossing function derivatives (prefactor table, identity, Leibniz rule).

Reference: arXiv:1203.6064, Sections 5-6
"""

import pytest
import numpy as np
from mpmath import mp, mpf, diff, power, factorial as mpfactorial

from ising_bootstrap.config import N_MAX, MAX_DERIV_ORDER, MPMATH_PRECISION
from ising_bootstrap.spectrum.index_set import generate_index_set
from ising_bootstrap.lp.crossing import (
    compute_prefactor_table,
    compute_identity_vector,
    compute_extended_h_array,
    compute_crossing_vector,
    generate_extended_pairs,
    extended_pair_count,
    build_comb_cache,
)


# ============================================================================
# Tests for extended index set
# ============================================================================

class TestExtendedPairs:
    """Tests for the extended index set (odd + even m)."""

    def test_count_n_max_10(self):
        pairs = generate_extended_pairs(10)
        assert len(pairs) == 132

    def test_count_function(self):
        assert extended_pair_count(10) == 132
        assert extended_pair_count(5) == 42

    def test_includes_odd_m(self):
        pairs = generate_extended_pairs(10)
        assert (1, 0) in pairs
        assert (3, 0) in pairs
        assert (21, 0) in pairs

    def test_includes_even_m(self):
        pairs = generate_extended_pairs(10)
        assert (0, 0) in pairs
        assert (2, 0) in pairs
        assert (20, 0) in pairs

    def test_all_satisfy_constraint(self):
        pairs = generate_extended_pairs(10)
        for p, q in pairs:
            assert p + 2 * q <= 21
            assert p >= 0
            assert q >= 0

    def test_contains_standard_index_set(self):
        """Standard index set (odd m) should be a subset of extended."""
        extended = set(generate_extended_pairs(10))
        standard = set(generate_index_set(10))
        assert standard.issubset(extended)

    def test_count_formula(self):
        """For n_max=N, count should be 2 * (n_max+1)^2 - (n_max+1) = ..."""
        # Actually the formula: sum over p=0..21 of floor((21-p)/2)+1
        # = 22 + 20 + 18 + ... + 2 (11 terms even) + 21 + 19 + ... + 1 (11 terms odd)
        # = 2 * (1+2+...+11) = 132
        for n in [2, 5, 10]:
            count = extended_pair_count(n)
            max_order = 2 * n + 1
            expected = sum((max_order - p) // 2 + 1 for p in range(max_order + 1))
            assert count == expected


# ============================================================================
# Tests for prefactor table
# ============================================================================

class TestPrefactorTable:
    """Tests for U^{j,k}(Δσ) = ∂_a^j ∂_b^k u^{Δσ} at (a=1, b=0)."""

    def test_shape_n_max_10(self):
        U = compute_prefactor_table(0.5, n_max=10)
        assert U.shape == (22, 11)

    def test_shape_n_max_2(self):
        U = compute_prefactor_table(0.5, n_max=2)
        assert U.shape == (6, 3)

    def test_zeroth_derivative(self):
        """U^{0,0} = u^{Δσ}(1,0) = (1/4)^{Δσ}."""
        for ds in [0.25, 0.5, 0.5182, 0.75, 1.0]:
            U = compute_prefactor_table(ds, n_max=2)
            expected = 0.25 ** ds
            assert abs(U[0, 0] - expected) < 1e-12, \
                f"ds={ds}: U[0,0]={U[0,0]}, expected={expected}"

    def test_first_a_derivative(self):
        """U^{1,0} = d/da u^{Δσ} = Δσ u^{Δσ-1} (a/2) at a=1, b=0."""
        for ds in [0.5, 0.5182, 1.0]:
            U = compute_prefactor_table(ds, n_max=2)
            expected = ds * 0.25 ** (ds - 1) * 0.5
            assert abs(U[1, 0] - expected) < 1e-10, \
                f"ds={ds}: U[1,0]={U[1,0]}, expected={expected}"

    def test_second_a_derivative(self):
        """U^{2,0} = d²/da² u^{Δσ} at a=1, b=0."""
        # u = (a²-b)/4. At b=0: u = a²/4.
        # u^α = (a²/4)^α = a^{2α}/4^α
        # d²/da²(a^{2α}) = 2α(2α-1) a^{2α-2}
        # At a=1: 2α(2α-1)/4^α
        for ds in [0.5, 0.5182, 1.0]:
            U = compute_prefactor_table(ds, n_max=2)
            expected = 2 * ds * (2 * ds - 1) * 0.25 ** ds
            assert abs(U[2, 0] - expected) < 1e-10, \
                f"ds={ds}: U[2,0]={U[2,0]}, expected={expected}"

    def test_first_b_derivative(self):
        """U^{0,1} = d/db u^{Δσ} = Δσ u^{Δσ-1} (-1/4) at a=1, b=0."""
        for ds in [0.5, 0.5182]:
            U = compute_prefactor_table(ds, n_max=2)
            expected = ds * 0.25 ** (ds - 1) * (-0.25)
            assert abs(U[0, 1] - expected) < 1e-10, \
                f"ds={ds}: U[0,1]={U[0,1]}, expected={expected}"

    def test_numerical_cross_check(self):
        """Cross-check against mpmath numerical differentiation."""
        mp.dps = 30
        ds = mpf('0.5182')

        def u_power(a, b):
            return power((a ** 2 - b) / 4, ds)

        U = compute_prefactor_table(0.5182, n_max=3)

        # Check (j=1, k=0)
        numerical_10 = float(diff(lambda a: u_power(a, mpf(0)), mpf(1), 1))
        assert abs(U[1, 0] - numerical_10) / max(abs(numerical_10), 1e-15) < 1e-8

        # Check (j=0, k=1)
        numerical_01 = float(diff(lambda b: u_power(mpf(1), b), mpf(0), 1))
        assert abs(U[0, 1] - numerical_01) / max(abs(numerical_01), 1e-15) < 1e-8

        # Check (j=2, k=0)
        numerical_20 = float(diff(lambda a: u_power(a, mpf(0)), mpf(1), 2))
        assert abs(U[2, 0] - numerical_20) / max(abs(numerical_20), 1e-15) < 1e-8

    def test_special_case_ds_half(self):
        """For Δσ=0.5, u^{0.5} = |a|/2 * sqrt(1 - b/a²) at b=0.

        At b=0: u^{0.5} = a/2 (for a > 0). So:
        - d^m/da^m(a/2) at a=1 is: 1/2 for m=0, 1/2 for m=1, 0 for m≥2
        Wait, d/da(a/2) = 1/2, d²/da² = 0. But U[1,0] = 0.5 and U[2,0] should be 0.
        """
        U = compute_prefactor_table(0.5, n_max=5)
        # At b=0, u^{0.5} = ((1+eps)^2)^{0.5}/2 = (1+eps)/2 for eps near 0
        # So derivatives in a: d^0 = 0.5, d^1 = 0.5, d^m = 0 for m >= 2
        assert abs(U[0, 0] - 0.5) < 1e-14
        assert abs(U[1, 0] - 0.5) < 1e-14
        assert abs(U[2, 0]) < 1e-14
        assert abs(U[3, 0]) < 1e-14

    def test_all_finite(self):
        """All computed entries should be finite."""
        U = compute_prefactor_table(0.5182, n_max=10)
        for j in range(22):
            for k in range(11):
                if j + 2 * k <= 21:
                    assert np.isfinite(U[j, k]), f"U[{j},{k}] is not finite"


# ============================================================================
# Tests for identity derivatives
# ============================================================================

class TestIdentityDerivatives:
    """Tests for F_id^{m,n} = -2 U^{m,n}."""

    def test_shape(self):
        f_id = compute_identity_vector(0.5)
        assert f_id.shape == (66,)

    def test_shape_n_max_2(self):
        f_id = compute_identity_vector(0.5, n_max=2)
        assert f_id.shape == (6,)

    def test_f_id_10_ds_half(self):
        """F_id^{1,0} = ∂_a(v^{0.5} - u^{0.5}) at (1,0).

        At b=0: v^{0.5} - u^{0.5} = (2-a)/2 - a/2 = 1 - a.
        So ∂_a(1-a) = -1.
        """
        f_id = compute_identity_vector(0.5, n_max=2)
        assert abs(f_id[0] - (-1.0)) < 1e-14

    def test_f_id_30_ds_half(self):
        """F_id^{3,0} at Δσ=0.5 should be 0 (1-a is linear)."""
        f_id = compute_identity_vector(0.5, n_max=2)
        index_set = generate_index_set(2)
        idx_30 = index_set.index((3, 0))
        assert abs(f_id[idx_30]) < 1e-14

    def test_f_id_is_minus_2_U(self):
        """F_id^{m,n} = -2 U^{m,n} for odd m."""
        ds = 0.5182
        U = compute_prefactor_table(ds, n_max=10)
        f_id = compute_identity_vector(ds, n_max=10)
        index_set = generate_index_set(10)

        for idx, (m, n) in enumerate(index_set):
            expected = -2.0 * U[m, n]
            assert abs(f_id[idx] - expected) < 1e-14 * max(abs(expected), 1.0)

    def test_antisymmetry_check(self):
        """F_id should be antisymmetric under u↔v, so F_id^{m,n} = 0 for even m.

        Since our index set only has odd m, all entries should be nonzero
        (in general). But for Δσ=0.5, b=0 contributions vanish for m≥2.
        """
        f_id = compute_identity_vector(0.5182, n_max=2)
        # At least the first entry should be nonzero
        assert abs(f_id[0]) > 1e-10

    def test_numerical_cross_check(self):
        """Cross-check identity derivatives against mpmath numerical diff."""
        mp.dps = 30
        ds = mpf('0.5182')

        def f_id_func(a, b):
            u = (a ** 2 - b) / 4
            v = ((2 - a) ** 2 - b) / 4
            return power(v, ds) - power(u, ds)

        f_id = compute_identity_vector(0.5182, n_max=3)
        index_set = generate_index_set(3)

        # Check (1, 0): d/da at b=0
        numerical = float(diff(lambda a: f_id_func(a, mpf(0)), mpf(1), 1))
        idx = index_set.index((1, 0))
        assert abs(f_id[idx] - numerical) / max(abs(numerical), 1e-15) < 1e-7

        # Check (3, 0): d³/da³ at b=0
        numerical = float(diff(lambda a: f_id_func(a, mpf(0)), mpf(1), 3))
        idx = index_set.index((3, 0))
        assert abs(f_id[idx] - numerical) < 1e-5

    def test_all_finite(self):
        """All identity derivatives should be finite."""
        f_id = compute_identity_vector(0.5182)
        assert np.all(np.isfinite(f_id))


# ============================================================================
# Tests for extended block derivatives
# ============================================================================

class TestExtendedBlockDerivatives:
    """Tests for computing h_{m,n} including even m."""

    def test_shape(self):
        H = compute_extended_h_array(1.5, 0, n_max=2)
        max_order = 2 * 2 + 1
        max_k = max_order // 2
        assert H.shape == (max_order + 1, max_k + 1)

    def test_all_finite(self):
        H = compute_extended_h_array(1.5, 0, n_max=2)
        for p in range(H.shape[0]):
            for q in range(H.shape[1]):
                if p + 2 * q <= 5:
                    assert np.isfinite(H[p, q]), f"H[{p},{q}] is not finite"

    def test_odd_m_matches_blocks_module(self):
        """Odd-m values should match the blocks module output."""
        from ising_bootstrap.blocks import block_derivatives_full
        delta, spin = mpf('1.5'), 0

        h_dict = block_derivatives_full(delta, spin, n_max=2)
        H = compute_extended_h_array(1.5, 0, n_max=2)

        index_set = generate_index_set(2)
        for m, n in index_set:
            expected = float(h_dict[(m, n)])
            actual = H[m, n]
            rel_err = abs(actual - expected) / max(abs(expected), 1e-30)
            assert rel_err < 1e-10, \
                f"h[{m},{n}]: actual={actual}, expected={expected}, rel_err={rel_err}"

    def test_spin2(self):
        """Test extended derivatives for a spin-2 operator."""
        H = compute_extended_h_array(3.0, 2, n_max=2)
        assert H.shape == (6, 3)
        assert np.all(np.isfinite(H))

    @pytest.mark.slow
    def test_full_n_max_10(self):
        """Test full n_max=10 extended derivatives."""
        H = compute_extended_h_array(1.41, 0, n_max=10)
        assert H.shape == (22, 11)
        for p in range(22):
            for q in range(11):
                if p + 2 * q <= 21:
                    assert np.isfinite(H[p, q]), f"H[{p},{q}] not finite"


# ============================================================================
# Tests for crossing function derivatives
# ============================================================================

class TestCrossingDerivatives:
    """Tests for F^{m,n}_{Δ,l}(Δσ) via the Leibniz rule."""

    def test_shape(self):
        H = compute_extended_h_array(1.5, 0, n_max=2)
        U = compute_prefactor_table(0.5, n_max=2)
        F = compute_crossing_vector(H, U, n_max=2)
        assert F.shape == (6,)

    def test_all_finite(self):
        H = compute_extended_h_array(1.5, 0, n_max=2)
        U = compute_prefactor_table(0.5182, n_max=2)
        F = compute_crossing_vector(H, U, n_max=2)
        assert np.all(np.isfinite(F))

    def test_identity_block(self):
        """For the identity block (G=1), crossing derivs should equal F_id.

        Identity: h_{0,0}=1, h_{m,n}=0 for m+n > 0.
        So F^{m,n} = 2 C(m,m) C(n,n) (-1)^m U^{m,n} h_{0,0}
                   = 2(-1)^m U^{m,n} = -2 U^{m,n} (m odd)
        """
        ds = 0.5182
        n_max = 3

        # Build identity block's extended H: h_{0,0}=1, all others 0
        max_order = 2 * n_max + 1
        max_k = max_order // 2
        H_id = np.zeros((max_order + 1, max_k + 1))
        H_id[0, 0] = 1.0

        U = compute_prefactor_table(ds, n_max=n_max)
        F = compute_crossing_vector(H_id, U, n_max=n_max)
        f_id = compute_identity_vector(ds, n_max=n_max)

        np.testing.assert_allclose(F, f_id, rtol=1e-12)

    def test_antisymmetry(self):
        """F^{Δσ}_{Δ,l} is antisymmetric under u↔v, so only odd m survive.

        Since our index set only has odd m, this is automatically satisfied.
        But let's verify that the Leibniz formula with odd m gives nonzero values.
        """
        H = compute_extended_h_array(1.5, 0, n_max=2)
        U = compute_prefactor_table(0.5, n_max=2)
        F = compute_crossing_vector(H, U, n_max=2)
        # Should have at least some nonzero values
        assert np.max(np.abs(F)) > 1e-10

    def test_spin2_operator(self):
        """Test crossing derivatives for a spin-2 operator."""
        H = compute_extended_h_array(3.0, 2, n_max=2)
        U = compute_prefactor_table(0.5182, n_max=2)
        F = compute_crossing_vector(H, U, n_max=2)
        assert F.shape == (6,)
        assert np.all(np.isfinite(F))

    def test_different_delta_sigma(self):
        """Crossing derivatives should change with Δσ."""
        H = compute_extended_h_array(1.5, 0, n_max=2)
        U1 = compute_prefactor_table(0.50, n_max=2)
        U2 = compute_prefactor_table(0.52, n_max=2)
        F1 = compute_crossing_vector(H, U1, n_max=2)
        F2 = compute_crossing_vector(H, U2, n_max=2)
        # Should differ
        assert np.max(np.abs(F1 - F2)) > 1e-10


# ============================================================================
# Tests for comb cache
# ============================================================================

class TestCombCache:
    """Tests for precomputed binomial coefficient cache."""

    def test_comb_cache_contains_needed_values(self):
        from math import comb
        cache = build_comb_cache(10)
        index_set = generate_index_set(10)
        for m, n in index_set:
            for j in range(m + 1):
                assert (m, j) in cache
                assert cache[(m, j)] == comb(m, j)
            for k in range(n + 1):
                assert (n, k) in cache
                assert cache[(n, k)] == comb(n, k)
