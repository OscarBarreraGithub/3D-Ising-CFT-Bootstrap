"""
Unit tests for the spectrum discretization module.

Tests Table 2 implementation (T1-T5) and unitarity bounds.
Reference: arXiv:1203.6064, Appendix D
"""

import pytest
import numpy as np
from ising_bootstrap.spectrum.discretization import (
    SpectrumPoint,
    generate_table_points,
    generate_full_spectrum,
    spectrum_to_array,
    count_by_spin,
    count_by_table,
    get_scalars,
    get_spinning,
    build_spectrum_with_gaps,
    estimate_spectrum_size,
    get_table_info,
)
from ising_bootstrap.spectrum.unitarity import (
    unitarity_bound,
    satisfies_unitarity,
    check_unitarity,
    is_allowed_spin,
    validate_operator,
)
from ising_bootstrap.config import (
    TABLE_1, TABLE_2, TABLE_3, TABLE_4, TABLE_5,
    FULL_DISCRETIZATION, REDUCED_DISCRETIZATION,
)


class TestUnitarityBounds:
    """Tests for unitarity bound functions."""

    def test_scalar_bound(self):
        """Scalar (l=0) unitarity bound should be 0.5 in D=3."""
        assert unitarity_bound(0) == 0.5

    def test_spin2_bound(self):
        """Spin-2 unitarity bound should be 3.0 in D=3."""
        assert unitarity_bound(2) == 3.0

    def test_spin4_bound(self):
        """Spin-4 unitarity bound should be 5.0 in D=3."""
        assert unitarity_bound(4) == 5.0

    def test_general_spinning_bound(self):
        """Spinning operator bound should be l+1 for D=3."""
        for l in range(1, 20):
            assert unitarity_bound(l) == l + 1

    def test_negative_spin_raises(self):
        """Negative spin should raise ValueError."""
        with pytest.raises(ValueError):
            unitarity_bound(-1)


class TestSatisfiesUnitarity:
    """Tests for satisfies_unitarity function."""

    def test_at_bound(self):
        """Operator at unitarity bound should satisfy it."""
        assert satisfies_unitarity(0.5, 0)
        assert satisfies_unitarity(3.0, 2)

    def test_above_bound(self):
        """Operator above unitarity bound should satisfy it."""
        assert satisfies_unitarity(1.0, 0)
        assert satisfies_unitarity(5.0, 2)

    def test_below_bound(self):
        """Operator below unitarity bound should not satisfy it."""
        assert not satisfies_unitarity(0.4, 0)
        assert not satisfies_unitarity(2.9, 2)

    def test_strict_at_bound(self):
        """With strict=True, operator exactly at bound should fail."""
        assert not satisfies_unitarity(0.5, 0, strict=True)
        assert satisfies_unitarity(0.5001, 0, strict=True)


class TestCheckUnitarity:
    """Tests for check_unitarity with tolerance."""

    def test_slightly_below_with_tolerance(self):
        """Slightly below bound should pass with tolerance."""
        assert check_unitarity(0.5 - 1e-12, 0, tolerance=1e-10)

    def test_far_below_fails(self):
        """Far below bound should fail even with tolerance."""
        assert not check_unitarity(0.4, 0, tolerance=1e-10)


class TestIsAllowedSpin:
    """Tests for is_allowed_spin function."""

    def test_even_spins_allowed(self):
        """Even spins should be allowed."""
        for l in [0, 2, 4, 6, 8, 10]:
            assert is_allowed_spin(l), f"Spin {l} should be allowed"

    def test_odd_spins_not_allowed(self):
        """Odd spins should not be allowed (Bose symmetry)."""
        for l in [1, 3, 5, 7, 9]:
            assert not is_allowed_spin(l), f"Spin {l} should not be allowed"

    def test_negative_spin_not_allowed(self):
        """Negative spins should not be allowed."""
        assert not is_allowed_spin(-1)
        assert not is_allowed_spin(-2)


class TestValidateOperator:
    """Tests for validate_operator function."""

    def test_valid_operator(self):
        """Valid operators should pass."""
        assert validate_operator(1.5, 0)  # Scalar above bound
        assert validate_operator(3.0, 2)  # Spin-2 at bound

    def test_odd_spin_fails(self):
        """Odd spin operators should fail."""
        assert not validate_operator(2.0, 1)
        assert not validate_operator(5.0, 3)

    def test_below_unitarity_fails(self):
        """Operators below unitarity bound should fail."""
        assert not validate_operator(0.4, 0)
        assert not validate_operator(2.5, 2)


class TestTable2Parameters:
    """Tests that Table 2 parameters match the paper exactly."""

    def test_t1_parameters(self):
        """T1: High-resolution scalars."""
        assert TABLE_1.name == "T1"
        assert TABLE_1.delta == 2e-5
        assert TABLE_1.delta_max == 3
        assert TABLE_1.l_max == 0

    def test_t2_parameters(self):
        """T2: Low-spin detail."""
        assert TABLE_2.name == "T2"
        assert TABLE_2.delta == 5e-4
        assert TABLE_2.delta_max == 8
        assert TABLE_2.l_max == 6

    def test_t3_parameters(self):
        """T3: Mid-range coverage."""
        assert TABLE_3.name == "T3"
        assert TABLE_3.delta == 2e-3
        assert TABLE_3.delta_max == 22
        assert TABLE_3.l_max == 20

    def test_t4_parameters(self):
        """T4: Intermediate asymptotics."""
        assert TABLE_4.name == "T4"
        assert TABLE_4.delta == 0.02
        assert TABLE_4.delta_max == 100
        assert TABLE_4.l_max == 50

    def test_t5_parameters(self):
        """T5: Far asymptotics."""
        assert TABLE_5.name == "T5"
        assert TABLE_5.delta == 1.0
        assert TABLE_5.delta_max == 500
        assert TABLE_5.l_max == 100

    def test_full_discretization_has_all_tables(self):
        """FULL_DISCRETIZATION should contain all 5 tables."""
        assert len(FULL_DISCRETIZATION) == 5
        assert TABLE_1 in FULL_DISCRETIZATION
        assert TABLE_5 in FULL_DISCRETIZATION

    def test_reduced_discretization(self):
        """REDUCED_DISCRETIZATION should contain T1 and T2 only."""
        assert len(REDUCED_DISCRETIZATION) == 2
        assert TABLE_1 in REDUCED_DISCRETIZATION
        assert TABLE_2 in REDUCED_DISCRETIZATION
        assert TABLE_3 not in REDUCED_DISCRETIZATION


class TestGenerateTablePoints:
    """Tests for generate_table_points function."""

    def test_t1_only_scalars(self):
        """T1 should only have scalars (l=0)."""
        points = generate_table_points(TABLE_1)
        spins = set(p.spin for p in points)
        assert spins == {0}

    def test_t1_scalar_count(self):
        """T1 should have correct number of scalars."""
        # Δ from 0.5 to 3.0 in steps of 2e-5
        # n_points = (3.0 - 0.5) / 2e-5 + 1 = 125001
        points = generate_table_points(TABLE_1)
        expected = int(round((3.0 - 0.5) / 2e-5)) + 1
        assert len(points) == expected

    def test_t2_spin_range(self):
        """T2 should have spins 0, 2, 4, 6."""
        points = generate_table_points(TABLE_2)
        spins = sorted(set(p.spin for p in points))
        assert spins == [0, 2, 4, 6]

    def test_t3_spin_range(self):
        """T3 should have spins 0, 2, 4, ..., 20."""
        points = generate_table_points(TABLE_3)
        spins = sorted(set(p.spin for p in points))
        assert spins == list(range(0, 21, 2))

    def test_all_points_satisfy_unitarity(self):
        """All generated points should satisfy unitarity bounds."""
        for table in FULL_DISCRETIZATION:
            points = generate_table_points(table)
            for p in points:
                assert check_unitarity(p.delta, p.spin), \
                    f"Point ({p.delta}, {p.spin}) from {p.table} violates unitarity"

    def test_only_even_spins(self):
        """All points should have even spin."""
        for table in FULL_DISCRETIZATION:
            points = generate_table_points(table)
            for p in points:
                assert p.spin % 2 == 0, f"Odd spin {p.spin} found in {table.name}"

    def test_dimension_upper_limit(self):
        """Check that Δ_max shifts correctly with spin."""
        # For T2: Δ_max(l) = 8 + 2*(6 - l)
        points = generate_table_points(TABLE_2)

        for p in points:
            expected_max = TABLE_2.delta_max + 2 * (TABLE_2.l_max - p.spin)
            # Allow small tolerance for floating point
            assert p.delta <= expected_max + 1e-8, \
                f"Point ({p.delta}, {p.spin}) exceeds max {expected_max}"

    def test_gap_constraint_scalars(self):
        """Gap constraint should exclude low-dimension scalars."""
        gap = 1.5
        points = generate_table_points(TABLE_1, gap_scalar=gap)

        scalars = [p for p in points if p.spin == 0]
        for p in scalars:
            assert p.delta >= gap - 1e-8, \
                f"Scalar at Δ={p.delta} should be excluded by gap={gap}"


class TestGenerateFullSpectrum:
    """Tests for generate_full_spectrum function."""

    def test_uses_all_tables_by_default(self):
        """Default should use all 5 tables."""
        points = generate_full_spectrum()
        tables_used = set(p.table for p in points)
        assert "T1" in tables_used
        assert "T5" in tables_used

    def test_reduced_discretization(self):
        """Reduced discretization should use T1-T2 only."""
        points = generate_full_spectrum(tables=REDUCED_DISCRETIZATION)
        tables_used = set(p.table for p in points)
        assert tables_used <= {"T1", "T2"}

    def test_sorted_by_spin_then_delta(self):
        """Points should be sorted by (spin, delta)."""
        points = generate_full_spectrum(tables=REDUCED_DISCRETIZATION)

        prev_spin, prev_delta = -1, -1
        for p in points:
            if p.spin == prev_spin:
                assert p.delta >= prev_delta - 1e-10
            else:
                assert p.spin > prev_spin
            prev_spin, prev_delta = p.spin, p.delta

    def test_all_satisfy_unitarity(self):
        """All points in full spectrum should satisfy unitarity."""
        points = generate_full_spectrum(tables=REDUCED_DISCRETIZATION)
        for p in points:
            assert check_unitarity(p.delta, p.spin)

    def test_no_exact_duplicates(self):
        """Should not have exact duplicate (Δ, l) pairs."""
        points = generate_full_spectrum(tables=REDUCED_DISCRETIZATION)
        seen = set()
        for p in points:
            key = (round(p.delta, 10), p.spin)
            assert key not in seen, f"Duplicate found: {key}"
            seen.add(key)


class TestSpectrumHelpers:
    """Tests for spectrum helper functions."""

    def test_spectrum_to_array(self):
        """spectrum_to_array should return correct shape."""
        points = generate_table_points(TABLE_1)
        arr = spectrum_to_array(points)

        assert arr.shape == (len(points), 2)
        assert arr[0, 0] == points[0].delta
        assert arr[0, 1] == points[0].spin

    def test_count_by_spin(self):
        """count_by_spin should count correctly."""
        points = generate_table_points(TABLE_2)
        counts = count_by_spin(points)

        # T2 has spins 0, 2, 4, 6
        assert set(counts.keys()) == {0, 2, 4, 6}
        assert sum(counts.values()) == len(points)

    def test_count_by_table(self):
        """count_by_table should count correctly."""
        points = generate_full_spectrum(tables=REDUCED_DISCRETIZATION)
        counts = count_by_table(points)

        assert "T1" in counts
        assert "T2" in counts
        assert sum(counts.values()) == len(points)

    def test_get_scalars(self):
        """get_scalars should return only l=0 points."""
        points = generate_full_spectrum(tables=REDUCED_DISCRETIZATION)
        scalars = get_scalars(points)

        for p in scalars:
            assert p.spin == 0

    def test_get_spinning(self):
        """get_spinning should return only l>0 points."""
        points = generate_full_spectrum(tables=REDUCED_DISCRETIZATION)
        spinning = get_spinning(points)

        for p in spinning:
            assert p.spin > 0


class TestBuildSpectrumWithGaps:
    """Tests for build_spectrum_with_gaps convenience function."""

    def test_no_gaps(self):
        """Without gaps, should include all scalars."""
        points = build_spectrum_with_gaps(tables=REDUCED_DISCRETIZATION)
        scalars = get_scalars(points)

        # Should include scalars from unitarity bound (0.5)
        min_delta = min(p.delta for p in scalars)
        assert abs(min_delta - 0.5) < 1e-6

    def test_stage_a_gap(self):
        """Stage A: gap below epsilon should exclude low scalars."""
        delta_eps = 1.4
        points = build_spectrum_with_gaps(
            delta_epsilon=delta_eps,
            tables=REDUCED_DISCRETIZATION
        )
        scalars = get_scalars(points)

        for p in scalars:
            assert p.delta >= delta_eps - 1e-8, \
                f"Scalar at Δ={p.delta} should be excluded"

    def test_stage_b_two_gaps(self):
        """Stage B: two gaps should exclude scalars between ε and ε'."""
        delta_eps = 1.4
        delta_eprime = 3.5
        points = build_spectrum_with_gaps(
            delta_epsilon=delta_eps,
            delta_epsilon_prime=delta_eprime,
            tables=REDUCED_DISCRETIZATION
        )
        scalars = get_scalars(points)

        for p in scalars:
            # Either at/above ε' or there shouldn't be any below ε
            assert p.delta >= delta_eprime - 1e-8, \
                f"Scalar at Δ={p.delta} in gap [{delta_eps}, {delta_eprime})"

    def test_spinning_unaffected_by_gaps(self):
        """Gap constraints should not affect spinning operators."""
        points_no_gap = build_spectrum_with_gaps(
            tables=REDUCED_DISCRETIZATION
        )
        points_with_gap = build_spectrum_with_gaps(
            delta_epsilon=1.4,
            tables=REDUCED_DISCRETIZATION
        )

        spinning_no_gap = sorted([(p.delta, p.spin) for p in get_spinning(points_no_gap)])
        spinning_with_gap = sorted([(p.delta, p.spin) for p in get_spinning(points_with_gap)])

        assert spinning_no_gap == spinning_with_gap


class TestSpectrumPoint:
    """Tests for SpectrumPoint dataclass."""

    def test_as_tuple(self):
        """as_tuple should return (Δ, l)."""
        p = SpectrumPoint(delta=1.5, spin=2, table="T2")
        assert p.as_tuple() == (1.5, 2)

    def test_attributes(self):
        """Attributes should be accessible."""
        p = SpectrumPoint(delta=1.5, spin=2, table="T2")
        assert p.delta == 1.5
        assert p.spin == 2
        assert p.table == "T2"


class TestGetTableInfo:
    """Tests for get_table_info function."""

    def test_t1_info(self):
        """T1 info should be correct."""
        info = get_table_info(TABLE_1)
        assert info['name'] == "T1"
        assert info['delta'] == 2e-5
        assert info['l_max'] == 0
        assert info['spins'] == [0]

    def test_t2_info(self):
        """T2 info should be correct."""
        info = get_table_info(TABLE_2)
        assert info['name'] == "T2"
        assert info['spins'] == [0, 2, 4, 6]


class TestEstimateSpectrumSize:
    """Tests for estimate_spectrum_size function."""

    def test_estimate_matches_actual_for_reduced(self):
        """Estimate should be upper bound on actual size."""
        estimate = estimate_spectrum_size(tables=REDUCED_DISCRETIZATION)
        actual = len(generate_full_spectrum(tables=REDUCED_DISCRETIZATION, remove_duplicates=False))
        assert estimate == actual  # Without duplicate removal, should match

    def test_estimate_reasonable(self):
        """Estimate for full discretization should be reasonable."""
        estimate = estimate_spectrum_size()
        # Should be large but finite
        assert estimate > 100000
        assert estimate < 10000000
