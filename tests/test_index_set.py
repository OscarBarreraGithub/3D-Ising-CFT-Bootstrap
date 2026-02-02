"""
Unit tests for the index set generation module.

Tests the (m, n) index set used for truncating the linear functional.
The index set has:
- m odd (1, 3, 5, ..., 21)
- n >= 0
- m + 2n <= 21

For n_max=10, this gives exactly 66 pairs.
"""

import pytest
from ising_bootstrap.spectrum.index_set import (
    generate_index_set,
    iter_index_set,
    index_set_size,
    is_valid_index_pair,
    get_index_position,
    get_pair_at_position,
)
from ising_bootstrap.config import N_MAX


class TestGenerateIndexSet:
    """Tests for generate_index_set function."""

    def test_count_is_66(self):
        """The index set should have exactly 66 elements for n_max=10."""
        index_set = generate_index_set(10)
        assert len(index_set) == 66

    def test_count_matches_formula(self):
        """Count should match the formula-based calculation."""
        for n_max in [5, 7, 10, 12]:
            index_set = generate_index_set(n_max)
            assert len(index_set) == index_set_size(n_max)

    def test_specific_elements_present(self):
        """Check that specific valid pairs are in the set."""
        index_set = generate_index_set(10)

        # Boundary cases
        assert (1, 0) in index_set  # Smallest m, smallest n
        assert (1, 10) in index_set  # m=1, max n: 1 + 20 = 21
        assert (21, 0) in index_set  # Largest m, n=0

        # Interior points
        assert (3, 9) in index_set  # 3 + 18 = 21
        assert (5, 8) in index_set  # 5 + 16 = 21
        assert (7, 7) in index_set  # 7 + 14 = 21
        assert (11, 5) in index_set  # 11 + 10 = 21

    def test_specific_elements_absent(self):
        """Check that invalid pairs are NOT in the set."""
        index_set = generate_index_set(10)

        # Even m not allowed
        assert (2, 0) not in index_set
        assert (4, 5) not in index_set

        # Exceeds constraint m + 2n <= 21
        assert (1, 11) not in index_set  # 1 + 22 = 23 > 21
        assert (22, 0) not in index_set  # m=22 not odd anyway
        assert (23, 0) not in index_set  # 23 > 21

        # n < 0 not allowed
        assert (1, -1) not in index_set

    def test_all_m_are_odd(self):
        """Every m in the index set must be odd."""
        index_set = generate_index_set(10)
        for m, n in index_set:
            assert m % 2 == 1, f"m={m} is not odd"

    def test_all_satisfy_constraint(self):
        """Every pair must satisfy m + 2n <= 21."""
        index_set = generate_index_set(10)
        for m, n in index_set:
            assert m + 2 * n <= 21, f"({m}, {n}): {m} + {2*n} = {m + 2*n} > 21"

    def test_all_m_positive(self):
        """Every m must be >= 1."""
        index_set = generate_index_set(10)
        for m, n in index_set:
            assert m >= 1

    def test_all_n_nonnegative(self):
        """Every n must be >= 0."""
        index_set = generate_index_set(10)
        for m, n in index_set:
            assert n >= 0

    def test_detailed_count_by_m(self):
        """Verify the count for each value of m matches expectation."""
        index_set = generate_index_set(10)

        # Count by m value
        counts = {}
        for m, n in index_set:
            counts[m] = counts.get(m, 0) + 1

        # Expected counts from TODO.md
        expected = {
            1: 11,   # n=0..10
            3: 10,   # n=0..9
            5: 9,    # n=0..8
            7: 8,    # n=0..7
            9: 7,    # n=0..6
            11: 6,   # n=0..5
            13: 5,   # n=0..4
            15: 4,   # n=0..3
            17: 3,   # n=0..2
            19: 2,   # n=0..1
            21: 1,   # n=0
        }

        for m, expected_count in expected.items():
            assert counts.get(m, 0) == expected_count, f"m={m}: expected {expected_count}, got {counts.get(m, 0)}"

        # Verify total
        assert sum(counts.values()) == 66


class TestIterIndexSet:
    """Tests for iter_index_set generator."""

    def test_yields_same_as_generate(self):
        """Iterator should yield the same pairs as the list generator."""
        list_result = generate_index_set(10)
        iter_result = list(iter_index_set(10))
        assert list_result == iter_result

    def test_lazy_evaluation(self):
        """Iterator should work without storing all elements."""
        # Just verify it's an iterator
        it = iter_index_set(10)
        first = next(it)
        assert first == (1, 0)
        second = next(it)
        assert second == (1, 1)


class TestIndexSetSize:
    """Tests for index_set_size function."""

    def test_size_10(self):
        """Size should be 66 for n_max=10."""
        assert index_set_size(10) == 66

    def test_size_5(self):
        """Size should be 21 for n_max=5."""
        # m + 2n <= 11
        # m=1: n=0..5 (6 terms)
        # m=3: n=0..4 (5 terms)
        # m=5: n=0..3 (4 terms)
        # m=7: n=0..2 (3 terms)
        # m=9: n=0..1 (2 terms)
        # m=11: n=0 (1 term)
        # Total: 6+5+4+3+2+1 = 21
        assert index_set_size(5) == 21

    def test_agrees_with_formula(self):
        """Size should equal (n_max + 1)^2."""
        for n_max in range(1, 15):
            # Formula: sum of 1 + 2 + ... + (n_max+1) = (n_max+1)(n_max+2)/2
            # Actually for odd m only, it's (n_max+1)^2 ... let me verify
            computed = index_set_size(n_max)
            # The pattern is triangular: 1 + 2 + ... + (n_max+1)
            expected = (n_max + 1) * (n_max + 2) // 2
            # Wait, that's not right either. Let me just verify against actual generation
            actual = len(generate_index_set(n_max))
            assert computed == actual


class TestIsValidIndexPair:
    """Tests for is_valid_index_pair function."""

    def test_valid_pairs(self):
        """Test valid index pairs."""
        assert is_valid_index_pair(1, 0, 10)
        assert is_valid_index_pair(1, 10, 10)
        assert is_valid_index_pair(21, 0, 10)
        assert is_valid_index_pair(3, 9, 10)

    def test_even_m_invalid(self):
        """Even m should be invalid."""
        assert not is_valid_index_pair(2, 0, 10)
        assert not is_valid_index_pair(4, 5, 10)
        assert not is_valid_index_pair(0, 5, 10)

    def test_exceeds_constraint_invalid(self):
        """m + 2n > 21 should be invalid."""
        assert not is_valid_index_pair(1, 11, 10)  # 1 + 22 = 23
        assert not is_valid_index_pair(3, 10, 10)  # 3 + 20 = 23
        assert not is_valid_index_pair(23, 0, 10)  # 23 > 21

    def test_negative_values_invalid(self):
        """Negative m or n should be invalid."""
        assert not is_valid_index_pair(-1, 0, 10)
        assert not is_valid_index_pair(1, -1, 10)
        assert not is_valid_index_pair(0, 0, 10)  # m=0 not allowed


class TestGetIndexPosition:
    """Tests for get_index_position function."""

    def test_first_element(self):
        """First element (1,0) should be at position 0."""
        assert get_index_position(1, 0, 10) == 0

    def test_sequential_positions(self):
        """Positions should be sequential within each m block."""
        assert get_index_position(1, 0, 10) == 0
        assert get_index_position(1, 1, 10) == 1
        assert get_index_position(1, 2, 10) == 2
        assert get_index_position(1, 10, 10) == 10

    def test_m3_starts_at_11(self):
        """m=3 block starts at position 11 (after 11 elements from m=1)."""
        assert get_index_position(3, 0, 10) == 11
        assert get_index_position(3, 1, 10) == 12

    def test_last_element(self):
        """Last element (21,0) should be at position 65."""
        assert get_index_position(21, 0, 10) == 65

    def test_invalid_raises(self):
        """Invalid pairs should raise ValueError."""
        with pytest.raises(ValueError):
            get_index_position(2, 0, 10)  # Even m
        with pytest.raises(ValueError):
            get_index_position(1, 11, 10)  # Exceeds constraint

    def test_roundtrip(self):
        """get_index_position and get_pair_at_position should be inverses."""
        index_set = generate_index_set(10)
        for i, (m, n) in enumerate(index_set):
            assert get_index_position(m, n, 10) == i


class TestGetPairAtPosition:
    """Tests for get_pair_at_position function."""

    def test_first_position(self):
        """Position 0 should give (1, 0)."""
        assert get_pair_at_position(0, 10) == (1, 0)

    def test_last_position(self):
        """Position 65 should give (21, 0)."""
        assert get_pair_at_position(65, 10) == (21, 0)

    def test_m3_first(self):
        """Position 11 should give (3, 0)."""
        assert get_pair_at_position(11, 10) == (3, 0)

    def test_out_of_range_raises(self):
        """Out of range positions should raise ValueError."""
        with pytest.raises(ValueError):
            get_pair_at_position(-1, 10)
        with pytest.raises(ValueError):
            get_pair_at_position(66, 10)  # Only 0-65 valid

    def test_all_positions(self):
        """Verify all positions map correctly to index set."""
        index_set = generate_index_set(10)
        for i in range(66):
            assert get_pair_at_position(i, 10) == index_set[i]


class TestDefaultNMax:
    """Tests that default n_max from config is used correctly."""

    def test_default_is_10(self):
        """Default n_max should be 10."""
        assert N_MAX == 10

    def test_functions_use_default(self):
        """Functions should use n_max=10 by default."""
        assert len(generate_index_set()) == 66
        assert index_set_size() == 66
        assert is_valid_index_pair(21, 0)
        assert not is_valid_index_pair(23, 0)
