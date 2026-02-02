"""
Tests for configuration module.
"""

import pytest
from ising_bootstrap import config


class TestConstants:
    """Test physical constants and parameters."""

    def test_dimension(self):
        """Verify D=3."""
        assert config.D == 3

    def test_n_max(self):
        """Verify n_max=10."""
        assert config.N_MAX == 10

    def test_alpha(self):
        """Verify α = D/2 - 1 = 0.5 for D=3."""
        assert config.ALPHA == 0.5

    def test_max_deriv_order(self):
        """Verify max derivative order = 2*n_max + 1 = 21."""
        assert config.MAX_DERIV_ORDER == 21


class TestCrossingPoint:
    """Test crossing-symmetric point values."""

    def test_z_point(self):
        """Verify z = 1/2."""
        assert config.Z_POINT == 0.5

    def test_a_point(self):
        """Verify a = 1 at crossing-symmetric point."""
        assert config.A_POINT == 1.0

    def test_b_point(self):
        """Verify b = 0 at crossing-symmetric point."""
        assert config.B_POINT == 0.0

    def test_u_v_values(self):
        """Verify u = v = 1/4 at crossing-symmetric point."""
        assert config.U_POINT == 0.25
        assert config.V_POINT == 0.25


class TestUnitarityBounds:
    """Test unitarity bound function."""

    def test_scalar_bound(self):
        """Scalars (l=0) have Δ ≥ 1/2 in D=3."""
        assert config.unitarity_bound(0) == 0.5

    def test_spin_1_bound(self):
        """Spin-1 operators have Δ ≥ 2 in D=3."""
        assert config.unitarity_bound(1) == 2

    def test_spin_2_bound(self):
        """Spin-2 operators have Δ ≥ 3 in D=3."""
        assert config.unitarity_bound(2) == 3

    def test_general_spin_bound(self):
        """General formula: Δ ≥ l + 1 for l ≥ 1."""
        for spin in range(1, 20):
            assert config.unitarity_bound(spin) == spin + 1

    def test_negative_spin_raises(self):
        """Negative spin should raise ValueError."""
        with pytest.raises(ValueError):
            config.unitarity_bound(-1)


class TestIndexSetCount:
    """Test index set counting function."""

    def test_count_n_max_10(self):
        """
        For n_max=10, the index set should have 66 elements.

        Index set: m odd, m ≥ 1, n ≥ 0, m + 2n ≤ 21

        m=1:  n=0,1,...,10  → 11 terms
        m=3:  n=0,1,...,9   → 10 terms
        m=5:  n=0,1,...,8   → 9 terms
        ...
        m=21: n=0           → 1 term

        Total: 11+10+9+8+7+6+5+4+3+2+1 = 66
        """
        assert config.get_index_set_count(10) == 66

    def test_count_n_max_5(self):
        """
        For n_max=5, m + 2n ≤ 11.

        m=1:  n=0,1,...,5  → 6 terms
        m=3:  n=0,1,...,4  → 5 terms
        m=5:  n=0,1,...,3  → 4 terms
        m=7:  n=0,1,2      → 3 terms
        m=9:  n=0,1        → 2 terms
        m=11: n=0          → 1 term

        Total: 6+5+4+3+2+1 = 21
        """
        assert config.get_index_set_count(5) == 21

    def test_count_n_max_1(self):
        """
        For n_max=1, m + 2n ≤ 3.

        m=1: n=0,1 → 2 terms
        m=3: n=0   → 1 term

        Total: 3
        """
        assert config.get_index_set_count(1) == 3


class TestDiscretizationTables:
    """Test Table 2 discretization parameters."""

    def test_table_1(self):
        """T1: δ=2×10⁻⁵, Δ_max=3, L_max=0."""
        assert config.TABLE_1.delta == 2e-5
        assert config.TABLE_1.delta_max == 3
        assert config.TABLE_1.l_max == 0

    def test_table_2(self):
        """T2: δ=5×10⁻⁴, Δ_max=8, L_max=6."""
        assert config.TABLE_2.delta == 5e-4
        assert config.TABLE_2.delta_max == 8
        assert config.TABLE_2.l_max == 6

    def test_table_3(self):
        """T3: δ=2×10⁻³, Δ_max=22, L_max=20."""
        assert config.TABLE_3.delta == 2e-3
        assert config.TABLE_3.delta_max == 22
        assert config.TABLE_3.l_max == 20

    def test_table_4(self):
        """T4: δ=0.02, Δ_max=100, L_max=50."""
        assert config.TABLE_4.delta == 0.02
        assert config.TABLE_4.delta_max == 100
        assert config.TABLE_4.l_max == 50

    def test_table_5(self):
        """T5: δ=1, Δ_max=500, L_max=100."""
        assert config.TABLE_5.delta == 1.0
        assert config.TABLE_5.delta_max == 500
        assert config.TABLE_5.l_max == 100

    def test_full_discretization_count(self):
        """Full discretization should have 5 tables."""
        assert len(config.FULL_DISCRETIZATION) == 5

    def test_reduced_discretization_count(self):
        """Reduced discretization should have 2 tables (T1, T2)."""
        assert len(config.REDUCED_DISCRETIZATION) == 2


class TestIsingValues:
    """Test 3D Ising model reference values."""

    def test_ising_delta_sigma(self):
        """Known Δσ ≈ 0.5182."""
        assert abs(config.ISING_DELTA_SIGMA - 0.5182) < 0.001

    def test_ising_delta_epsilon(self):
        """Known Δε ≈ 1.413."""
        assert abs(config.ISING_DELTA_EPSILON - 1.413) < 0.01

    def test_ising_delta_epsilon_prime(self):
        """Expected Δε' ≈ 3.84 at n_max=10."""
        assert abs(config.ISING_DELTA_EPSILON_PRIME - 3.84) < 0.1
