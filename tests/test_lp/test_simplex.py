"""
Tests for the extended-precision simplex solver.

Tests the Phase I simplex, column generation, and end-to-end feasibility
checks at both n_max=2 (well-conditioned) and n_max=10 (production).

Reference: arXiv:1203.6064
"""

import pytest
import numpy as np

from ising_bootstrap.config import (
    N_MAX, DiscretizationTable,
    ISING_DELTA_SIGMA,
)
from ising_bootstrap.spectrum.index_set import generate_index_set
from ising_bootstrap.spectrum.discretization import (
    SpectrumPoint, generate_full_spectrum,
)
from ising_bootstrap.lp.crossing import (
    compute_prefactor_table, compute_identity_vector,
    compute_crossing_vector_fast, build_comb_cache,
    compute_prefactor_table_mp, compute_identity_vector_mp,
    compute_extended_h_array_mp, compute_crossing_vector_mp,
)
from ising_bootstrap.lp.constraint_matrix import (
    build_constraint_matrix,
    build_constraint_matrix_from_cache,
)
from ising_bootstrap.lp.solver import (
    FeasibilityResult, check_feasibility,
)
from ising_bootstrap.lp.simplex import (
    _phase1_simplex, Phase1Result,
    check_feasibility_extended,
)


# ============================================================================
# Test Phase I simplex on synthetic problems
# ============================================================================

class TestPhase1Synthetic:
    """Test the raw Phase I simplex on small hand-built systems."""

    def test_trivially_feasible(self):
        """System x1 + x2 = 1, x >= 0 is feasible."""
        from mpmath import mpf
        columns = [
            [mpf(1)],  # x1
            [mpf(1)],  # x2
        ]
        b = [mpf(1)]
        result = _phase1_simplex(columns, b, m=1, dps=30)
        assert result.feasible is True
        assert float(result.objective) < 1e-20

    def test_identity_system(self):
        """System I x = b, x >= 0 is feasible when b >= 0."""
        from mpmath import mpf
        columns = [
            [mpf(1), mpf(0), mpf(0)],
            [mpf(0), mpf(1), mpf(0)],
            [mpf(0), mpf(0), mpf(1)],
        ]
        b = [mpf(2), mpf(3), mpf(5)]
        result = _phase1_simplex(columns, b, m=3, dps=30)
        assert result.feasible is True

    def test_infeasible_negative_rhs(self):
        """System x = -1, x >= 0 has no solution.

        Phase I is set up with sign-adjusted RHS, so this becomes
        -x + a = 1, and the artificial can never be driven out.
        """
        from mpmath import mpf
        # We need the sign-adjusted version: D*A^T*v + a = |b|
        # If b = [-1], D = -1, so column becomes [-1], b_adj = [1]
        columns = [
            [mpf(-1)],  # D * column = -1 * 1 = -1
        ]
        b = [mpf(1)]  # |b| = 1
        result = _phase1_simplex(columns, b, m=1, dps=30)
        # x = -1 can't satisfy x >= 0, so Phase I should fail
        # But actually with column [-1], we need v*(-1) + a = 1, v >= 0, a >= 0
        # If v=0, a=1 (not feasible). Can v enter? rc = 0 - pi*(-1) = pi.
        # pi = 1 (cost of artificial). So rc = 1 > 0? No, rc should be negative to enter.
        # rc = c_v - pi^T * col = 0 - 1*(-1) = 1 > 0. Can't enter. Infeasible.
        assert result.feasible is False

    def test_2d_feasible(self):
        """System [1,1; 2,1] x = [3; 4], x >= 0 is feasible (x=[1,2])."""
        from mpmath import mpf
        columns = [
            [mpf(1), mpf(2)],  # x1
            [mpf(1), mpf(1)],  # x2
        ]
        b = [mpf(3), mpf(4)]
        result = _phase1_simplex(columns, b, m=2, dps=30)
        assert result.feasible is True

    def test_overdetermined_infeasible(self):
        """Overdetermined system with no non-negative solution."""
        from mpmath import mpf
        # x1 = 1, x1 = 2 (contradictory)
        columns = [
            [mpf(1), mpf(1)],
        ]
        b = [mpf(1), mpf(2)]
        result = _phase1_simplex(columns, b, m=2, dps=30)
        assert result.feasible is False


# ============================================================================
# Test Farkas-based feasibility on synthetic bootstrap problems
# ============================================================================

class TestFarkasFeasibility:
    """Test that Farkas approach gives correct excluded/allowed results."""

    def test_obviously_feasible_excluded(self):
        """A simple problem where a functional clearly exists -> EXCLUDED."""
        A = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])
        f_id = np.array([1.0, 0.0, 0.0])
        spectrum = [
            SpectrumPoint(delta=1.0, spin=0, table="test"),
            SpectrumPoint(delta=2.0, spin=0, table="test"),
        ]
        result = check_feasibility_extended(
            A, f_id, spectrum, delta_sigma=0.5,
            n_max=2, dps=30, dps_verify=None,
        )
        assert result.excluded is True

    def test_obviously_infeasible_allowed(self):
        """A problem where no functional can exist -> ALLOWED."""
        A = np.array([[-1.0, 0.0]])
        f_id = np.array([1.0, 0.0])
        spectrum = [
            SpectrumPoint(delta=1.0, spin=0, table="test"),
        ]
        result = check_feasibility_extended(
            A, f_id, spectrum, delta_sigma=0.5,
            n_max=2, dps=30, dps_verify=None,
        )
        assert result.excluded is False


# ============================================================================
# Cross-validation: extended precision vs scipy at n_max=2
# ============================================================================

COARSE_T1 = DiscretizationTable("T1_test", delta=0.1, delta_max=3, l_max=0)
COARSE_T2 = DiscretizationTable("T2_test", delta=0.2, delta_max=8, l_max=6)


class TestCrossValidation:
    """Verify extended-precision solver agrees with scipy at n_max=2."""

    @pytest.mark.slow
    def test_nmax2_agrees_with_scipy(self):
        """At n_max=2, both solvers should agree on feasibility."""
        tables = [COARSE_T1, COARSE_T2]
        spectrum = generate_full_spectrum(tables=tables)
        A, f_id = build_constraint_matrix(spectrum, 0.518, n_max=2)

        result_scipy = check_feasibility(A, f_id, scale=False)
        result_ext = check_feasibility_extended(
            A, f_id, spectrum, delta_sigma=0.518,
            n_max=2, dps=30, dps_verify=None,
        )
        assert result_scipy.excluded == result_ext.excluded


# ============================================================================
# n_max=10 sanity checks
# ============================================================================

class TestNmax10Sanity:
    """Sanity checks at production n_max=10."""

    @pytest.mark.slow
    def test_no_gap_is_allowed(self):
        """Full spectrum with no gap should be ALLOWED (not excluded)."""
        tables = [COARSE_T1, COARSE_T2]
        spectrum = generate_full_spectrum(tables=tables)
        A, f_id = build_constraint_matrix(spectrum, 0.518, n_max=10)

        result = check_feasibility_extended(
            A, f_id, spectrum, delta_sigma=0.518,
            n_max=10, dps=50, dps_verify=None,
            verbose=True,
        )
        assert result.excluded is False, (
            f"No-gap spectrum should be ALLOWED, got: {result.status}"
        )

    @pytest.mark.slow
    def test_large_gap_is_excluded(self):
        """A gap of 100.0 should be easily EXCLUDED."""
        tables = [COARSE_T1, COARSE_T2]
        spectrum = generate_full_spectrum(tables=tables)
        A, f_id = build_constraint_matrix(spectrum, 0.518, n_max=10)

        # Filter to only spinning + scalars above gap 100
        scalar_mask = np.array([p.spin == 0 for p in spectrum])
        scalar_deltas = np.array([p.delta for p in spectrum])
        mask = ~scalar_mask | (scalar_mask & (scalar_deltas >= 100.0))
        A_sub = A[mask]
        spectrum_sub = [p for p, m in zip(spectrum, mask) if m]

        result = check_feasibility_extended(
            A_sub, f_id, spectrum_sub, delta_sigma=0.518,
            n_max=10, dps=50, dps_verify=None,
            verbose=True,
        )
        assert result.excluded is True, (
            f"Gap=100 should be EXCLUDED, got: {result.status}"
        )


# ============================================================================
# Test mpmath crossing vector accuracy
# ============================================================================

class TestMpmathCrossingVectors:
    """Verify mpmath crossing vectors are consistent with float64 versions."""

    def test_prefactor_table_mp_matches_f64(self):
        """mpmath prefactor table should match float64 to ~15 digits."""
        U_f64 = compute_prefactor_table(0.518, n_max=2)
        U_mp = compute_prefactor_table_mp(0.518, n_max=2, dps=50)

        max_order = 2 * 2 + 1
        max_k = max_order // 2
        for j in range(max_order + 1):
            for k in range(max_k + 1):
                if j + 2 * k <= max_order:
                    val_f64 = U_f64[j, k]
                    val_mp = float(U_mp[(j, k)])
                    if abs(val_f64) > 1e-30:
                        rel_err = abs(val_f64 - val_mp) / abs(val_f64)
                        assert rel_err < 1e-14, (
                            f"U[{j},{k}]: f64={val_f64}, mp={val_mp}, err={rel_err}"
                        )

    def test_identity_vector_mp_matches_f64(self):
        """mpmath identity vector should match float64."""
        f_id_f64 = compute_identity_vector(0.518, n_max=2)
        f_id_mp = compute_identity_vector_mp(0.518, n_max=2, dps=50)

        for i in range(len(f_id_f64)):
            val_f64 = f_id_f64[i]
            val_mp = float(f_id_mp[i])
            if abs(val_f64) > 1e-30:
                rel_err = abs(val_f64 - val_mp) / abs(val_f64)
                assert rel_err < 1e-14
