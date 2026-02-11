"""
Tests for the Stage B scan (Δε' bound computation).

Tests the two-gap filtering logic, CSV I/O, configuration, and the
full pipeline with coarse discretization tables.

Reference: arXiv:1203.6064, Sections 5-6 and Figure 6
"""

import pytest
import numpy as np
from pathlib import Path

from ising_bootstrap.config import (
    BINARY_SEARCH_MAX_ITER,
    DEFAULT_SIGMA_MIN, DEFAULT_SIGMA_MAX, DEFAULT_SIGMA_STEP,
    DEFAULT_EPSPRIME_TOLERANCE,
    DiscretizationTable,
)
from ising_bootstrap.scans.stage_a import (
    build_full_constraint_matrix,
    write_csv_header as write_stage_a_header,
    append_result_to_csv as append_stage_a_result,
)
from ising_bootstrap.scans.stage_b import (
    StageBConfig,
    find_eps_prime_bound,
    run_scan,
    write_csv_header,
    append_result_to_csv,
    load_stage_b_results,
    load_eps_bound_map,
    _snap_delta_eps_to_scalar_grid,
)
from ising_bootstrap.spectrum.discretization import (
    SpectrumPoint, generate_full_spectrum,
)
from ising_bootstrap.lp.constraint_matrix import precompute_extended_blocks
from ising_bootstrap.lp.solver import FeasibilityResult


# ============================================================================
# Tests for two-gap filtering logic
# ============================================================================

class TestTwoGapFiltering:
    """Tests for two-gap row-subsetting logic used in Stage B."""

    def _make_spectrum(self):
        """Create a small spectrum for testing two-gap filtering."""
        return [
            SpectrumPoint(delta=0.6, spin=0, table="T1"),   # below eps
            SpectrumPoint(delta=1.0, spin=0, table="T1"),   # below eps
            SpectrumPoint(delta=1.5, spin=0, table="T1"),   # between eps and eps'
            SpectrumPoint(delta=2.0, spin=0, table="T1"),   # between eps and eps'
            SpectrumPoint(delta=3.0, spin=0, table="T1"),   # above eps'
            SpectrumPoint(delta=4.0, spin=0, table="T1"),   # above eps'
            SpectrumPoint(delta=3.0, spin=2, table="T2"),   # spinning
            SpectrumPoint(delta=4.0, spin=2, table="T2"),   # spinning
            SpectrumPoint(delta=5.0, spin=4, table="T2"),   # spinning
        ]

    def _make_masks(self, spectrum):
        scalar_mask = np.array([p.spin == 0 for p in spectrum], dtype=bool)
        spinning_mask = ~scalar_mask
        scalar_deltas = np.array([p.delta for p in spectrum], dtype=np.float64)
        return scalar_mask, scalar_deltas, spinning_mask

    def _apply_two_gap_mask(self, scalar_mask, scalar_deltas, spinning_mask,
                            delta_eps, delta_eps_prime):
        """Apply two-gap row selection (same logic as find_eps_prime_bound)."""
        scalar_at_eps = scalar_mask & np.isclose(
            scalar_deltas, delta_eps, atol=1e-10, rtol=0.0
        )
        scalars_above_eps_prime = scalar_mask & (
            scalar_deltas >= delta_eps_prime - 1e-10
        )
        return spinning_mask | scalar_at_eps | scalars_above_eps_prime

    def test_two_gaps_exclude_middle_scalars(self):
        """Scalars between Δε and Δε' should be excluded."""
        spectrum = self._make_spectrum()
        scalar_mask, scalar_deltas, spinning_mask = self._make_masks(spectrum)

        delta_eps = 1.0
        delta_eps_prime = 2.5

        mask = self._apply_two_gap_mask(
            scalar_mask, scalar_deltas, spinning_mask,
            delta_eps, delta_eps_prime,
        )

        # Should include: scalar at 1.0 (anchored epsilon),
        #                 scalars at 3.0, 4.0 (above eps'),
        #                 3 spinning operators
        # Should exclude: scalar below eps (0.6) and in-gap scalars (1.5, 2.0)
        assert np.sum(mask) == 6

    def test_no_second_gap_includes_all_above_eps(self):
        """With Δε' = Δε, all scalars at/above ε are included plus anchored ε."""
        spectrum = self._make_spectrum()
        scalar_mask, scalar_deltas, spinning_mask = self._make_masks(spectrum)

        delta_eps = 1.0
        delta_eps_prime = delta_eps  # no second gap

        mask = self._apply_two_gap_mask(
            scalar_mask, scalar_deltas, spinning_mask,
            delta_eps, delta_eps_prime,
        )

        # Scalars included: 1.0, 1.5, 2.0, 3.0, 4.0
        # Spinning: 3
        # Scalar below epsilon (0.6) remains excluded
        assert np.sum(mask) == 8

    def test_very_large_second_gap_excludes_all_scalars_above_eps(self):
        """Large Δε' should exclude all scalars between Δε and Δε'."""
        spectrum = self._make_spectrum()
        scalar_mask, scalar_deltas, spinning_mask = self._make_masks(spectrum)

        delta_eps = 1.0
        delta_eps_prime = 100.0  # above all scalars

        mask = self._apply_two_gap_mask(
            scalar_mask, scalar_deltas, spinning_mask,
            delta_eps, delta_eps_prime,
        )

        # Only anchored epsilon scalar (1.0) and spinning (3) remain
        assert np.sum(mask) == 4

    def test_spinning_always_included(self):
        """Spinning operators should always be included regardless of gaps."""
        spectrum = self._make_spectrum()
        scalar_mask, scalar_deltas, spinning_mask = self._make_masks(spectrum)

        for delta_eps in [1.0, 2.0, 4.0]:
            for delta_eps_prime in [delta_eps, 2.0, 10.0, 100.0]:
                mask = self._apply_two_gap_mask(
                    scalar_mask, scalar_deltas, spinning_mask,
                    delta_eps, delta_eps_prime,
                )
                assert np.sum(mask & spinning_mask) == 3

    def test_gap_boundary_precision(self):
        """Scalars at exactly the boundary should be handled correctly."""
        spectrum = self._make_spectrum()
        scalar_mask, scalar_deltas, spinning_mask = self._make_masks(spectrum)

        # Set eps' exactly at a scalar value (3.0)
        delta_eps = 1.0
        delta_eps_prime = 3.0

        mask = self._apply_two_gap_mask(
            scalar_mask, scalar_deltas, spinning_mask,
            delta_eps, delta_eps_prime,
        )

        # Scalar at 1.0 is the anchor and must be included.
        included_scalar_deltas = scalar_deltas[mask & scalar_mask]
        assert 1.0 in included_scalar_deltas

        # Scalar at 3.0 should be included (>= eps' - 1e-10)
        # Scalars at 1.5, 2.0 should be excluded
        assert 3.0 in included_scalar_deltas
        assert 1.5 not in included_scalar_deltas
        assert 2.0 not in included_scalar_deltas

    def test_progressive_second_gap(self):
        """Increasing Δε' should monotonically decrease included operators."""
        spectrum = self._make_spectrum()
        scalar_mask, scalar_deltas, spinning_mask = self._make_masks(spectrum)

        delta_eps = 1.0
        prev_count = np.inf
        for delta_eps_prime in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 100.0]:
            mask = self._apply_two_gap_mask(
                scalar_mask, scalar_deltas, spinning_mask,
                delta_eps, delta_eps_prime,
            )
            count = np.sum(mask)
            assert count <= prev_count, \
                f"Count increased from {prev_count} to {count} " \
                f"at eps_prime={delta_eps_prime}"
            prev_count = count


class TestEpsilonAnchoring:
    """Tests for Stage B epsilon anchoring onto the scalar grid."""

    def test_snap_to_nearest_scalar(self):
        scalar_mask = np.array([True, True, True, False], dtype=bool)
        scalar_deltas = np.array([0.6, 1.0, 1.5, 3.0], dtype=np.float64)
        snapped = _snap_delta_eps_to_scalar_grid(
            1.01, scalar_deltas, scalar_mask, tolerance=0.05
        )
        assert abs(snapped - 1.0) < 1e-12

    def test_snap_fails_outside_tolerance(self):
        scalar_mask = np.array([True, True, True], dtype=bool)
        scalar_deltas = np.array([0.6, 1.0, 1.5], dtype=np.float64)
        with pytest.raises(RuntimeError, match="exceeding tolerance"):
            _snap_delta_eps_to_scalar_grid(
                1.2, scalar_deltas, scalar_mask, tolerance=1e-3
            )

    def test_snap_fails_without_scalars(self):
        scalar_mask = np.array([False, False], dtype=bool)
        scalar_deltas = np.array([3.0, 4.0], dtype=np.float64)
        with pytest.raises(RuntimeError, match="No scalar operators"):
            _snap_delta_eps_to_scalar_grid(
                1.0, scalar_deltas, scalar_mask, tolerance=0.1
            )


# ============================================================================
# Tests for CSV I/O
# ============================================================================

class TestCSVIO:
    """Tests for Stage B CSV read/write utilities."""

    def test_write_and_read_roundtrip(self, tmp_path):
        """Writing and reading should produce the same data."""
        csv_path = tmp_path / "stage_b_results.csv"
        write_csv_header(csv_path)
        append_result_to_csv(csv_path, 0.500, 1.234, 3.456)
        append_result_to_csv(csv_path, 0.502, 1.238, 3.480)
        append_result_to_csv(csv_path, 0.504, 1.242, 3.510)

        results = load_stage_b_results(csv_path)
        assert len(results) == 3
        assert abs(results[0][0] - 0.500) < 1e-6
        assert abs(results[0][1] - 1.234) < 1e-6
        assert abs(results[0][2] - 3.456) < 1e-6
        assert abs(results[2][0] - 0.504) < 1e-6
        assert abs(results[2][2] - 3.510) < 1e-6

    def test_header_format(self, tmp_path):
        """CSV header should have expected 3-column format."""
        csv_path = tmp_path / "test.csv"
        write_csv_header(csv_path)
        with open(csv_path) as f:
            header = f.readline().strip()
        assert header == "delta_sigma,delta_eps,delta_eps_prime_max"

    def test_append_mode(self, tmp_path):
        """Multiple appends should create a valid multi-row file."""
        csv_path = tmp_path / "test.csv"
        write_csv_header(csv_path)
        for i in range(10):
            append_result_to_csv(
                csv_path,
                0.50 + i * 0.002,
                1.0 + i * 0.01,
                3.0 + i * 0.05,
            )
        results = load_stage_b_results(csv_path)
        assert len(results) == 10

    def test_empty_file(self, tmp_path):
        """Loading a header-only file should return empty list."""
        csv_path = tmp_path / "empty.csv"
        write_csv_header(csv_path)
        results = load_stage_b_results(csv_path)
        assert results == []

    def test_three_column_values(self, tmp_path):
        """Each result should have exactly 3 values."""
        csv_path = tmp_path / "test.csv"
        write_csv_header(csv_path)
        append_result_to_csv(csv_path, 0.518, 1.41, 3.84)

        results = load_stage_b_results(csv_path)
        assert len(results) == 1
        ds, de, dep = results[0]
        assert abs(ds - 0.518) < 1e-3
        assert abs(de - 1.41) < 1e-2
        assert abs(dep - 3.84) < 1e-2


# ============================================================================
# Tests for StageBConfig
# ============================================================================

class TestStageBConfig:
    """Tests for the StageBConfig dataclass."""

    def test_default_values(self):
        """Default config should use values from config.py."""
        config = StageBConfig()
        assert config.sigma_min == DEFAULT_SIGMA_MIN
        assert config.sigma_max == DEFAULT_SIGMA_MAX
        assert config.sigma_step == DEFAULT_SIGMA_STEP
        assert config.tolerance == DEFAULT_EPSPRIME_TOLERANCE
        assert config.max_iter == BINARY_SEARCH_MAX_ITER
        assert config.eps_bound_path is None

    def test_sigma_grid_count(self):
        """Default grid should have 51 points."""
        config = StageBConfig()
        grid = config.get_sigma_grid()
        assert len(grid) == 51

    def test_custom_grid(self):
        """Custom range should produce correct grid."""
        config = StageBConfig(sigma_min=0.51, sigma_max=0.53, sigma_step=0.005)
        grid = config.get_sigma_grid()
        assert len(grid) == 5

    def test_reduced_tables(self):
        """Reduced flag should select T1-T2 only."""
        config = StageBConfig(reduced=True)
        tables = config.get_tables()
        assert len(tables) == 2
        names = [t.name for t in tables]
        assert "T1" in names
        assert "T2" in names

    def test_full_tables(self):
        """Full tables should include T1-T5."""
        config = StageBConfig(reduced=False)
        tables = config.get_tables()
        assert len(tables) == 5

    def test_eps_bound_path(self):
        """Should accept and store eps_bound_path."""
        config = StageBConfig(eps_bound_path=Path("/tmp/eps_bound.csv"))
        assert config.eps_bound_path == Path("/tmp/eps_bound.csv")


# ============================================================================
# Tests for Stage A result loading
# ============================================================================

class TestLoadEpsBoundMap:
    """Tests for loading Stage A results."""

    def test_load_creates_mapping(self, tmp_path):
        """Should create a dict mapping delta_sigma to delta_eps."""
        csv_path = tmp_path / "stage_a.csv"
        write_stage_a_header(csv_path)
        append_stage_a_result(csv_path, 0.510, 1.200)
        append_stage_a_result(csv_path, 0.520, 1.300)
        append_stage_a_result(csv_path, 0.530, 1.400)

        mapping = load_eps_bound_map(csv_path)
        assert len(mapping) == 3
        assert abs(mapping[0.51] - 1.200) < 1e-3
        assert abs(mapping[0.52] - 1.300) < 1e-3
        assert abs(mapping[0.53] - 1.400) < 1e-3

    def test_empty_csv(self, tmp_path):
        """Empty CSV should produce empty dict."""
        csv_path = tmp_path / "empty.csv"
        write_stage_a_header(csv_path)
        mapping = load_eps_bound_map(csv_path)
        assert mapping == {}

    def test_rounding_for_matching(self, tmp_path):
        """Keys should be rounded for robust grid matching."""
        csv_path = tmp_path / "stage_a.csv"
        write_stage_a_header(csv_path)
        append_stage_a_result(csv_path, 0.518200, 1.410)

        mapping = load_eps_bound_map(csv_path)
        assert 0.5182 in mapping

    def test_nonfinite_values_are_dropped(self, tmp_path):
        """NaN/inf values should be filtered out before Stage B."""
        csv_path = tmp_path / "stage_a.csv"
        write_stage_a_header(csv_path)
        append_stage_a_result(csv_path, 0.510, 1.200)
        append_stage_a_result(csv_path, 0.520, float("nan"))
        append_stage_a_result(csv_path, 0.530, float("inf"))

        mapping = load_eps_bound_map(csv_path)
        assert len(mapping) == 1
        assert abs(mapping[0.51] - 1.2) < 1e-6


# ============================================================================
# Tests for run_scan validation
# ============================================================================

class TestRunScanValidation:
    """Tests for scan configuration validation."""

    def test_missing_eps_bound_raises(self):
        """run_scan should raise if eps_bound_path is None."""
        config = StageBConfig(eps_bound_path=None)
        with pytest.raises(ValueError, match="eps_bound_path"):
            run_scan(config)

    def test_missing_sigma_points_raise(self, tmp_path):
        """Strict mode should fail if any requested Δσ is missing in Stage A."""
        # Create Stage A CSV with only one point
        stage_a_path = tmp_path / "stage_a.csv"
        write_stage_a_header(stage_a_path)
        append_stage_a_result(stage_a_path, 0.520, 0.500)

        output_path = tmp_path / "stage_b.csv"

        config = StageBConfig(
            eps_bound_path=stage_a_path,
            sigma_min=0.51,
            sigma_max=0.53,
            sigma_step=0.01,
            tolerance=0.01,
            max_iter=5,
            n_max=2,
            tables=[
                DiscretizationTable("T1_test", delta=0.1, delta_max=3, l_max=0),
                DiscretizationTable("T2_test", delta=0.2, delta_max=8, l_max=6),
            ],
            output=output_path,
        )

        with pytest.raises(RuntimeError, match="Missing valid Stage A"):
            run_scan(config)


class TestFailureHandling:
    """Tests for strict fail-fast behavior in Stage B."""

    def test_solver_failure_raises_immediately(self, monkeypatch):
        import ising_bootstrap.scans.stage_b as stage_b_module

        def _failing_solver(*args, **kwargs):
            return FeasibilityResult(
                excluded=False,
                status="simulated solver failure",
                lp_status=-1,
                success=False,
            )

        monkeypatch.setattr(stage_b_module, "check_feasibility", _failing_solver)

        config = StageBConfig(
            tolerance=0.1,
            max_iter=3,
            eps_prime_hi=2.0,
            eps_snap_tolerance=1e-6,
        )
        A = np.array([[1.0], [1.0]], dtype=np.float64)
        f_id = np.array([1.0], dtype=np.float64)
        scalar_mask = np.array([True, False], dtype=bool)
        scalar_deltas = np.array([1.0, 3.0], dtype=np.float64)
        spinning_mask = ~scalar_mask

        with pytest.raises(RuntimeError, match="Solver failed while testing"):
            find_eps_prime_bound(
                0.518,
                1.0,
                A,
                f_id,
                scalar_mask,
                scalar_deltas,
                spinning_mask,
                config,
            )


# ============================================================================
# Integration tests with coarse spectrum
# ============================================================================

# Coarse tables for fast testing (~219 operators)
COARSE_T1 = DiscretizationTable("T1_test", delta=0.1, delta_max=3, l_max=0)
COARSE_T2 = DiscretizationTable("T2_test", delta=0.2, delta_max=8, l_max=6)


class TestStageBIntegration:
    """Integration tests for the Stage B scan pipeline.

    Uses coarse discretization tables (~219 operators) with n_max=2
    to keep tests manageable while exercising the full pipeline.
    """

    def _create_stage_a_csv(self, tmp_path, sigma_values, eps_values):
        """Helper to create a mock Stage A CSV file."""
        csv_path = tmp_path / "stage_a_mock.csv"
        write_stage_a_header(csv_path)
        for ds, de in zip(sigma_values, eps_values):
            append_stage_a_result(csv_path, ds, de)
        return csv_path

    @pytest.mark.slow
    def test_single_sigma_runs(self, tmp_path):
        """find_eps_prime_bound should run without errors at a single Δσ."""
        tables = [COARSE_T1, COARSE_T2]
        spectrum = generate_full_spectrum(tables=tables)
        h_cache = precompute_extended_blocks(spectrum, n_max=2)

        config = StageBConfig(
            tolerance=0.01, max_iter=20, n_max=2,
            tables=tables,
        )

        A, f_id, scalar_mask, scalar_deltas, spinning_mask = \
            build_full_constraint_matrix(
                spectrum, 0.518, h_cache, n_max=2
            )

        # Use a plausible delta_eps from Stage A
        delta_eps = 0.5  # unitarity bound (conservative)

        eps_prime_max, n_iter = find_eps_prime_bound(
            0.518, delta_eps,
            A, f_id,
            scalar_mask, scalar_deltas, spinning_mask,
            config,
        )

        assert isinstance(eps_prime_max, float)
        assert eps_prime_max >= delta_eps
        assert eps_prime_max <= 6.0
        assert n_iter > 0

    @pytest.mark.slow
    def test_scan_three_points(self, tmp_path):
        """Run a 3-point Stage B scan and verify results are valid."""
        # Create mock Stage A results
        sigma_values = [0.51, 0.52, 0.53]
        eps_values = [0.50, 0.50, 0.50]  # unitarity bound
        stage_a_path = self._create_stage_a_csv(
            tmp_path, sigma_values, eps_values
        )

        output_path = tmp_path / "epsprime_bound_test.csv"
        config = StageBConfig(
            eps_bound_path=stage_a_path,
            sigma_min=0.51,
            sigma_max=0.53,
            sigma_step=0.01,
            tolerance=0.01,
            max_iter=20,
            n_max=2,
            tables=[COARSE_T1, COARSE_T2],
            output=output_path,
        )

        results = run_scan(config)

        assert len(results) == 3
        for delta_sigma, delta_eps, eps_prime_max in results:
            assert 0.5 <= eps_prime_max <= 6.0
            assert delta_eps >= 0.5

        # Verify CSV output
        loaded = load_stage_b_results(output_path)
        assert len(loaded) == 3
        for (ds1, de1, dp1), (ds2, de2, dp2) in zip(results, loaded):
            assert abs(ds1 - ds2) < 1e-5
            assert abs(de1 - de2) < 1e-5
            assert abs(dp1 - dp2) < 1e-5

    @pytest.mark.slow
    def test_csv_round_trip_through_pipeline(self, tmp_path):
        """Full pipeline: Stage A CSV → Stage B scan → Stage B CSV."""
        # Create mock Stage A results
        stage_a_path = self._create_stage_a_csv(
            tmp_path,
            [0.52],
            [0.50],
        )

        output_path = tmp_path / "epsprime_test.csv"
        config = StageBConfig(
            eps_bound_path=stage_a_path,
            sigma_min=0.52,
            sigma_max=0.52,
            sigma_step=0.01,
            tolerance=0.01,
            max_iter=20,
            n_max=2,
            tables=[COARSE_T1, COARSE_T2],
            output=output_path,
        )

        results = run_scan(config)
        assert len(results) == 1

        # Load and verify
        loaded = load_stage_b_results(output_path)
        assert len(loaded) == 1
        ds, de, dp = loaded[0]
        assert abs(ds - 0.52) < 1e-4
        assert de >= 0.5
        assert dp >= de
