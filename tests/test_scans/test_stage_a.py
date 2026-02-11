"""
Tests for the Stage A scan (Δε bound computation).

Tests the binary search logic, gap filtering, CSV I/O, and the
full pipeline with coarse discretization tables.

Reference: arXiv:1203.6064, Sections 5-6
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from ising_bootstrap.config import (
    N_MAX, BINARY_SEARCH_MAX_ITER,
    DEFAULT_SIGMA_MIN, DEFAULT_SIGMA_MAX, DEFAULT_SIGMA_STEP,
    DEFAULT_EPS_TOLERANCE,
    DiscretizationTable,
)
from ising_bootstrap.scans.stage_a import (
    ScanConfig,
    binary_search_eps,
    find_eps_bound,
    build_full_constraint_matrix,
    run_scan,
    write_csv_header,
    append_result_to_csv,
    load_scan_results,
)
from ising_bootstrap.spectrum.discretization import (
    SpectrumPoint, generate_full_spectrum,
)
from ising_bootstrap.lp.constraint_matrix import (
    build_constraint_matrix_from_cache,
    precompute_extended_blocks,
)
from ising_bootstrap.lp.solver import FeasibilityResult


# ============================================================================
# Tests for binary search logic
# ============================================================================

class TestBinarySearchLogic:
    """Tests for the generic binary search function."""

    def test_finds_threshold(self):
        """Binary search should find the threshold within tolerance."""
        threshold = 1.41
        is_excluded = lambda gap: gap > threshold
        result, n_iter = binary_search_eps(is_excluded, 0.5, 2.5, 1e-4, 50)
        assert abs(result - threshold) < 1e-4

    def test_excluded_lowers_hi(self):
        """When excluded, the upper bound should decrease."""
        # Threshold at 1.0: excluded above, allowed below
        calls = []

        def is_excluded(gap):
            calls.append(gap)
            return gap > 1.0

        result, _ = binary_search_eps(is_excluded, 0.0, 2.0, 0.01, 50)
        # Result should be close to 1.0
        assert abs(result - 1.0) < 0.01
        # First call should be at midpoint 1.0
        assert abs(calls[0] - 1.0) < 1e-10

    def test_allowed_raises_lo(self):
        """When allowed, the lower bound should increase."""
        # Everything is allowed (never excluded)
        result, n_iter = binary_search_eps(lambda gap: False, 0.5, 2.5, 0.01, 50)
        # lo should have converged near hi
        assert result > 2.4

    def test_always_excluded_converges_to_lo(self):
        """If everything is excluded, result should be near initial lo."""
        result, _ = binary_search_eps(lambda gap: True, 0.5, 2.5, 0.01, 50)
        assert result < 0.51

    def test_convergence_within_tolerance(self):
        """Binary search should converge within specified tolerance."""
        threshold = 1.234
        tol = 1e-3
        result, _ = binary_search_eps(
            lambda gap: gap > threshold, 0.5, 2.5, tol, 50
        )
        assert abs(result - threshold) < tol

    def test_max_iterations_respected(self):
        """Should not exceed max_iter iterations."""
        calls = []

        def is_excluded(gap):
            calls.append(gap)
            return gap > 1.5

        _, n_iter = binary_search_eps(is_excluded, 0.0, 100.0, 1e-20, 10)
        assert n_iter == 10
        assert len(calls) == 10

    def test_narrow_range(self):
        """Should handle a range narrower than tolerance."""
        result, n_iter = binary_search_eps(
            lambda gap: gap > 1.0, 0.999, 1.001, 0.01, 50
        )
        assert n_iter == 1  # immediately converges
        assert 0.999 <= result <= 1.001

    def test_different_thresholds(self):
        """Binary search should work for various threshold values."""
        for threshold in [0.6, 1.0, 1.5, 2.0, 2.3]:
            result, _ = binary_search_eps(
                lambda gap, t=threshold: gap > t, 0.5, 2.5, 1e-4, 50
            )
            assert abs(result - threshold) < 1e-4, \
                f"Failed for threshold={threshold}: got {result}"


# ============================================================================
# Tests for gap filtering
# ============================================================================

class TestGapFiltering:
    """Tests for row-subsetting logic in find_eps_bound."""

    def _make_spectrum(self):
        """Create a small spectrum for testing gap filtering."""
        return [
            SpectrumPoint(delta=0.6, spin=0, table="T1"),
            SpectrumPoint(delta=1.0, spin=0, table="T1"),
            SpectrumPoint(delta=1.5, spin=0, table="T1"),
            SpectrumPoint(delta=2.0, spin=0, table="T1"),
            SpectrumPoint(delta=3.0, spin=2, table="T2"),
            SpectrumPoint(delta=4.0, spin=2, table="T2"),
            SpectrumPoint(delta=5.0, spin=4, table="T2"),
        ]

    def _make_masks(self, spectrum):
        scalar_mask = np.array([p.spin == 0 for p in spectrum], dtype=bool)
        spinning_mask = ~scalar_mask
        scalar_deltas = np.array([p.delta for p in spectrum], dtype=np.float64)
        return scalar_mask, scalar_deltas, spinning_mask

    def test_no_gap_includes_all(self):
        """With gap at unitarity bound, all operators should be included."""
        spectrum = self._make_spectrum()
        scalar_mask, scalar_deltas, spinning_mask = self._make_masks(spectrum)
        gap = 0.5
        mask = spinning_mask | (scalar_mask & (scalar_deltas >= gap - 1e-10))
        assert np.sum(mask) == len(spectrum)

    def test_large_gap_excludes_all_scalars(self):
        """With gap above all scalar deltas, only spinning operators remain."""
        spectrum = self._make_spectrum()
        scalar_mask, scalar_deltas, spinning_mask = self._make_masks(spectrum)
        gap = 100.0
        mask = spinning_mask | (scalar_mask & (scalar_deltas >= gap - 1e-10))
        assert np.sum(mask) == 3  # only spinning operators

    def test_gap_preserves_spinning(self):
        """Spinning operators should never be filtered out."""
        spectrum = self._make_spectrum()
        scalar_mask, scalar_deltas, spinning_mask = self._make_masks(spectrum)
        for gap in [0.5, 1.0, 2.0, 100.0]:
            mask = spinning_mask | (scalar_mask & (scalar_deltas >= gap - 1e-10))
            assert np.sum(mask & spinning_mask) == 3

    def test_gap_boundary_inclusion(self):
        """Scalar with delta exactly at gap boundary should be included."""
        spectrum = self._make_spectrum()
        scalar_mask, scalar_deltas, spinning_mask = self._make_masks(spectrum)
        gap = 1.5
        mask = spinning_mask | (scalar_mask & (scalar_deltas >= gap - 1e-10))
        # Scalars at 1.5 and 2.0 should be included, 0.6 and 1.0 excluded
        scalar_included = np.sum(mask & scalar_mask)
        assert scalar_included == 2  # delta=1.5 and delta=2.0

    def test_progressive_gap(self):
        """Increasing gap should monotonically decrease included scalars."""
        spectrum = self._make_spectrum()
        scalar_mask, scalar_deltas, spinning_mask = self._make_masks(spectrum)
        prev_count = len(spectrum)
        for gap in [0.5, 0.7, 1.1, 1.6, 2.1, 3.0]:
            mask = spinning_mask | (scalar_mask & (scalar_deltas >= gap - 1e-10))
            count = np.sum(mask)
            assert count <= prev_count, \
                f"Count increased from {prev_count} to {count} at gap={gap}"
            prev_count = count


class TestFailureHandling:
    """Tests for strict fail-fast behavior in Stage A."""

    def test_solver_failure_raises_immediately(self, monkeypatch):
        import ising_bootstrap.scans.stage_a as stage_a_module

        def _failing_solver(*args, **kwargs):
            return FeasibilityResult(
                excluded=False,
                status="simulated solver failure",
                lp_status=-1,
                success=False,
            )

        monkeypatch.setattr(stage_a_module, "check_feasibility", _failing_solver)

        config = ScanConfig(
            tolerance=0.1,
            max_iter=3,
            eps_lo=0.5,
            eps_hi=1.5,
        )

        A = np.array([[1.0], [1.0]], dtype=np.float64)
        f_id = np.array([1.0], dtype=np.float64)
        scalar_mask = np.array([True, False], dtype=bool)
        scalar_deltas = np.array([1.0, 3.0], dtype=np.float64)
        spinning_mask = ~scalar_mask

        with pytest.raises(RuntimeError, match="Solver failed while testing gap"):
            find_eps_bound(
                0.518,
                A,
                f_id,
                scalar_mask,
                scalar_deltas,
                spinning_mask,
                config,
            )


# ============================================================================
# Tests for CSV I/O
# ============================================================================

class TestCSVIO:
    """Tests for CSV read/write utilities."""

    def test_write_and_read_roundtrip(self, tmp_path):
        """Writing and reading should produce the same data."""
        csv_path = tmp_path / "test_results.csv"
        write_csv_header(csv_path)
        append_result_to_csv(csv_path, 0.500, 1.234)
        append_result_to_csv(csv_path, 0.502, 1.238)
        append_result_to_csv(csv_path, 0.504, 1.242)

        results = load_scan_results(csv_path)
        assert len(results) == 3
        assert abs(results[0][0] - 0.500) < 1e-6
        assert abs(results[0][1] - 1.234) < 1e-6
        assert abs(results[2][0] - 0.504) < 1e-6
        assert abs(results[2][1] - 1.242) < 1e-6

    def test_header_format(self, tmp_path):
        """CSV header should have expected column names."""
        csv_path = tmp_path / "test.csv"
        write_csv_header(csv_path)
        with open(csv_path) as f:
            header = f.readline().strip()
        assert header == "delta_sigma,delta_eps_max"

    def test_append_mode(self, tmp_path):
        """Multiple appends should create a valid multi-row file."""
        csv_path = tmp_path / "test.csv"
        write_csv_header(csv_path)
        for i in range(10):
            append_result_to_csv(csv_path, 0.50 + i * 0.002, 1.0 + i * 0.01)
        results = load_scan_results(csv_path)
        assert len(results) == 10

    def test_empty_file(self, tmp_path):
        """Loading a header-only file should return empty list."""
        csv_path = tmp_path / "empty.csv"
        write_csv_header(csv_path)
        results = load_scan_results(csv_path)
        assert results == []


# ============================================================================
# Tests for ScanConfig
# ============================================================================

class TestScanConfig:
    """Tests for the ScanConfig dataclass."""

    def test_default_values(self):
        """Default config should use values from config.py."""
        config = ScanConfig()
        assert config.sigma_min == DEFAULT_SIGMA_MIN
        assert config.sigma_max == DEFAULT_SIGMA_MAX
        assert config.sigma_step == DEFAULT_SIGMA_STEP
        assert config.tolerance == DEFAULT_EPS_TOLERANCE
        assert config.max_iter == BINARY_SEARCH_MAX_ITER

    def test_sigma_grid_count(self):
        """Default grid should have 51 points."""
        config = ScanConfig()
        grid = config.get_sigma_grid()
        assert len(grid) == 51

    def test_custom_grid(self):
        """Custom range should produce correct grid."""
        config = ScanConfig(sigma_min=0.51, sigma_max=0.53, sigma_step=0.005)
        grid = config.get_sigma_grid()
        assert len(grid) == 5  # 0.51, 0.515, 0.52, 0.525, 0.53

    def test_reduced_tables(self):
        """Reduced flag should select T1-T2 only."""
        config = ScanConfig(reduced=True)
        tables = config.get_tables()
        assert len(tables) == 2
        names = [t.name for t in tables]
        assert "T1" in names
        assert "T2" in names

    def test_full_tables(self):
        """Full tables should include T1-T5."""
        config = ScanConfig(reduced=False)
        tables = config.get_tables()
        assert len(tables) == 5


# ============================================================================
# Tests for extended cache
# ============================================================================

class TestExtendedCache:
    """Tests for extended H array caching."""

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        """Saving and loading an extended H array should roundtrip."""
        import ising_bootstrap.blocks.cache as cache_module
        monkeypatch.setattr(cache_module, 'CACHE_DIR', tmp_path)

        n_max = 2
        max_order = 2 * n_max + 1
        max_k = max_order // 2
        H = np.random.randn(max_order + 1, max_k + 1)

        from ising_bootstrap.blocks.cache import (
            save_extended_h_array, load_extended_h_array,
            extended_cache_exists,
        )

        assert not extended_cache_exists(1.5, 2)

        save_extended_h_array(1.5, 2, H, n_max=n_max, overwrite=True)
        assert extended_cache_exists(1.5, 2)

        H_loaded = load_extended_h_array(1.5, 2, n_max=n_max)
        np.testing.assert_array_equal(H, H_loaded)

    def test_shape_validation(self, tmp_path, monkeypatch):
        """Should reject H arrays with wrong shape."""
        import ising_bootstrap.blocks.cache as cache_module
        monkeypatch.setattr(cache_module, 'CACHE_DIR', tmp_path)

        from ising_bootstrap.blocks.cache import save_extended_h_array

        H_bad = np.zeros((5, 5))
        with pytest.raises(ValueError, match="shape"):
            save_extended_h_array(1.5, 0, H_bad, n_max=2)


# ============================================================================
# Integration tests with coarse spectrum
# ============================================================================

# Coarse tables for fast testing (~100 operators)
COARSE_T1 = DiscretizationTable("T1_test", delta=0.1, delta_max=3, l_max=0)
COARSE_T2 = DiscretizationTable("T2_test", delta=0.2, delta_max=8, l_max=6)


class TestStageAIntegration:
    """Integration tests for the Stage A scan pipeline.

    Uses coarse discretization tables (~100 operators) to keep
    tests manageable while exercising the full pipeline.
    """

    @pytest.mark.slow
    def test_single_sigma_runs(self):
        """find_eps_bound should run without errors at a single Δσ."""
        tables = [COARSE_T1, COARSE_T2]
        spectrum = generate_full_spectrum(tables=tables)
        h_cache = precompute_extended_blocks(spectrum, n_max=2)

        config = ScanConfig(
            tolerance=0.01, max_iter=20, n_max=2,
            tables=tables,
        )

        A, f_id, scalar_mask, scalar_deltas, spinning_mask = \
            build_full_constraint_matrix(
                spectrum, 0.518, h_cache, n_max=2
            )

        eps_max, n_iter = find_eps_bound(
            0.518, A, f_id,
            scalar_mask, scalar_deltas, spinning_mask,
            config,
        )

        assert isinstance(eps_max, float)
        assert 0.5 <= eps_max <= 2.5
        assert n_iter > 0

    @pytest.mark.slow
    def test_single_sigma_plausible(self):
        """At the Ising point, Δε_max should be in a plausible range.

        With coarse tables and n_max=2, the bound will be loose,
        but it should still be in the range [0.5, 2.5].
        """
        tables = [COARSE_T1, COARSE_T2]
        spectrum = generate_full_spectrum(tables=tables)
        h_cache = precompute_extended_blocks(spectrum, n_max=2)

        config = ScanConfig(
            tolerance=0.01, max_iter=30, n_max=2,
            tables=tables,
        )

        A, f_id, scalar_mask, scalar_deltas, spinning_mask = \
            build_full_constraint_matrix(
                spectrum, 0.5182, h_cache, n_max=2
            )

        eps_max, _ = find_eps_bound(
            0.5182, A, f_id,
            scalar_mask, scalar_deltas, spinning_mask,
            config,
        )

        # With coarse discretization and low n_max, the bound may be
        # very different from the production value (~1.41), but it
        # should be a valid result in range.
        assert 0.5 <= eps_max <= 2.5

    @pytest.mark.slow
    def test_scan_three_points(self, tmp_path):
        """Run a 3-point scan and verify results are valid."""
        output_path = tmp_path / "eps_bound_test.csv"
        config = ScanConfig(
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
        for delta_sigma, eps_max in results:
            assert 0.5 <= eps_max <= 2.5

        # Verify CSV output
        loaded = load_scan_results(output_path)
        assert len(loaded) == 3
        for (ds1, em1), (ds2, em2) in zip(results, loaded):
            assert abs(ds1 - ds2) < 1e-5
            assert abs(em1 - em2) < 1e-5
