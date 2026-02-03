"""
Unit tests for block derivative caching.

Tests the disk caching functionality for precomputed block derivatives
h_{m,n}(Δ, l).

Cache format: NPZ files at data/cached_blocks/d{delta:.6f}_l{spin}.npz
"""

import pytest
import numpy as np
from pathlib import Path
from mpmath import mp, mpf
import tempfile
import shutil

from ising_bootstrap.blocks.cache import (
    get_cache_filename,
    cache_exists,
    save_block_derivatives,
    load_block_derivatives,
    load_block_derivatives_as_vector,
    compute_and_cache,
    get_or_compute,
    cache_stats,
    clear_cache,
)
from ising_bootstrap.blocks.transverse_derivs import block_derivatives_full
from ising_bootstrap.spectrum.index_set import generate_index_set
from ising_bootstrap.config import CACHE_DIR, N_MAX, MPMATH_PRECISION


# Set precision for tests
mp.dps = MPMATH_PRECISION


class TestGetCacheFilename:
    """Tests for cache filename generation."""

    def test_filename_format(self):
        """Filename should follow d{delta:.6f}_l{spin}.npz format."""
        path = get_cache_filename(1.5, 2)
        assert path.name == "d1.500000_l2.npz"

    def test_filename_precision(self):
        """Delta should be formatted with 6 decimal places."""
        path = get_cache_filename(1.234567890, 0)
        assert path.name == "d1.234568_l0.npz"  # Rounded

    def test_returns_path_object(self):
        """Should return a Path object."""
        path = get_cache_filename(1.0, 0)
        assert isinstance(path, Path)

    def test_in_cache_dir(self):
        """File should be in the CACHE_DIR."""
        path = get_cache_filename(1.0, 0)
        assert path.parent == CACHE_DIR


class TestCacheExists:
    """Tests for cache existence checking."""

    def test_nonexistent_cache(self):
        """Should return False for non-existent cache."""
        # Use an unlikely delta value
        assert not cache_exists(999.123456, 99)


class TestSaveLoadRoundTrip:
    """Tests for saving and loading block derivatives."""

    @pytest.fixture
    def temp_cache_dir(self, monkeypatch):
        """Create a temporary cache directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        # Patch CACHE_DIR to use temp directory
        import ising_bootstrap.blocks.cache as cache_module
        original_cache_dir = cache_module.CACHE_DIR
        cache_module.CACHE_DIR = temp_dir
        monkeypatch.setattr(cache_module, 'CACHE_DIR', temp_dir)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        cache_module.CACHE_DIR = original_cache_dir

    def test_save_and_load_dict(self, temp_cache_dir):
        """Save and load should preserve dictionary structure."""
        delta = 1.5
        spin = 0
        n_max = 3

        # Create test data
        index_set = generate_index_set(n_max)
        h_values = {(m, n): mpf(m + n * 0.1) for m, n in index_set}

        # Save
        save_block_derivatives(delta, spin, h_values, n_max=n_max, overwrite=True)

        # Load
        loaded = load_block_derivatives(delta, spin, n_max=n_max)

        # Compare
        for (m, n), value in h_values.items():
            assert abs(loaded[(m, n)] - float(value)) < 1e-10

    def test_save_and_load_vector(self, temp_cache_dir):
        """Save and load as vector should preserve order."""
        delta = 2.0
        spin = 2
        n_max = 3

        # Create test data
        index_set = generate_index_set(n_max)
        h_values = {(m, n): mpf(m * 0.5 + n * 0.2) for m, n in index_set}

        # Save
        save_block_derivatives(delta, spin, h_values, n_max=n_max, overwrite=True)

        # Load as vector
        vec = load_block_derivatives_as_vector(delta, spin, n_max=n_max)

        # Compare in order
        for i, (m, n) in enumerate(index_set):
            assert abs(vec[i] - float(h_values[(m, n)])) < 1e-10

    def test_overwrite_protection(self, temp_cache_dir):
        """Should raise FileExistsError without overwrite=True."""
        delta = 1.0
        spin = 0
        n_max = 3
        index_set = generate_index_set(n_max)
        h_values = {(m, n): mpf(1.0) for m, n in index_set}

        # First save
        save_block_derivatives(delta, spin, h_values, n_max=n_max, overwrite=True)

        # Second save without overwrite should fail
        with pytest.raises(FileExistsError):
            save_block_derivatives(delta, spin, h_values, n_max=n_max, overwrite=False)

    def test_n_max_mismatch_raises(self, temp_cache_dir):
        """Loading with wrong n_max should raise ValueError."""
        delta = 1.0
        spin = 0
        n_max = 3
        index_set = generate_index_set(n_max)
        h_values = {(m, n): mpf(1.0) for m, n in index_set}

        # Save with n_max=3
        save_block_derivatives(delta, spin, h_values, n_max=n_max, overwrite=True)

        # Load with n_max=5 should fail
        with pytest.raises(ValueError):
            load_block_derivatives(delta, spin, n_max=5)


class TestComputeAndCache:
    """Tests for compute_and_cache function."""

    @pytest.fixture
    def temp_cache_dir(self, monkeypatch):
        """Create a temporary cache directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        import ising_bootstrap.blocks.cache as cache_module
        original_cache_dir = cache_module.CACHE_DIR
        cache_module.CACHE_DIR = temp_dir
        monkeypatch.setattr(cache_module, 'CACHE_DIR', temp_dir)

        yield temp_dir

        shutil.rmtree(temp_dir, ignore_errors=True)
        cache_module.CACHE_DIR = original_cache_dir

    def test_computes_and_saves(self, temp_cache_dir):
        """Should compute block derivatives and save to cache."""
        delta = mpf('1.5')
        spin = 0
        n_max = 3

        # Should not exist yet
        assert not cache_exists(float(delta), spin)

        # Compute and cache
        h = compute_and_cache(delta, spin, n_max=n_max)

        # Should exist now
        assert cache_exists(float(delta), spin)

        # Result should be valid
        assert len(h) > 0
        assert all(isinstance(v, mpf) for v in h.values())

    def test_loads_from_cache(self, temp_cache_dir):
        """Should load from cache if available."""
        delta = mpf('2.0')
        spin = 0
        n_max = 3

        # First call computes
        h1 = compute_and_cache(delta, spin, n_max=n_max)

        # Second call should load from cache
        h2 = compute_and_cache(delta, spin, n_max=n_max)

        # Results should match (use relative tolerance due to float64 round-trip)
        # Cache stores float64, so precision is limited to ~15 significant digits
        for key in h1:
            if abs(h1[key]) > 0:
                rel_err = abs((h1[key] - h2[key]) / h1[key])
                assert rel_err < mpf('1e-14'), f"Relative error {rel_err} at {key}"
            else:
                assert abs(h2[key]) < mpf('1e-40')


class TestGetOrCompute:
    """Tests for get_or_compute function."""

    @pytest.fixture
    def temp_cache_dir(self, monkeypatch):
        """Create a temporary cache directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        import ising_bootstrap.blocks.cache as cache_module
        original_cache_dir = cache_module.CACHE_DIR
        cache_module.CACHE_DIR = temp_dir
        monkeypatch.setattr(cache_module, 'CACHE_DIR', temp_dir)

        yield temp_dir

        shutil.rmtree(temp_dir, ignore_errors=True)
        cache_module.CACHE_DIR = original_cache_dir

    def test_returns_numpy_array(self, temp_cache_dir):
        """Should return numpy array."""
        delta = mpf('1.5')
        spin = 0
        n_max = 3

        vec = get_or_compute(delta, spin, n_max=n_max)

        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float64

    def test_correct_length(self, temp_cache_dir):
        """Should return correct number of elements."""
        delta = mpf('2.0')
        spin = 2
        n_max = 5

        vec = get_or_compute(delta, spin, n_max=n_max)

        from ising_bootstrap.spectrum.index_set import index_set_size
        assert len(vec) == index_set_size(n_max)


class TestCacheStats:
    """Tests for cache statistics."""

    def test_stats_keys(self):
        """Stats should include expected keys."""
        stats = cache_stats()

        assert 'num_files' in stats
        assert 'total_size_mb' in stats
        assert 'delta_range' in stats
        assert 'spin_range' in stats
        assert 'cache_dir' in stats

    def test_stats_types(self):
        """Stats should have correct types."""
        stats = cache_stats()

        assert isinstance(stats['num_files'], int)
        assert isinstance(stats['total_size_mb'], float)
        assert isinstance(stats['cache_dir'], str)
