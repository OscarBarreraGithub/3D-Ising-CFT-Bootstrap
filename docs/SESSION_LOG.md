# Implementation Session Log

This file documents completed implementation sessions with details for future sessions to understand what was done, what works, and what remains.

---

## Session 2026-02-03: Milestone 4 - Stage A Scan Implementation

**Date**: 2026-02-03
**Duration**: Full implementation session
**Test Status**: 277/277 passing (37.02s, 1 expected warning)

### Session Summary

Completed Milestone 4 (Stage A scan) implementing the Delta_epsilon upper bound scan as a function of Delta_sigma. The implementation includes binary search logic for gap exclusion, extended H array caching for performance, full constraint matrix assembly with row subsetting, and CSV output for stage handoff. All 27 new tests pass, bringing total test count from 250 to 277.

### Changes Made

**Files Created:**
- `src/ising_bootstrap/scans/stage_a.py` (560 lines)
  - `ScanConfig` dataclass with grid parameters and discretization tables
  - `binary_search_eps()` - Generic binary search decoupled from LP (8 unit tests)
  - `build_full_constraint_matrix()` - Builds A matrix + scalar/spinning masks
  - `find_eps_bound()` - Row-subsetting binary search using precomputed A matrix
  - `load_h_cache_from_disk()` - Loads extended H arrays from cache
  - `run_scan()` - Main loop: iterate Delta_sigma grid, binary search, CSV output
  - `run_precompute()` - Block precomputation mode for extended H arrays
  - `main()` - CLI with argparse for production runs
  - CSV utilities: `write_csv_header`, `append_result_to_csv`, `load_scan_results`

- `tests/test_scans/__init__.py` (1 line)

- `tests/test_scans/test_stage_a.py` (427 lines, 27 tests)
  - `TestBinarySearchLogic` (8 tests): Generic binary search algorithm
  - `TestGapFiltering` (5 tests): Spectrum gap filtering behavior
  - `TestCSVIO` (4 tests): CSV read/write round-trip
  - `TestScanConfig` (5 tests): Configuration dataclass and grid generation
  - `TestExtendedCache` (2 tests): Extended H array cache format
  - `TestStageAIntegration` (3 slow tests): End-to-end scan with coarse discretization

**Files Modified:**
- `src/ising_bootstrap/scans/__init__.py` (7 → 30 lines)
  - Added exports for public API: `ScanConfig`, `run_scan`, `run_precompute`, etc.

- `src/ising_bootstrap/blocks/cache.py` (406 → 613 lines, +207 lines)
  - `get_extended_cache_filename()` - Filename format for extended H arrays
  - `extended_cache_exists()` - Check if extended cache file exists
  - `save_extended_h_array()` - Save (22,11) array to disk as .npy
  - `load_extended_h_array()` - Load extended H array from disk
  - `precompute_extended_spectrum_blocks()` - Bulk precomputation for all unique (Delta,l) pairs
  - Updated `clear_cache()` to also delete `ext_*.npy` files

- `src/ising_bootstrap/blocks/__init__.py` (106 → 116 lines, +10 lines)
  - Added exports for extended cache functions

**Total Line Changes:**
- Source code: 4,384 → 4,974 lines (+590 lines)
- Test code: 2,792 → 3,220 lines (+428 lines)
- Tests: 250 → 277 (+27 tests)

### Test Results

**Full Test Suite**: 277 passed, 1 warning in 37.27s

The warning is expected and documented:
```
tests/test_lp/test_solver.py::TestBootstrapFeasibilityReduced::test_no_gap_pipeline_runs
  UserWarning: Skipped 1/219 operators due to block computation errors
  (e.g., 3F2 pole at unitarity bound).
```

This is the known spin-0 at exact unitarity bound issue (Delta = alpha = 0.5 hits 3F2 pole).

**New Tests (27 total)**:
- 24 fast tests (< 0.1s each): Binary search logic, gap filtering, CSV I/O, config
- 3 slow tests (~3-5s each): Integration tests with coarse discretization (n_max=2, step=0.1)

### Key Architectural Decisions

1. **Binary Search Bug Fix**:
   - The pseudocode in `docs/TODO.md` had the binary search direction reversed
   - **Corrected logic**: Larger gap → fewer constraints → LP more likely feasible (excluded)
   - **Correct**: `if excluded: hi = mid` (gap too large, inconsistent with crossing)
   - **Incorrect**: `if is_feasible: lo = mid` (would maximize excluded region)

2. **Extended H Array Cache**:
   - Block derivatives h_{m,n}(Delta,l) are Delta_sigma-independent
   - Precompute once for all unique (Delta,l) pairs in discretization (~57k pairs for full tables)
   - Cache separately as `ext_*.npy` files with (22,11) shape (includes even m derivatives)
   - Reuse for all 51 Delta_sigma grid points → huge speedup (avoids 1000+ redundant computations)

3. **Full Constraint Matrix Approach**:
   - Build A matrix (N_ops × 66) once per Delta_sigma value
   - Binary search uses numpy boolean masking to select scalar/spinning row subsets
   - Much faster than rebuilding matrix each iteration

4. **Decoupled Binary Search**:
   - `binary_search_eps()` is a pure function taking a predicate
   - Independently testable without LP dependency
   - Makes testing robust and fast

### Nuances & Issues Discovered

1. **Binary Search Direction**:
   - Initial pseudocode had intuition backwards
   - Key insight: Removing scalars (larger gap) RELAXES constraints → makes LP easier to satisfy
   - Feasible LP means gap is inconsistent with physics → too large → lower `hi`

2. **Extended Cache Format**:
   - Existing cache stores 66-element vectors (odd m only, for crossing equation)
   - Extended cache stores full (22,11) arrays (both odd and even m, for Leibniz rule)
   - Separate file format (`ext_*.npy` vs `d*.npz`) avoids confusion

3. **Coarse Discretization in Tests**:
   - Integration tests use custom tables: step=0.1 (scalars), step=0.2 (spinning)
   - ~219 operators total vs ~201k for production
   - Keeps test runtime under 10s while exercising full pipeline
   - Results are qualitatively correct but not production-quality bounds

4. **Memory Usage**:
   - Full A matrix: (N_ops × 66) float64 → ~100 MB for 201k operators
   - Manageable on modern hardware
   - Extended H cache: ~57k files × 2KB each ≈ 110 MB total

5. **Precomputation Time**:
   - Full spectrum (~57k unique (Delta,l) pairs) with n_max=10 takes several hours
   - Once cached, scans are fast (minutes to hours depending on grid density)
   - Recommended workflow: run precomputation once, then run scans

### Current State / What Works

**Verified Working**:
- Binary search correctly finds gap bounds (8 unit tests confirm logic)
- Gap filtering correctly removes scalars below threshold (5 tests)
- CSV output round-trips correctly (4 tests)
- Extended H array cache saves/loads correctly (2 tests)
- End-to-end scan runs with coarse discretization (3 integration tests)
- All 277 tests passing

**Performance**:
- Single Delta_sigma point with coarse discretization (~219 ops): ~3-5s
- Binary search typically converges in 15-20 iterations (tolerance 1e-4)
- Full matrix build: ~0.5s per Delta_sigma for coarse discretization
- Cache loading: negligible once files are in OS page cache

**API Stability**:
- Public API exported from `src/ising_bootstrap/scans/` is stable
- CSV format is fixed: `delta_sigma,delta_eps_max`
- Extended cache format is fixed: `.npy` files with (22,11) shape

### Remaining Work / Next Steps

**Milestone 5: Stage B Scan** (not started):
1. Load Delta_epsilon_max(Delta_sigma) curve from Stage A CSV
2. For each Delta_sigma, fix Delta_epsilon gap and binary search Delta_epsilon'
3. Implement two-gap spectrum filtering (gap at Delta_epsilon and at Delta_epsilon')
4. Output CSV: `delta_sigma,delta_eps,delta_eps_prime_max`
5. This will be the final data for Figure 6

**Milestone 6: Plotting & Validation** (not started):
1. Load Stage B CSV
2. Plot Delta_epsilon' vs Delta_sigma (reproduce Fig. 6)
3. Validate sharp spike at Delta_sigma ~ 0.5182 with Delta_epsilon' ~ 3.84
4. Export PDF/PNG

**Production Run Requirements**:
1. Run extended H array precomputation with full discretization (Tables T1-T5)
   ```bash
   python -m ising_bootstrap.scans.stage_a --precompute --output data/cache_stats.txt
   ```
   Expected time: several hours
   Expected disk usage: ~110 MB for extended cache

2. Run Stage A scan with fine grid
   ```bash
   python -m ising_bootstrap.scans.stage_a \
       --sigma-min 0.50 --sigma-max 0.60 --sigma-step 0.002 \
       --output data/eps_bound.csv --verbose
   ```
   Expected time: 2-4 hours (51 grid points)
   Expected result: Delta_epsilon_max ~ 1.41 at Delta_sigma ~ 0.5182

### Commands to Verify Current State

**Run all tests**:
```bash
cd "/Users/oscar/Documents/Research_Code/Schwartz/3D Ising CFT Bootstrap"
pytest tests/ -v
```
Expected: 277 passed, 1 warning, ~37s

**Run only new Stage A tests**:
```bash
pytest tests/test_scans/test_stage_a.py -v
```
Expected: 27 passed, ~10s

**Run fast tests only** (skip slow integration tests):
```bash
pytest tests/test_scans/test_stage_a.py -v -m "not slow"
```
Expected: 24 passed, ~0.5s

**Check extended cache functionality**:
```python
from ising_bootstrap.blocks.cache import (
    save_extended_h_array, load_extended_h_array, extended_cache_exists
)
import numpy as np
from mpmath import mpf

# Test save/load
test_array = np.random.rand(22, 11)
save_extended_h_array(test_array, mpf("1.5"), 2, overwrite=True)
assert extended_cache_exists(mpf("1.5"), 2)
loaded = load_extended_h_array(mpf("1.5"), 2)
assert np.allclose(test_array, loaded)
```

### Important Context for Future Sessions

1. **The binary search direction is critical**: The corrected logic is now in `stage_a.py` and tested. Do NOT revert to the old pseudocode in `docs/TODO.md` (though it's now marked as corrected).

2. **Extended cache is separate from standard cache**:
   - Standard cache: `d*.npz` files with 66-element vectors (odd m only)
   - Extended cache: `ext_*.npy` files with (22,11) arrays (all m)
   - Do not mix them up

3. **Integration tests use coarse discretization**: Do not expect production-quality bounds from `test_stage_a.py` integration tests. They verify the pipeline works, not the physics.

4. **LP tolerance matters**: The default `LP_TOLERANCE = 1e-9` in `config.py` works well. Tighter tolerance can cause solver issues; looser tolerance can miss bounds.

5. **Row/column scaling is essential**: The LP constraint matrix has entries spanning many orders of magnitude. Geometric mean scaling (3 iterations) brings row maxima close to 1, dramatically improving HiGHS solver conditioning. Do not skip scaling.

6. **CLAUDE.md constraints still apply**:
   - mpmath precision (50+ digits)
   - Dolan-Osborn normalization
   - Table 2 discretization must be followed exactly
   - Caching to `data/cached_blocks/`

### Files Changed (Summary for Git Commit)

**New files**:
- `src/ising_bootstrap/scans/stage_a.py`
- `tests/test_scans/__init__.py`
- `tests/test_scans/test_stage_a.py`

**Modified files**:
- `src/ising_bootstrap/scans/__init__.py`
- `src/ising_bootstrap/blocks/cache.py`
- `src/ising_bootstrap/blocks/__init__.py`
- `docs/PROGRESS.md`
- `docs/TODO.md`

**Suggested commit message**:
```
Add Milestone 4: Stage A scan with extended H array cache

Implement Delta_epsilon upper bound scan as function of Delta_sigma:
- Binary search for gap exclusion with corrected logic
- Extended H array cache (22,11) for Delta_sigma-independent blocks
- Full constraint matrix assembly with row subsetting
- CSV output for stage handoff
- 27 new tests (24 fast + 3 slow integration)

All 277 tests passing (37s).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

---

## Session Archive

Older sessions will be added above as the project progresses.
