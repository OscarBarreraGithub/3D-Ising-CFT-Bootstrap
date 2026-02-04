# Implementation TODO: Reproducing Figure 6 of arXiv:1203.6064

## Overview

This document tracks the implementation of a complete pipeline to reproduce Figure 6 from "Solving the 3D Ising Model with the Conformal Bootstrap" (El-Showk et al., 2012).

**Target**: Upper bound on Δε' vs Δσ with n_max=10, matching the paper's Fig. 6.

**Key Validation Points**:
- At Ising point (Δσ ≈ 0.5182): Δε' ≈ 3.84
- Sharp spike/rapid growth just below Ising Δσ
- Qualitative match to paper's figure

---

## Milestone 0: Repository Scaffolding & Dependencies

### Tasks
- [ ] Create `pyproject.toml` with dependencies:
  - numpy, scipy (LP solver via HiGHS)
  - mpmath (extended-precision hypergeometric)
  - matplotlib (plotting)
  - pytest (testing)
- [ ] Create package structure `src/ising_bootstrap/` with subpackages:
  - `blocks/` - conformal block computation
  - `spectrum/` - discretization and index sets
  - `lp/` - linear programming
  - `scans/` - Stage A and B scanning
  - `plot/` - figure generation
- [ ] Create `src/ising_bootstrap/config.py` with constants:
  ```python
  D = 3           # Spacetime dimension
  N_MAX = 10      # Derivative truncation parameter
  ALPHA = D/2 - 1 # = 0.5 for D=3
  Z_POINT = 0.5   # Crossing-symmetric point z=z̄=1/2
  ```
- [ ] Create `docs/implementation_choices.md` (see separate file)
- [ ] Verify: `pip install -e .` succeeds
- [ ] Verify: `pytest` runs

### Files to Create
```
pyproject.toml
src/ising_bootstrap/__init__.py
src/ising_bootstrap/config.py
src/ising_bootstrap/blocks/__init__.py
src/ising_bootstrap/spectrum/__init__.py
src/ising_bootstrap/lp/__init__.py
src/ising_bootstrap/scans/__init__.py
src/ising_bootstrap/plot/__init__.py
```

### Acceptance Criteria
- [ ] `pip install -e .` completes without error
- [ ] `python -c "from ising_bootstrap import config; print(config.N_MAX)"` prints `10`
- [ ] `pytest --collect-only` shows test collection works

---

## Milestone 1: Conformal Block Evaluation & Derivative Engine

### Tasks

#### 1.1 Diagonal Blocks (z=z̄)
- [ ] Implement spin-0 block `G_{Δ,0}(z)` using mpmath ₃F₂ (Eq. 4.10):
  ```
  G_{Δ,0}(z) = (z²/(1-z))^{Δ/2} × ₃F₂(Δ/2, Δ/2, Δ/2-α; (Δ+1)/2, Δ-α; z²/(4(z-1)))
  ```
  where α = 1/2 for D=3

- [ ] Implement spin-1 block `G_{Δ,1}(z)` (Eq. 4.11):
  ```
  G_{Δ,1}(z) = (2-z)/(2z) × (z²/(1-z))^{(Δ+1)/2} × ₃F₂(...)
  ```

- [ ] Implement z-derivatives using ₃F₂ differential equation (Eq. 4.12):
  ```
  (xD̂_{a₁}D̂_{a₂}D̂_{a₃} - D̂₀D̂_{b₁-1}D̂_{b₂-1}) ₃F₂ = 0
  ```
  where D̂_c = x∂_x + c. This gives recursion for 3rd+ order derivatives.

#### 1.2 Higher Spin Recursion
- [ ] Implement spin recursion (Eq. 4.9) for l ≥ 2 at z=z̄:
  ```
  (l+D-3)(2Δ+2-D) G_{Δ,l}(z) =
      (D-2)(Δ+l-1) G_{Δ,l-2}(z)
    + (2-z)/(2z) × (2l+D-4)(Δ-D+2) G_{Δ+1,l-1}(z)
    - [coefficient] × G_{Δ+2,l-2}(z)
  ```
  Note: At z=z̄, this is non-derivative (the F₂ term vanishes).

#### 1.3 Coordinate Transformation
- [ ] Implement transformation from z to (a,b) coordinates:
  ```
  z = (a + √b)/2,  z̄ = (a - √b)/2
  ```
  At crossing-symmetric point: a=1, b=0 → z=z̄=1/2

- [ ] Implement chain rule for ∂_a derivatives in terms of ∂_z:
  ```
  ∂_a = (1/2)(∂_z + ∂_z̄)  →  at z=z̄: ∂_a = ∂_z
  ```

#### 1.4 Transverse Derivatives
- [ ] Implement Casimir recursion for b-derivatives (Eq. C.1):
  ```
  2(D+2n-3) h_{m,n} =
      2m(D+2n-3)[-h_{m-1,n} + (m-1)h_{m-2,n} + (m-1)(m-2)h_{m-3,n}]
    - h_{m+2,n-1} + (D-m-4n+4) h_{m+1,n-1}
    + [2C_{Δ,l} + ...] h_{m,n-1}
    + m[...] h_{m-1,n-1}
    + (n-1)[h_{m+2,n-2} - (D-3m-4n+4) h_{m+1,n-2}]
  ```
  where h_{m,n} = ∂_a^m ∂_b^n G|_{a=1,b=0} and C_{Δ,l} = Δ(Δ-D) + l(l+D-2)

#### 1.5 Caching
- [ ] Implement disk-based cache for derivative tensors
- [ ] Cache key: (Δ, l, precision) → h_{m,n} array
- [ ] Cache location: `data/cached_blocks/`

### Files to Create
```
src/ising_bootstrap/blocks/diagonal_blocks.py
src/ising_bootstrap/blocks/spin_recursion.py
src/ising_bootstrap/blocks/transverse_derivs.py
src/ising_bootstrap/blocks/cache.py
```

### Acceptance Criteria
- [ ] Unit test: `G_{1.5,0}(0.5)` ≈ known value (compute independently)
- [ ] Unit test: Spin recursion gives `G_{Δ,2}` consistent with direct formula
- [ ] Unit test: Casimir recursion satisfies consistency check
- [ ] Performance: All derivatives for one (Δ,l) computed in < 1 second

---

## Milestone 2: Spectrum Discretization & Index Set

### Tasks

#### 2.1 Index Set Generation
- [ ] Implement (m,n) index set with constraints:
  - m ≥ 1, m odd (only odd a-derivatives contribute due to antisymmetry)
  - n ≥ 0
  - m + 2n ≤ 2×n_max + 1 = 21 (for n_max=10)

- [ ] Verify count: should be exactly 66 elements
  ```
  m=1:  n=0..10  → 11 terms
  m=3:  n=0..9   → 10 terms
  m=5:  n=0..8   → 9 terms
  m=7:  n=0..7   → 8 terms
  m=9:  n=0..6   → 7 terms
  m=11: n=0..5   → 6 terms
  m=13: n=0..4   → 5 terms
  m=15: n=0..3   → 4 terms
  m=17: n=0..2   → 3 terms
  m=19: n=0..1   → 2 terms
  m=21: n=0      → 1 term
  Total: 11+10+9+8+7+6+5+4+3+2+1 = 66
  ```

#### 2.2 Table 2 Discretization
- [ ] Implement exact Table 2 from paper:

| Table | δ (step in Δ) | Δ_max | L_max | Description |
|-------|---------------|-------|-------|-------------|
| T1 | 2×10⁻⁵ | 3 | 0 | High-resolution scalars |
| T2 | 5×10⁻⁴ | 8 | 6 | Low-spin detail |
| T3 | 2×10⁻³ | 22 | 20 | Mid-range coverage |
| T4 | 0.02 | 100 | 50 | Intermediate asymptotics |
| T5 | 1 | 500 | 100 | Far asymptotics |

- [ ] For each table, sample dimensions from:
  ```
  Δ_min(l) = l + 1 - (1/2)δ_{l,0}   (unitarity bound)
  ```
  to:
  ```
  Δ_max^table + 2(L_max^table - l)  (upper limit shifts with spin)
  ```
  in steps of δ.

- [ ] Even spins only: l = 0, 2, 4, ... (Bose symmetry for identical scalars)

#### 2.3 Union of Tables
- [ ] Combine all (Δ,l) points from T1-T5 into single constraint set
- [ ] Remove duplicates (if any exact overlaps)
- [ ] Implement "reduced discretization" option for fast testing (T1-T2 only)

#### 2.4 Unitarity Bounds
- [ ] Implement unitarity check:
  - Scalars (l=0): Δ ≥ 1/2
  - Spinning (l≥1): Δ ≥ l + 1

### Files to Create
```
src/ising_bootstrap/spectrum/index_set.py
src/ising_bootstrap/spectrum/discretization.py
src/ising_bootstrap/spectrum/unitarity.py
```

### Acceptance Criteria
- [ ] Unit test: `len(generate_index_set(n_max=10))` == 66
- [ ] Unit test: (1,0), (1,10), (3,9), (21,0) in index set
- [ ] Unit test: (2,0), (22,0), (1,11) NOT in index set
- [ ] Unit test: T1 has correct number of scalar points
- [ ] Unit test: All generated (Δ,l) satisfy unitarity bounds
- [ ] Unit test: Only even l values appear

---

## Milestone 3: LP Builder & Solver Wrapper -- DONE

### Tasks

#### 3.1 Crossing Function Derivatives
- [x] Implement F^{Δσ}_{Δ,l} computation via Leibniz rule in `lp/crossing.py`
  - Key symmetry: V^{j,k} = (-1)^j U^{j,k} since v(a,b) = u(2-a,b)
  - Formula: F^{m,n} = 2 Σ C(m,j)C(n,k)(-1)^j U^{j,k} h_{m-j,n-k}
- [x] Compute prefactor table U^{j,k}(Δσ) via stable Taylor recursion
- [x] Handle extended index set (132 pairs, including even m) for Leibniz rule
- [x] Coordinate Jacobians: u = (a²-b)/4, v = ((2-a)²-b)/4

#### 3.2 Identity Term
- [x] Implement F_id^{m,n} = -2 U^{m,n} for m odd (analytical formula)
- [x] Cross-validated against mpmath numerical differentiation

#### 3.3 Constraint Matrix Assembly
- [x] Build A (N_operators × 66) and f_id (66,) in `lp/constraint_matrix.py`
- [x] Build from cache variant (`build_constraint_matrix_from_cache`)
- [x] Graceful error handling for 3F2 pole at spin-0 unitarity bound

#### 3.4 LP Solver Wrapper
- [x] Feasibility via scipy.optimize.linprog (HiGHS) in `lp/solver.py`
- [x] Result interpretation: status 0 = excluded, status 2 = allowed
- [x] Geometric mean row/column scaling (3 iterations)
- [x] End-to-end `solve_bootstrap()` function

### Files Created
```
src/ising_bootstrap/lp/__init__.py        (61 lines, public API)
src/ising_bootstrap/lp/crossing.py        (331 lines, Eq. 2.6 + Leibniz)
src/ising_bootstrap/lp/constraint_matrix.py (218 lines, matrix assembly)
src/ising_bootstrap/lp/solver.py          (302 lines, LP feasibility)
tests/test_lp/__init__.py
tests/test_lp/test_crossing.py            (396 lines, 36 tests)
tests/test_lp/test_solver.py              (354 lines, 21 tests)
```

### Acceptance Criteria
- [x] Unit test: Identity derivatives match analytical formulas (F_id^{1,0}=-1 at Δσ=0.5)
- [x] Unit test: F_{Δ,l} is antisymmetric under u↔v (odd m only)
- [x] Unit test: LP returns feasible (excluded) for synthetic feasible problem
- [x] Unit test: LP returns infeasible (allowed) for synthetic infeasible problem
- [x] Integration: End-to-end pipeline with coarse spectrum at Δσ=0.5182, gap Δε=1.41

**Note on "unconstrained = allowed" test**: At low n_max with coarse discretization,
the LP can spuriously exclude unconstrained spectra (the grid misses critical operators).
The physical assertion requires n_max ≈ 10 + fine Table 2 grids (production scans).

---

## Milestone 4: Stage A Scan (Δε Bound) -- DONE (2026-02-03)

### Tasks

#### 4.1 Binary Search Implementation
- [x] Implement binary search for Δε exclusion boundary
- [x] **CORRECTION**: Fixed binary search direction in pseudocode below
  - **Original (incorrect)**: `if is_feasible: lo = mid`
  - **Correct**: `if excluded: hi = mid`
  - **Reasoning**: Larger gap removes more scalars → makes LP easier to satisfy → more likely excluded
  ```python
  # CORRECTED VERSION:
  def find_eps_bound(delta_sigma, tol=1e-4, max_iter=50):
      lo = 0.5   # Unitarity bound for scalars
      hi = 2.5   # Generous upper bound

      for _ in range(max_iter):
          if hi - lo < tol:
              break
          mid = (lo + hi) / 2

          # Apply gap assumption: no scalar with Δ < mid
          spectrum = build_spectrum_with_scalar_gap(gap_min=mid)

          excluded = is_excluded(delta_sigma, spectrum)  # LP feasible
          if excluded:
              # Gap is too large (spectrum inconsistent) → lower hi
              hi = mid
          else:
              # Gap is consistent (allowed) → raise lo
              lo = mid

      return lo  # Δε_max(Δσ)
  ```

#### 4.2 Gap Implementation
- [x] Implement scalar gap constraint in `build_spectrum_with_scalar_gap`
  - Include identity (Δ=0, l=0) with F_id term (already in crossing equation)
  - Exclude scalars with Δ < Δε_trial from positivity constraints
  - Include scalars with Δ ≥ Δε_trial
  - Include all spinning operators per unitarity

#### 4.3 Scan Loop
- [x] Implement Δσ grid scan in `run_scan()`
  ```python
  sigma_grid = np.arange(0.50, 0.60 + step, step)
  for delta_sigma in sigma_grid:
      eps_max = find_eps_bound(delta_sigma)
      results.append((delta_sigma, eps_max))
  ```

- [x] Add progress logging (verbose mode)
- [x] Save intermediate results after each point

#### 4.4 Output
- [x] Save to CSV: `data/eps_bound.csv`
  ```
  delta_sigma,delta_eps_max
  0.500,1.234
  0.502,1.238
  ...
  ```

#### 4.5 Extended H Array Cache (Added)
- [x] Implement `save_extended_h_array` / `load_extended_h_array` in `blocks/cache.py`
- [x] Implement `precompute_extended_spectrum_blocks` for bulk precomputation
- [x] Extend `clear_cache()` to delete `ext_*.npy` files

### Files Created
```
src/ising_bootstrap/scans/stage_a.py       (560 lines)
tests/test_scans/__init__.py                (1 line)
tests/test_scans/test_stage_a.py            (427 lines, 27 tests)
```

### Files Modified
```
src/ising_bootstrap/scans/__init__.py       (30 lines, was 7-line stub)
src/ising_bootstrap/blocks/cache.py         (+207 lines, now 613 total)
src/ising_bootstrap/blocks/__init__.py      (+10 lines, now 116 total)
```

### Acceptance Criteria
- [x] At Δσ ≈ 0.518: Δε_max in plausible range (integration test verifies this)
- [x] Binary search logic correctly implemented and tested (8 unit tests)
- [x] CSV output is valid and readable (4 CSV I/O tests)
- [x] Can run with `--reduced` flag for fast testing (config test)
- [x] 27 tests passing (24 fast + 3 slow integration tests with coarse discretization)

### Implementation Notes

1. **Binary search bug fix**: The pseudocode above (lines 271-290) originally had the
   binary search direction reversed. The correct logic is:
   - Larger gap → fewer scalar constraints → LP more likely to find feasible functional → excluded
   - If excluded: gap is inconsistent with crossing → too large → `hi = mid`
   - If allowed: gap is consistent → can try larger → `lo = mid`

2. **Extended H array cache**: Block derivatives h_{m,n}(Δ,l) are Δσ-independent.
   Precompute once for all unique (Δ,l) pairs and cache as `ext_*.npy` files.
   Reuse for all 51 Δσ grid points (huge speedup).

3. **Full matrix approach**: Build constraint matrix A once per Δσ, then use boolean
   masks to select row subsets per binary search iteration. Much faster than rebuilding.

4. **Test discretization**: Integration tests use coarse tables (step 0.1/0.2) to keep
   runtime under 10s. Production scans use fine Table 2 discretization.

---

## Milestone 5: Stage B Scan (Δε' Bound) -- DONE (2026-02-03)

### Tasks

#### 5.1 Load Stage A Results
- [x] Read Stage A CSV via `load_eps_bound_map()` (rounded keys for robust matching)
- [x] Use exact grid match (no interpolation needed — same sigma grid)

#### 5.2 Two-Gap Implementation
- [x] Implement two-gap row masking in `find_eps_prime_bound()`:
  - Exclude scalars with Δ < Δε (below first gap)
  - Exclude scalars with Δε ≤ Δ < Δε' (between first and second gap)
  - Include scalars with Δ ≥ Δε' (above second gap)
  - Include all spinning operators unconditionally

#### 5.3 Binary Search for Δε'
- [x] Reuse `binary_search_eps()` from Stage A with two-gap `is_excluded` predicate
- [x] Search range: lo = Δε, hi = 6.0

#### 5.4 Scan Loop
- [x] For each Δσ in grid:
  1. Look up Δε = Δε_max(Δσ) from Stage A CSV
  2. Build full constraint matrix once per Δσ
  3. Binary search for Δε'_max using two-gap masking
  4. Write CSV row immediately (crash recovery)
- [x] Skip sigma points without Stage A data (with warning)

#### 5.5 Output
- [x] Save to CSV: `data/epsprime_bound.csv`
  ```
  delta_sigma,delta_eps,delta_eps_prime_max
  0.500000,1.234000,3.456000
  ...
  ```

#### 5.6 CLI
- [x] `python -m ising_bootstrap.scans.stage_b --eps-bound <path> [options]`
- [x] `--precompute-only` mode (delegates to Stage A precomputation)

### Files Created
```
src/ising_bootstrap/scans/stage_b.py       (456 lines)
tests/test_scans/test_stage_b.py           (510 lines, 25 tests)
```

### Files Modified
```
src/ising_bootstrap/scans/__init__.py      (42 lines, was 30 lines)
```

### Acceptance Criteria
- [x] Two-gap filtering logic correct (6 unit tests)
- [x] CSV 3-column round-trip works (5 CSV I/O tests)
- [x] Config defaults match config.py (6 config tests)
- [x] Stage A result loading works (3 loading tests)
- [x] Validation errors raised correctly (2 validation tests)
- [x] Integration: single-point and 3-point scans complete with coarse tables (3 slow tests)
- [ ] At Δσ ≈ 0.5182: Δε'_max ≈ 3.84 (requires production run with full discretization)
- [ ] Sharp spike visible just below Ising point (requires production run)
- [ ] Curve qualitatively matches Fig. 6 (requires production run + Milestone 6)

---

## Milestone 6: Plotting & Validation -- DONE (2026-02-04)

### Tasks

#### 6.1 Figure Generation
- [x] Create matplotlib figure matching Fig. 6 in `plot/fig6.py` (~207 lines)
  - `plot_fig6(data, output, dpi, show) -> Figure`
  - Plots bound curve, fills allowed region, marks Ising point
  - Uses Agg backend by default for headless cluster compatibility

#### 6.2 Sanity Check Output
- [x] Implement `print_sanity_check(data) -> None`
  - Prints validation summary to stdout (nearest Ising point values, qualitative checks)
  - Controllable via `--no-sanity-check` CLI flag

#### 6.3 Save Output
- [x] Save figure as PNG (configurable DPI, default 300)
- [x] Auto-save PDF sibling alongside PNG for publication quality

#### 6.4 CLI & Entry Point
- [x] CLI: `python -m ising_bootstrap.plot.fig6 --data --output --dpi --show --no-sanity-check`
- [x] Entry point `ising-plot` registered in pyproject.toml

#### 6.5 SLURM Infrastructure
- [x] Created `jobs/stage_b.slurm` -- SLURM array job for Stage B scan (mirrors stage_a.slurm)
- [x] Created `jobs/merge_stage_b.sh` -- merge script for Stage B per-task CSVs
- [x] Fixed `jobs/precompute.slurm` -- time limit 6h to 24h (precompute needs ~18h)

#### 6.6 Package Exports
- [x] Updated `plot/__init__.py` to export `plot_fig6` and `print_sanity_check`

### Files Created
```
src/ising_bootstrap/plot/fig6.py        (~207 lines, figure generation + CLI)
jobs/stage_b.slurm                      (SLURM array job for Stage B)
jobs/merge_stage_b.sh                   (merge script for Stage B CSVs)
```

### Files Modified
```
src/ising_bootstrap/plot/__init__.py    (exports plot_fig6, print_sanity_check)
jobs/precompute.slurm                   (time limit 6h -> 24h)
```

### Acceptance Criteria
- [x] Imports work (`from ising_bootstrap.plot import plot_fig6`)
- [x] CLI help works (`python -m ising_bootstrap.plot.fig6 --help`)
- [x] Synthetic test plot generates PNG + PDF
- [x] All 302 existing tests still pass
- [ ] Figure visually matches paper's Fig. 6 (requires production data)
- [ ] Spike feature is clearly visible (requires production data)
- [ ] Ising line at correct position (requires production data)

---

## Testing Requirements

### Unit Tests (`tests/`)

#### test_index_set.py
- [ ] Test count is 66
- [ ] Test specific elements present: (1,0), (1,10), (21,0)
- [ ] Test specific elements absent: (2,0), (1,11), (23,0)
- [ ] Test all m are odd
- [ ] Test all satisfy m + 2n ≤ 21

#### test_discretization.py
- [ ] Test T1-T5 table parameters match paper
- [ ] Test scalar count for T1
- [ ] Test spin range for each table
- [ ] Test unitarity bounds satisfied
- [ ] Test union has no invalid entries

#### test_identity_derivs.py
- [ ] Test ∂_a(v^{Δσ} - u^{Δσ})|_{a=1,b=0} analytically
- [ ] Test several (m,n) combinations
- [ ] Test antisymmetry property

#### test_blocks.py
- [ ] Test G_{Δ,0}(1/2) for specific Δ values
- [ ] Test spin recursion consistency
- [ ] Test transverse derivative recursion

### Integration Test (`tests/test_integration.py`)
- [ ] Mark as `@pytest.mark.slow`
- [ ] Run Stage A + B on coarse grid: Δσ ∈ {0.515, 0.518, 0.521}
- [ ] Use reduced discretization (T1-T2 only)
- [ ] Verify Δε' in range [3.0, 5.0] near Ising
- [ ] Verify pipeline completes without error

---

## Run Commands

### Development/Testing
```bash
# Install in development mode
pip install -e .

# Run all unit tests
pytest tests/ -v --ignore=tests/test_integration.py

# Run integration test (slow)
pytest tests/test_integration.py -v -s
```

### Production Run
```bash
# Stage A: Compute Δε bound (~2-4 hours)
python -m ising_bootstrap.scans.stage_a \
    --sigma-min 0.50 --sigma-max 0.60 --sigma-step 0.002 \
    --output data/eps_bound.csv

# Stage B: Compute Δε' bound (~4-8 hours)
python -m ising_bootstrap.scans.stage_b \
    --eps-bound data/eps_bound.csv \
    --output data/epsprime_bound.csv

# Generate figure
python -m ising_bootstrap.plot.fig6 \
    --data data/epsprime_bound.csv \
    --output figures/fig6_reproduction.png
```

### Quick Test Run
```bash
# Fast run with reduced discretization
python -m ising_bootstrap.scans.stage_a \
    --sigma-min 0.51 --sigma-max 0.53 --sigma-step 0.005 \
    --reduced \
    --output data/eps_bound_test.csv
```

---

## Dependencies

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "mpmath>=1.3",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
]
```

---

## Notes & Warnings

### From the Paper (Section 10)

1. **Block normalization mismatch** - Different libraries use different G_{Δ,l} prefactors. Symptoms: systematically shifted bounds, no spike near Ising. Mitigation: validate against known values.

2. **Derivative indexing** - Must use paper's rule m + 2n ≤ 2n_max + 1 exactly. Do NOT substitute "total derivative order."

3. **Discretization must match Table 2** - If T4/T5 are skipped, solver may exploit missing asymptotic constraints.

4. **LP conditioning** - Expect poor conditioning. Use row/column scaling, tune tolerances.

### Implementation Tips

1. **mpmath precision**: Use 50+ decimal digits for block computation, convert to float64 only when building LP matrix.

2. **Caching strategy**: Block derivatives h_{m,n}(Δ,l) don't depend on Δσ. Cache these to disk and reuse.

3. **Binary search tolerance**: 1e-4 for Δε, 1e-3 for Δε' is usually sufficient.

4. **Parallelization**: Δσ grid points are independent. Use multiprocessing for speedup.
