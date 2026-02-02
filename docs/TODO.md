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

## Milestone 3: LP Builder & Solver Wrapper

### Tasks

#### 3.1 Crossing Function Derivatives
- [ ] Implement F^{Δσ}_{Δ,l} computation:
  ```
  F^{Δσ}_{Δ,l}(u,v) = v^{Δσ} G_{Δ,l}(u,v) - u^{Δσ} G_{Δ,l}(v,u)
  ```

- [ ] At z=z̄=1/2: u = zz̄ = 1/4, v = (1-z)(1-z̄) = 1/4
  So at the crossing-symmetric point, u=v=1/4.

- [ ] Compute derivatives using Leibniz rule:
  ```
  ∂_a^m ∂_b^n [v^{Δσ} G(u,v)] = Σ (binomial coefficients) ×
                                  (∂^i v^{Δσ}) × (∂^j G)
  ```

- [ ] Handle coordinate Jacobians properly:
  - u = zz̄ = (a² - b)/4
  - v = (1-z)(1-z̄) = (1-a)² + b)/4 ... [verify this]

#### 3.2 Identity Term
- [ ] Implement F_id = v^{Δσ} - u^{Δσ}
- [ ] Compute analytical derivatives:
  ```
  ∂_a^m ∂_b^n (v^{Δσ} - u^{Δσ})|_{a=1,b=0}
  ```
  At a=1, b=0: u=v=1/4, so this requires careful limit handling.

#### 3.3 Constraint Matrix Assembly
- [ ] Build inequality matrix A_ub where:
  - Rows: one per (Δ,l) in discretized spectrum
  - Columns: one per (m,n) in index set (66 columns)
  - Entry: -v_{Δ,l}^{(m,n)} (negative because scipy uses A_ub @ x ≤ b_ub)

- [ ] Build equality constraint for normalization:
  ```
  A_eq[0,:] = v_id^{(m,n)} for all (m,n)
  b_eq[0] = 1
  ```

#### 3.4 LP Solver Wrapper
- [ ] Implement feasibility test using scipy.optimize.linprog:
  ```python
  result = linprog(
      c=np.zeros(n_vars),  # No objective
      A_ub=-A,             # -A @ λ ≤ 0  ⟺  A @ λ ≥ 0
      b_ub=np.zeros(n_constraints),
      A_eq=A_eq,
      b_eq=b_eq,
      method='highs-ds',
      options={'presolve': True, 'disp': False}
  )
  ```

- [ ] Interpret result:
  - `result.success` and status==0: Feasible (functional α exists → spectrum excluded)
  - Infeasible: Spectrum cannot be excluded (is allowed)

- [ ] Add constraint scaling for numerical stability

### Files to Create
```
src/ising_bootstrap/blocks/crossing_function.py
src/ising_bootstrap/lp/build_constraints.py
src/ising_bootstrap/lp/feasibility.py
```

### Acceptance Criteria
- [ ] Unit test: Identity derivatives match analytical formulas
- [ ] Unit test: F_{Δ,l} is antisymmetric under u↔v (odd in a)
- [ ] Unit test: LP returns feasible for unconstrained spectrum
- [ ] Unit test: LP returns infeasible for impossible constraints
- [ ] Integration: Single feasibility test for (Δσ=0.52, gap Δε=1.4) completes

---

## Milestone 4: Stage A Scan (Δε Bound)

### Tasks

#### 4.1 Binary Search Implementation
- [ ] Implement binary search for Δε exclusion boundary:
  ```python
  def find_eps_bound(delta_sigma, tol=1e-4, max_iter=50):
      lo = 0.5   # Unitarity bound for scalars
      hi = 2.5   # Generous upper bound

      for _ in range(max_iter):
          if hi - lo < tol:
              break
          mid = (lo + hi) / 2

          # Apply gap assumption: no scalar with Δ < mid
          spectrum = build_spectrum_with_scalar_gap(gap_min=mid)

          if is_feasible(delta_sigma, spectrum):
              # Can exclude this gap → push higher
              lo = mid
          else:
              # Cannot exclude → gap is too large
              hi = mid

      return lo  # Δε_max(Δσ)
  ```

#### 4.2 Gap Implementation
- [ ] Implement scalar gap constraint:
  - Include identity (Δ=0, l=0) with F_id term (already in crossing equation)
  - Exclude scalars with Δ < Δε_trial from positivity constraints
  - Include scalars with Δ ≥ Δε_trial
  - Include all spinning operators per unitarity

#### 4.3 Scan Loop
- [ ] Implement Δσ grid scan:
  ```python
  sigma_grid = np.arange(0.50, 0.60 + step, step)
  for delta_sigma in sigma_grid:
      eps_max = find_eps_bound(delta_sigma)
      results.append((delta_sigma, eps_max))
  ```

- [ ] Add progress bar/logging
- [ ] Save intermediate results

#### 4.4 Output
- [ ] Save to CSV: `data/eps_bound.csv`
  ```
  delta_sigma,delta_eps_max
  0.500,1.234
  0.502,1.238
  ...
  ```

### Files to Create
```
src/ising_bootstrap/scans/stage_a.py
```

### Acceptance Criteria
- [ ] At Δσ ≈ 0.518: Δε_max ≈ 1.41 (compare to Fig. 3 in paper)
- [ ] Curve is roughly monotonic/smooth
- [ ] CSV output is valid and readable
- [ ] Can run with `--reduced` flag for fast testing

---

## Milestone 5: Stage B Scan (Δε' Bound)

### Tasks

#### 5.1 Load Stage A Results
- [ ] Read `data/eps_bound.csv`
- [ ] Interpolate Δε_max(Δσ) if needed (or use exact grid match)

#### 5.2 Two-Gap Implementation
- [ ] Implement spectrum with two gaps:
  ```python
  def build_spectrum_stage_b(delta_eps, delta_eps_prime_trial):
      # Gap 1: No scalars below ε (except identity)
      # Gap 2: No scalars between ε and ε' trial
      # Include: scalars with Δ ≥ delta_eps_prime_trial
      # Include: all spinning operators per unitarity
  ```

#### 5.3 Binary Search for Δε'
- [ ] Implement binary search:
  ```python
  def find_eps_prime_bound(delta_sigma, delta_eps, tol=1e-3):
      lo = delta_eps + 0.01  # Just above ε
      hi = 6.0               # Generous upper bound

      for _ in range(max_iter):
          if hi - lo < tol:
              break
          mid = (lo + hi) / 2

          spectrum = build_spectrum_stage_b(delta_eps, mid)

          if is_feasible(delta_sigma, spectrum):
              # Can exclude → push higher
              lo = mid
          else:
              # Cannot exclude
              hi = mid

      return lo  # Δε'_max(Δσ)
  ```

#### 5.4 Scan Loop
- [ ] For each Δσ in grid:
  1. Look up Δε = Δε_max(Δσ) from Stage A
  2. Binary search for Δε'_max
  3. Record result

#### 5.5 Output
- [ ] Save to CSV: `data/epsprime_bound.csv`
  ```
  delta_sigma,delta_eps,delta_eps_prime_max
  0.500,1.234,2.567
  ...
  ```

### Files to Create
```
src/ising_bootstrap/scans/stage_b.py
```

### Acceptance Criteria
- [ ] At Δσ ≈ 0.5182: Δε'_max ≈ 3.84
- [ ] Sharp spike visible just below Ising point
- [ ] Curve qualitatively matches Fig. 6
- [ ] CSV output valid

---

## Milestone 6: Plotting & Validation

### Tasks

#### 6.1 Figure Generation
- [ ] Create matplotlib figure matching Fig. 6:
  ```python
  fig, ax = plt.subplots(figsize=(8, 6))

  # Plot bound curve
  ax.plot(sigma, eps_prime_max, 'b-', linewidth=1.5)

  # Fill allowed region
  ax.fill_between(sigma, y_min, eps_prime_max, alpha=0.3, color='blue')

  # Ising vertical line
  ax.axvline(x=0.5182, color='red', linewidth=2, label='Ising')

  # Labels
  ax.set_xlabel(r'$\Delta_\sigma$', fontsize=14)
  ax.set_ylabel(r'$\Delta_{\epsilon\'}$', fontsize=14)
  ax.set_xlim(0.50, 0.60)
  ax.set_ylim(2.0, 4.5)
  ```

#### 6.2 Sanity Check Output
- [ ] Print to stdout:
  ```
  === Sanity Check ===
  Near Ising point (Δσ ≈ 0.5182):
    Δε_max  = 1.41 (expected ~1.41)
    Δε'_max = 3.84 (expected ~3.84)

  Qualitative features:
    [✓] Spike present below Ising Δσ
    [✓] Bound increases for Δσ > 0.52
  ```

#### 6.3 Save Output
- [ ] Save figure: `figures/fig6_reproduction.png` (300 dpi)
- [ ] Also save PDF version for publication quality

### Files to Create
```
src/ising_bootstrap/plot/fig6.py
docs/RUN.md
```

### Acceptance Criteria
- [ ] Figure visually matches paper's Fig. 6
- [ ] Spike feature is clearly visible
- [ ] Ising line at correct position
- [ ] File saved successfully

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
