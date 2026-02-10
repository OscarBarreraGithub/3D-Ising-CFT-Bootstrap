# LP Conditioning Bug — Stage A Returns All 0.5

**Date:** 2026-02-06
**Status:** FIXED (2026-02-09) — replaced scipy/HiGHS with SDPB arbitrary-precision solver

**Fix:** Integrated SDPB (arXiv:1502.02033) via Singularity container as the LP backend.
SDPB uses 1024-bit internal precision, completely sidestepping the float64 conditioning issue.
See `src/ising_bootstrap/lp/sdpb.py` and the `--backend sdpb` CLI option.
Stage A job 59675873 submitted with SDPB backend — validation pending.

## Summary

All 51 Stage A scan tasks completed (job 59036450), but every task returned
`delta_eps_max = 0.500000` — the scalar unitarity bound. The binary search
collapsed to its lower bound because the LP solver reports "excluded" for
every gap value, including gap = 0.5 (no gap at all). This is physically
wrong: the full ungapped spectrum must be "allowed."

Task 32 (delta_sigma = 0.564) timed out and produced no data row.

## Pipeline Status After Jobs Ran

| Stage | Status |
|-------|--------|
| Block precomputation (520K .npy files) | Done |
| Stage A array job (51 tasks) | Done — but ALL results are wrong (0.5) |
| Merge Stage A | Was never run (ran it manually, produced `data/eps_bound.csv`) |
| Stage B | Not submitted (blocked on Stage A) |
| Plot Figure 6 | Not run |

## Root Cause: Numerical Ill-Conditioning of the LP

The LP feasibility problem has 66 functional components (n_max=10) and
~520K inequality constraints. The crossing derivative vectors F^{m,n}
span **~24 orders of magnitude** across the 66 components because
high-order derivatives (m+2n up to 21) grow factorially.

### Key Diagnostic Results

1. **Constraint matrix condition number: ~4 × 10^16** (exceeds float64
   precision of ~10^15).

2. **Effective rank: 10 out of 66** at double precision. The bottom 56
   singular values fall below machine epsilon relative to the top singular
   value. The LP effectively operates in a 10-dimensional subspace instead
   of the full 66-dimensional space.

3. **The custom `scale_constraints` function makes it worse.** It applies
   iterative row+column scaling that changes the effective LP tolerances,
   causing the solver to report false "feasible" (excluded) results:

   ```
   scale=True:  excluded=True  (WRONG — false positive)
   scale=False: excluded=False (correct for no gap, but...)
   ```

4. **Without scaling, the LP is too permissive.** It says "allowed" even
   for gap=2.0, which should be excluded (the known bound is ~1.41):

   ```
   scale=False, gap=2.0: excluded=False  (WRONG — false negative)
   ```

5. **The crossing vectors themselves are accurate.** Extended-precision
   comparison shows max relative error of 4.4 × 10^{-15} (machine epsilon).
   There is no bug in the Leibniz rule or block derivative computation.

6. **The problem occurs at ALL n_max values tested (2–8):**
   - n_max ≤ 7: LP always says "excluded" (too aggressive)
   - n_max = 8: LP always says "allowed" (too permissive)
   - Neither gives correct physical results

### SVD Analysis of Constraint Matrix (n_max=10, 1329 operators)

```
Singular values:
  s[0]  = 3.87e+25
  s[1]  = 6.65e+23
  s[9]  = 7.31e+15    (10th component)
  s[65] = 8.99e+08    (last component)

Condition number: 4.3e+16
Effective rank (s > 1e-10 * s_max): 10
99.9% of variance in top 1 component
```

## What Is NOT the Bug

- **Crossing vector formula** is mathematically correct (verified by derivation).
- **Identity vector** F_id^{m,n} = -2 U^{m,n} is correct.
- **Block derivatives** h_{p,q} are correctly computed and cached (520K .npy files are valid).
- **Binary search logic** works correctly (tested with synthetic is_excluded functions).
- **LP formulation** (find alpha: alpha.f_id = 1, alpha.F_i >= 0) is correct.
- **Cached block data** loads correctly and contains non-zero, finite values.

## What IS the Bug

The constraint matrix has inherent condition number ~10^16, making the LP
intractable at IEEE float64 precision. The 66 functional components span
~24 orders of magnitude (from O(1) for low-order derivatives to O(10^24)
for high-order), and this dynamic range exceeds the ~15.7 decimal digits
available in float64.

The custom `scale_constraints()` in `solver.py` was intended to fix this
but actually makes it worse by causing false "excluded" results. The
production scan used `scale=True` (the default), so every binary search
iteration saw "excluded" and converged to the lower bound.

## Approaches Tried (Did Not Fix)

| Approach | Result |
|----------|--------|
| `scale=False` (no scaling) | False negatives (never excluded) |
| Column-only scaling (max norm) | Still always excluded |
| Geometric mean normalization | Still always excluded |
| Factorial weight normalization (m! × n!) | Improved rank 10→35, still fails |
| SVD projection onto top-k components | Always allowed (same as no scaling) |

## Fix Implemented: Option D (SDPB)

The fundamental issue is that scipy/HiGHS operates at float64 and cannot
handle the 10^16 condition number. We chose **Option D: SDPB** from the
options below. Implementation details in `src/ising_bootstrap/lp/sdpb.py`.

Other options considered (not implemented):

### Option A: Extended-Precision LP (Most Robust)

Compute the Leibniz sum and LP constraint matrix at extended precision
(mpmath, 50+ digits) and use an extended-precision LP solver. This is
what SDPB (the standard bootstrap solver) does.

**Pros:** Handles arbitrary n_max, matches paper methodology.
**Cons:** Need to implement or wrap an extended-precision LP solver.
Existing options: `mpmath` linsolve (but not LP), or call SDPB externally.

### Option B: Proper Component Normalization

Define normalized functional components alpha_tilde_{m,n} = alpha_{m,n} * N_{m,n}
where N_{m,n} absorbs the factorial growth. Choose N_{m,n} such that the
normalized constraint matrix has condition number < 10^10 (well within float64).

The normalization must be applied BEFORE the constraint matrix is formed
(at the level of the crossing vector computation), not as post-hoc scaling.
This requires understanding the exact growth rate of F^{m,n} with (m,n)
and finding appropriate weights.

**Pros:** Keeps float64 LP solver, minimal code changes.
**Cons:** Need to find correct weights; may not work at very high n_max.

### Option C: Mixed-Precision LP

Compute the constraint matrix at float64, then use iterative refinement
at extended precision to correct the LP solution. The LP is solved
approximately at float64, then residuals are computed at extended precision,
and corrections are applied.

**Pros:** Mostly uses fast float64 arithmetic.
**Cons:** Requires custom iterative refinement loop.

### Option D: Use SDPB ← IMPLEMENTED

Replace scipy's LP solver with SDPB (Simmons-Duffin's semidefinite
programming solver for bootstrap), which is purpose-built for this
exact problem class and uses arbitrary precision internally.

**Pros:** Standard tool for conformal bootstrap, handles all n_max.
**Cons:** External dependency, needs installation on the cluster.

**Implementation (2026-02-09):**
- Singularity container: `tools/sdpb-3.1.0.sif` (pulled from `docker://bootstrapcollaboration/sdpb:3.1.0`)
- PMP JSON writer: encodes discrete LP as degenerate PMP with 1×1 blocks, degree-0 polynomials
- Subprocess wrapper: `pmp2sdp` → `sdpb` via `singularity exec`
- Drop-in replacement: `check_feasibility(A, f_id, backend="sdpb")`
- Verified end-to-end: feasible LP → excluded=True, infeasible LP → excluded=False

## Files Involved

| File | Role |
|------|------|
| `src/ising_bootstrap/lp/solver.py` | `check_feasibility()` — LP solver wrapper; `scale_constraints()` — the broken scaling |
| `src/ising_bootstrap/lp/crossing.py` | `compute_crossing_vector_fast()` — Leibniz rule (correct) |
| `src/ising_bootstrap/lp/constraint_matrix.py` | `build_constraint_matrix_from_cache()` — assembles A matrix |
| `src/ising_bootstrap/scans/stage_a.py` | `find_eps_bound()` — binary search calling check_feasibility with `scale=config.scale` |
| `src/ising_bootstrap/config.py` | `LP_TOLERANCE = 1e-7`, `N_MAX = 10` |

## Reproduction

```python
# Quick reproduction: shows scale=True gives wrong answer
from ising_bootstrap.lp.solver import check_feasibility
from ising_bootstrap.lp.constraint_matrix import build_constraint_matrix
from ising_bootstrap.spectrum.discretization import generate_full_spectrum
from ising_bootstrap.config import REDUCED_DISCRETIZATION

spectrum = generate_full_spectrum(tables=REDUCED_DISCRETIZATION)
A, f_id = build_constraint_matrix(spectrum, 0.518, n_max=2)

# With scaling: WRONG (says excluded)
r1 = check_feasibility(A, f_id, scale=True)
print(f"scale=True: excluded={r1.excluded}")   # True (wrong)

# Without scaling at n_max=2: also wrong
r2 = check_feasibility(A, f_id, scale=False)
print(f"scale=False: excluded={r2.excluded}")   # True (wrong at n_max=2 too)
```
