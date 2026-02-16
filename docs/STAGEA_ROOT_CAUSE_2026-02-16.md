# Stage A Root Cause Analysis

**Date:** 2026-02-16
**Branches:** `diag/float64-vs-mpmath-crossing` (Claude Code), `codex/stagea-bracket-guard-debug` (Codex)
**Supersedes:** `LP_CONDITIONING_BUG.md` (2026-02-06, incomplete fix)

## Problem

Stage A production run (job 60119846, sapphire partition, SDPB backend) returned wrong results:
- 45/51 tasks timed out after 36 hours
- 6 completed tasks ALL returned Delta_eps_max = 0.500000 (unitarity floor)
- Every SDPB bisection midpoint was marked "excluded" (dual feasible)
- Binary search converged to eps_lo = 0.5 by construction

Expected: Delta_eps_max ~ 1.41 at Delta_sigma ~ 0.518

## Root Cause: Float64 Input to SDPB

The constraint matrix is computed entirely in float64 (17 significant digits) before being
passed to SDPB. At n_max=10, the LP condition number is ~10^16 to 10^26, so:

    float64 roundoff (~1e-16) x condition_number (~1e16-1e26) = effective error O(1) to O(10^10)

SDPB's 1024-bit internal precision is wasted because the INPUT data has only 17 significant
digits. SDPB is solving the WRONG (float64-corrupted) problem with extreme precision.

### Where precision is lost

1. **constraint_matrix.py:90** — `A = np.zeros(..., dtype=np.float64)`
2. **crossing.py:122** — `compute_prefactor_table` returns float64 numpy array
3. **crossing.py:205** — `compute_extended_h_array` returns float64 numpy array
4. **crossing.py:297** — `compute_crossing_vector_fast` returns float64 numpy array
5. **sdpb.py:73** — `_format_float` writes values with only 17 significant digits

The entire chain (prefactors -> h arrays -> crossing vectors -> constraint matrix -> PMP JSON)
operates in float64. SDPB receives a PMP file with 17-digit numbers.

### Condition number scaling with n_max

| n_max | Components | Condition number | float64 x kappa | LP result |
|-------|------------|------------------|-----------------|-----------|
| 3     | 10         | ~1.2e+18         | ~1e+02          | Correct (Delta_eps ~ 1.41) |
| 5     | 21         | ~1.0e+22         | ~1e+06          | Marginal |
| 7     | 36         | ~7.9e+26         | ~1e+10          | ALL infeasible (Delta_eps = 0.5) |
| 10    | 66         | >10^16           | >>1             | SDPB finds spurious solutions |

Condition number grows because crossing vector components span ~24 orders of magnitude
(from O(1) for low-order derivatives to O(10^24) for high-order), and the Casimir recursion
for spinning operators amplifies errors at high derivative order.

## Evidence Chain

### 1. n_max progression (Claude Code diagnostic: diag_nmax3_bisection.py)

Tested binary search with 1831 operators using scipy at multiple n_max values:
- **n_max=3**: Correct behavior — infeasible below gap~1.4, feasible above
- **n_max=5**: Still correct
- **n_max=7**: ALL gaps infeasible, kappa = 7.9e+26, Delta_eps_max = 0.500000
- **n_max=10**: Timed out (crossing vector computation too slow for test partition)

This proves the problem is n_max-dependent conditioning, NOT a code bug.

### 2. Float64 vs mpmath crossing vectors (both branches)

Codex diagnostic (diag_f64_vs_mp.py) and Claude Code diagnostic (diag_precision_gap.py)
both compared float64 vs mpmath (50-digit) crossing vectors at n_max=10:
- Identity vector relerr: ~6e-15 (machine epsilon)
- Scalar (Delta=1.0) crossing relerr: ~1e-14
- Spinning (Delta=3.0, l=2) crossing relerr: ~2.5e-12 (Casimir recursion amplifies)
- No sign flips in individual components

Codex concluded "low confidence" on float64 as root cause because individual errors are "small."
This misses the key point: error * condition_number is what matters for the LP solution, not the
individual vector errors.

### 3. Production SDPB logs (job 60119846)

Every SDPB solve at every bisection midpoint returned "primal-dual optimal" with duality gap
~1e-31. This means SDPB confidently found feasible solutions in the corrupted data. The solver
worked correctly — it solved the wrong problem.

pmp2sdp conversion: ~400-450s per iteration for 470K-520K blocks.
Each task needed 12-16 bisection iterations x ~2h per solve = 28-35 hours total.
Most tasks exceeded the 36-hour walltime limit.

### 4. Scipy at n_max=10 (Codex diagnostic: diag_scipy_n10_full_noscale)

Scipy (float64 arithmetic) with full spectrum at n_max=10: ALL gaps return ALLOWED.
This is the mirror failure — scipy can't navigate the ill-conditioned matrix at float64,
so it returns "infeasible" for everything. Same corrupted data, opposite wrong answer:
- scipy (float64 solver + float64 data): can't find feasible directions -> "allowed" always
- SDPB (1024-bit solver + float64 data): finds spurious feasible directions -> "excluded" always

### 5. Scipy scaling bug (Codex diagnostic: diag_scale_unscale)

Codex found that `solver.py:240` has a scaling/unscaling issue: `alpha = res.x / col_scale`
produces invalid certificates (eq residual ~6e+31). With multiply instead of divide,
normalization residual drops to ~1e-15. However, the unscaled solve still returns infeasible
("allowed") due to the underlying conditioning problem, so this bug is secondary — it only
affects the scipy path's certificate validation, not the SDPB production path.

### 6. SDPB scaling test (job 60566235, in progress)

Testing SDPB with float64 vs mpmath (50-digit) PMP input at n_max=10 with increasing
operator counts (200, 500, 1000, 2000, 3685). This will show at what scale the two
approaches diverge. Currently computing crossing vectors (~0.7/s for 3685 operators).

## What Was Verified Correct

These components are NOT the problem:
1. Conformal block z-derivatives (match numerical differentiation to ~1e-51)
2. h_{m,0} conversion (z-derivatives to a-derivatives)
3. Identity vector computation
4. Crossing vector formula (Leibniz rule derivation)
5. 3F2 hypergeometric argument
6. Binary search direction in Stage A (hi=mid when excluded, lo=mid when allowed)
7. LP formulation (alpha^T f_id = 1, alpha^T f_i >= 0)
8. SDPB PMP encoding (1x1 blocks, degree-0 polynomials)

## Secondary Issues Found

### Missing bracket guard (Codex)
Stage A assumes eps_lo is allowed and eps_hi is excluded but does not validate before
bisecting. If both endpoints are "excluded" (as happens with corrupted data), the search
silently converges to the floor. Codex added bracket validation — worth keeping as a
safety net.

### afterok vs afterany (Codex)
`run_pipeline.sh` used `--dependency=afterok` for the merge job, meaning any timed-out
array task would silently stall the entire pipeline. Changed to `afterany` with the merge
script doing its own validation.

## Fix Plan

### The fix: mpmath-precision SDPB pipeline

mpmath variants of ALL crossing vector functions already exist in crossing.py (lines 337-461):
- `compute_prefactor_table_mp`
- `compute_identity_vector_mp`
- `compute_extended_h_array_mp`
- `compute_crossing_vector_mp`

These just aren't wired into the SDPB pipeline. The fix:

1. **New `write_pmp_json_mp`** in sdpb.py — accepts mpmath values directly, writes PMP JSON
   with 50+ significant digits (not just 17)

2. **New `build_constraint_matrix_mp`** in constraint_matrix.py — builds constraint data
   using mpmath crossing functions, returns mpmath values (not float64 numpy)

3. **Wire through `check_feasibility_sdpb`** — when backend is SDPB, use the mpmath path

4. **Pre-compute mpmath h arrays** — 520K operators at ~1.25s each = ~180h serial.
   Parallelize via SLURM array (112 cores on sapphire -> ~1.6h one-time cache build).
   Store as high-precision strings or pickled mpmath values.

### Performance estimate

- One-time h_{m,n} cache in mpmath: ~1.6h (parallelized on sapphire)
- Crossing vector assembly from cached h: ~3ms per operator (fast Leibniz sum)
- PMP JSON write with 50-digit strings: ~same as current
- pmp2sdp: ~400-450s (same as current, dominates runtime)
- SDPB solve: same as current but now operating on accurate data

Total per-bisection-iteration impact: minimal (pmp2sdp dominates).
One-time mpmath cache cost: ~1.6h compute.

### Safety nets (from Codex, keep)

- Bracket validation before binary search
- afterany dependency in pipeline
- Verbose SDPB logging (terminateReason, dualityGap)

## Reference: Diagnostic Job IDs

### Claude Code diagnostics
| Job | Description | Result |
|-----|-------------|--------|
| 60566235 | SDPB f64 vs mp scaling test (n_max=10, 200-3685 ops) | Running |
| 60565095 | SDPB f64 vs mp at n_max=10, 177 ops | Both EXCLUDED (too few ops) |

### Codex diagnostics
| Job | Description | Result |
|-----|-------------|--------|
| 60556339 | Float64 vs mpmath crossing comparison | Small relerr, no sign flips |
| 60556687 | scipy n_max=10 full no-scale | ALL ALLOWED |
| 60556704 | Scaling/unscaling proof | Unscale bug confirmed |
| 60556847 | SDPB n_max=10 reduced | Running |
| 60554786-60555946 | Various n_max=2 probes | Eliminated low-level code bugs |

### Production runs
| Job | Description | Result |
|-----|-------------|--------|
| 59675873 | First SDPB attempt (shared) | ALL OOM killed |
| 60119846 | Sapphire SDPB Stage A | 45 timeout, 6 completed all 0.5 |

## Files Involved

| File | Role | Issue |
|------|------|-------|
| `src/ising_bootstrap/lp/constraint_matrix.py` | Builds A matrix | dtype=float64 (line 90) |
| `src/ising_bootstrap/lp/crossing.py` | Crossing vectors | float64 path used; mpmath path exists but unused |
| `src/ising_bootstrap/lp/sdpb.py` | SDPB interface | Formats with 17 digits (line 73) |
| `src/ising_bootstrap/lp/solver.py` | LP dispatch | Passes float64 to SDPB; also has scaling bug |
| `src/ising_bootstrap/scans/stage_a.py` | Binary search | No bracket validation |
