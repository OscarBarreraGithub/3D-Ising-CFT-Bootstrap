# Mpmath-Precision SDPB Pipeline Fix

**Date:** 2026-02-16
**Branch:** `fix/mpmath-precision-sdpb`
**Status:** Planning
**Prereq:** `docs/STAGEA_ROOT_CAUSE_2026-02-16.md` (root cause analysis)

## Current Status

**What's broken:** The SDPB pipeline receives float64-precision data (17 significant digits)
despite using 1024-bit internal arithmetic. At n_max=10, the LP condition number exceeds 10^16,
so float64 roundoff gets amplified to O(1) or worse. SDPB solves the wrong problem correctly.

**What exists and works:**
- mpmath crossing vector functions in `crossing.py` (lines 337-461)
  - `compute_prefactor_table_mp(delta_sigma, n_max, dps=50)` -> dict of mpf
  - `compute_identity_vector_mp(delta_sigma, n_max, dps=50)` -> list of mpf
  - `compute_extended_h_array_mp(delta, spin, n_max, dps=50)` -> dict of mpf
  - `compute_crossing_vector_mp(H_mp, U_mp, index_set, comb_cache, max_order)` -> list of mpf
- These are tested and correct (verified against float64 to ~1e-12 to 1e-15 relerr)
- The float64 h cache (`data/cached_blocks/ext_cache_consolidated.npz`, 520K operators)
- SDPB container works (pmp2sdp + sdpb solve chain)

**What's missing:**
- PMP JSON writer that accepts mpmath values (not just float64)
- Constraint matrix builder using mpmath crossing vectors
- mpmath h cache (the float64 cache has only 17-digit h values)
- Stage A wired to use mpmath path when backend=SDPB

## Architecture

### Current pipeline (float64, broken at n_max=10)

```
h_cache (float64 .npz)
  -> build_constraint_matrix_from_cache()     [float64 A, f_id]
    -> find_eps_bound()                        [bisection loop]
      -> check_feasibility(A_sub, f_id, backend="sdpb")
        -> write_pmp_json(A, f_id)             [17-digit PMP JSON]
          -> pmp2sdp -> sdpb                   [1024-bit, wasted]
```

### Fixed pipeline (mpmath, 50-digit precision)

```
mp_h_cache (mpmath pickle, 50 digits)
  -> build_full_constraint_data_mp()           [mpf crossing vectors + float64 masks]
    -> find_eps_bound()                        [bisection loop]
      -> check_feasibility_sdpb_mp(cv_mp_sub, f_id_mp)
        -> write_pmp_json_mp(cv_mp, f_id_mp)   [50-digit PMP JSON]
          -> pmp2sdp -> sdpb                    [1024-bit, now meaningful]
```

### Key design decision: pre-compute all crossing vectors once per delta_sigma

In the bisection, delta_sigma is fixed. Only the gap changes, which filters which operators
are included. So crossing vectors are computed ONCE and reused across all bisection iterations.

For each delta_sigma:
1. Compute U_mp and f_id_mp (fast, ~1s)
2. Load mpmath h cache and assemble all 520K crossing vectors via Leibniz sum (~10ms each)
3. Store as list of mpf lists alongside float64 masks
4. Each bisection iteration: filter list by gap, write PMP JSON with 50-digit strings

This means the mpmath overhead happens once per delta_sigma, not per bisection iteration.

## Implementation Plan

### Phase 1: mpmath h cache build (one-time, ~1.5h on sapphire)

**Why needed:** The existing float64 h cache has only 17-digit values. We need 50-digit h_{m,n}
for the Leibniz sum to produce 50-digit crossing vectors.

**Bottleneck:** `compute_extended_h_array_mp(delta, spin, n_max=10, dps=50)` takes ~1s per
operator (3F2 evaluation + Casimir recursion in mpmath). For 520K operators: ~144h serial.

**Parallelization:** SLURM array job on sapphire (112 cores/node).
- 520K operators / 112 workers = ~4600 per worker
- ~4600s = ~77 min per worker
- Total walltime: ~1.5h with one 112-core node

**Storage:** Sharded pickle files, one per worker.
- Each operator: dict of (p,q) -> mpf, ~242 entries, ~15KB pickled
- Total: 520K × 15KB = ~7.5 GB uncompressed, ~2-3 GB compressed
- Consolidation script merges shards into single archive

**Files to create:**
- `scripts/precompute_mp_h_cache.py` — worker script
- `jobs/precompute_mp_h_cache.slurm` — SLURM array job
- `scripts/consolidate_mp_h_cache.py` — merge shards into one file

**Output:** `data/cached_blocks/mp_h_cache.pkl.gz` (or similar)

### Phase 2: Code changes (sdpb.py, constraint_matrix.py, stage_a.py)

#### 2a. `write_pmp_json_mp` in sdpb.py

New function that accepts mpmath crossing vectors and writes 50-digit PMP JSON.

```python
def write_pmp_json_mp(
    cv_list_mp: List[List[mpf]],    # N_operators x n_components mpf values
    f_id_mp: List[mpf],              # n_components mpf values
    output_path: Path,
    significant_digits: int = 50,
) -> Path:
```

Key difference from `write_pmp_json`: uses `mp.nstr(x, sig_digits)` instead of `f"{x:.17e}"`.

#### 2b. `check_feasibility_sdpb_mp` in sdpb.py

New high-level wrapper that accepts mpmath data:

```python
def check_feasibility_sdpb_mp(
    cv_list_mp: List[List[mpf]],
    f_id_mp: List[mpf],
    config: Optional[SdpbConfig] = None,
) -> FeasibilityResult:
```

Same flow as `check_feasibility_sdpb` but calls `write_pmp_json_mp`.

#### 2c. `build_full_constraint_data_mp` in stage_a.py (or constraint_matrix.py)

New function that builds mpmath crossing vectors from the mpmath h cache:

```python
def build_full_constraint_data_mp(
    spectrum: List[SpectrumPoint],
    delta_sigma: float,
    mp_h_cache: dict,  # (delta, spin) -> dict of (p,q) -> mpf
    n_max: int = N_MAX,
    dps: int = 50,
) -> Tuple[List[List[mpf]], List[mpf], np.ndarray, np.ndarray, np.ndarray]:
    """Returns (cv_list_mp, f_id_mp, scalar_mask, scalar_deltas, spinning_mask)"""
```

This does:
1. `compute_prefactor_table_mp(delta_sigma)` — U_mp (once)
2. For each operator: `compute_crossing_vector_mp(H_mp, U_mp, ...)` using cached H_mp
3. Returns mpf crossing vectors + float64 masks for fast filtering

#### 2d. Modified `find_eps_bound` in stage_a.py

When backend=SDPB, accept mpmath data instead of float64 A:

```python
def find_eps_bound(
    delta_sigma, A, f_id,          # float64 (for scipy)
    scalar_mask, scalar_deltas, spinning_mask,
    config,
    cv_list_mp=None, f_id_mp=None,  # mpmath (for SDPB)
    ...
)
```

The `is_excluded(gap)` function routes to:
- scipy backend: uses float64 A_sub as before
- sdpb backend: filters cv_list_mp by mask, calls check_feasibility_sdpb_mp

#### 2e. Modified `run_scan` in stage_a.py

When backend=SDPB:
1. Load mpmath h cache instead of (or alongside) float64 h cache
2. Call `build_full_constraint_data_mp` to get mpf crossing vectors
3. Pass both float64 masks and mpf vectors to `find_eps_bound`

#### 2f. `load_mp_h_cache` in stage_a.py

New function to load the mpmath h cache from disk:

```python
def load_mp_h_cache(
    spectrum: List[SpectrumPoint],
    cache_path: Path = DATA_DIR / "cached_blocks" / "mp_h_cache.pkl.gz",
) -> dict:
```

### Phase 3: Tests

- Unit test: `write_pmp_json_mp` produces valid JSON with 50-digit strings
- Unit test: `check_feasibility_sdpb_mp` gives same results as float64 for small test cases
- Unit test: mpmath crossing vector assembly from cached h matches direct computation
- Integration test: bracket validation passes with mpmath pipeline at n_max=3

### Phase 4: Validation (single-point test on sapphire)

Before full production run:

```bash
# 1. Build mpmath h cache (~1.5h)
sbatch jobs/precompute_mp_h_cache.slurm

# 2. Run single-point Stage A at delta_sigma=0.518
sbatch --partition=sapphire --mem=128G --cpus-per-task=16 --time=36:00:00 \
  --export=ALL,SDPB_TIMEOUT=18000,SIGMA=0.518 \
  jobs/stage_a_sdpb.slurm
```

**Success criterion:** Delta_eps_max ~ 1.41 at delta_sigma = 0.518 (not 0.5, not NaN)

### Phase 5: Full production run

```bash
sbatch --array=0-50 jobs/stage_a_sdpb.slurm  # 51 delta_sigma points
```

## Performance Estimates

### One-time: mpmath h cache build
- 520K operators × ~1s/op (mpmath 3F2 + Casimir recursion)
- 112-core sapphire node: ~1.3h
- Storage: ~2-3 GB compressed

### Per delta_sigma point
| Step | Time | Notes |
|------|------|-------|
| Load mp h cache | ~2-5 min | Read + deserialize ~7.5 GB |
| Compute U_mp, f_id_mp | ~1s | Single prefactor table |
| Assemble 520K crossing vectors (mpmath Leibniz) | ~30-90 min | ~5-10ms per operator |
| Per bisection iteration: write PMP JSON | ~1-2 min | 520K × 50-char strings |
| Per bisection iteration: pmp2sdp | ~7-8 min | Same as before |
| Per bisection iteration: SDPB solve | ~1-2h | Same as before |
| Total per iteration | ~1-2h | Dominated by SDPB |
| Total per sigma (12-16 iterations) | ~15-30h | Similar to before |

The mpmath overhead (crossing vector assembly) is a one-time cost per delta_sigma (~30-90 min),
amortized across all bisection iterations. The per-iteration cost is unchanged from the float64
pipeline since pmp2sdp and SDPB dominate.

### Full pipeline
- mpmath h cache build: ~1.5h (one-time)
- Stage A (51 points × ~24h each): parallelized via SLURM array → ~24-30h walltime
- Stage B: similar
- Total: ~60h (same as before, since SDPB dominates)

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| mpmath crossing assembly too slow (>2h) | Medium | Would add significant per-sigma overhead | Benchmark on 10K operators first; consider caching crossing vectors too |
| mpmath h cache too large (>10 GB) | Low | Storage/loading bottleneck | Use more aggressive compression; shard loading |
| SDPB still gives wrong answer with 50 digits | Very low | Need even higher precision | Increase to 100 digits; check with known solution |
| pickle compatibility issues | Low | Cache unusable across Python versions | Use JSON with decimal string representation instead |
| pmp2sdp can't parse 50-digit numbers | Very low | PMP format broken | SDPB docs say string numbers are arbitrary precision; test first |

## Open Questions

1. **Crossing vector caching:** Should we also cache the assembled mpf crossing vectors
   (not just h_{m,n})? Pro: skip Leibniz sum on every run. Con: 520K × 66 mpf values = much
   larger cache. Decision: benchmark Leibniz sum time first, only cache if >1h.

2. **Fallback to float64 for scipy:** Keep the float64 path for scipy backend? Yes — scipy
   can't use mpmath data anyway, and float64 works at low n_max. The mpmath path is SDPB-only.

3. **Minimum precision:** Is 50 digits sufficient or do we need more? SDPB uses 1024-bit
   (~308 decimal digits) internally. 50 digits should be plenty for the input data since the
   condition number is ~10^26 and 50-digit data gives ~10^-24 effective error after conditioning.

4. **Pickle vs JSON for h cache:** Pickle is faster to load but Python-version-dependent.
   JSON with mpf string values is portable. Start with pickle for speed, add JSON export later.

## Files to Create/Modify

### New files
| File | Purpose |
|------|---------|
| `scripts/precompute_mp_h_cache.py` | Worker script for mpmath h cache build |
| `jobs/precompute_mp_h_cache.slurm` | SLURM array job for cache build |
| `scripts/consolidate_mp_h_cache.py` | Merge shard files into single archive |

### Modified files
| File | Change |
|------|--------|
| `src/ising_bootstrap/lp/sdpb.py` | Add `write_pmp_json_mp`, `check_feasibility_sdpb_mp` |
| `src/ising_bootstrap/lp/constraint_matrix.py` | Add `build_constraint_matrix_mp` |
| `src/ising_bootstrap/scans/stage_a.py` | Add mpmath h cache loading, mpmath constraint build, wire through find_eps_bound |
| `tests/test_lp/test_sdpb.py` | Tests for write_pmp_json_mp |
| `tests/test_scans/test_stage_a.py` | Tests for mpmath pipeline integration |
| `docs/LP_CONDITIONING_BUG.md` | Update status to FIXED once validated |

## Execution Order

```
1. [SLURM] Build mpmath h cache (Phase 1)          ~1.5h
2. [Code]  Implement write_pmp_json_mp (Phase 2a)   ~30min
3. [Code]  Implement check_feasibility_sdpb_mp (2b) ~30min
4. [Code]  Implement build_constraint_data_mp (2c)   ~1h
5. [Code]  Wire through stage_a.py (2d, 2e, 2f)     ~1h
6. [Code]  Tests (Phase 3)                           ~1h
7. [SLURM] Single-point validation (Phase 4)         ~24-30h
8. [SLURM] Full production run (Phase 5)             ~24-30h
```

Steps 1 and 2-6 can run in parallel (cache build on cluster while writing code locally).
