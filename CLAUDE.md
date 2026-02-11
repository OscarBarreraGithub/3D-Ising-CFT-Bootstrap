# 3D Ising CFT Bootstrap

## Project Overview

Reproducing Figure 6 from arXiv:1203.6064 ("Solving the 3D Ising Model with the Conformal Bootstrap") using conformal bootstrap and linear programming.

## Development Setup

- **Conda env**: `conda activate ising_bootstrap`
- **Install**: `pip install -e .[dev]`
- **Test**: `pytest tests/`

## Key Files

| Path | Purpose |
|------|---------|
| `src/ising_bootstrap/config.py` | Physical constants (D=3, n_max=10) |
| `src/ising_bootstrap/blocks/` | Conformal block computation at z=z̄=1/2 |
| `src/ising_bootstrap/spectrum/` | Table 2 discretization (T1-T5) |
| `src/ising_bootstrap/lp/solver.py` | LP feasibility (scipy + SDPB backends) |
| `src/ising_bootstrap/lp/sdpb.py` | SDPB integration (PMP writer, subprocess runner) |
| `src/ising_bootstrap/scans/` | Stage A and B bootstrap scans |
| `src/ising_bootstrap/plot/fig6.py` | Figure 6 generation |
| `tools/sdpb-3.1.0.sif` | SDPB Singularity container (gitignored) |

## LP Solver

Production runs use **SDPB** (arbitrary-precision SDP solver) via `--backend sdpb`.
The scipy/HiGHS backend fails at n_max=10 due to float64 conditioning (condition number ~4e16).
See `docs/LP_CONDITIONING_BUG.md` for the full diagnosis and fix.

## Critical Parameters

- `n_max = 10` → 66 index pairs (m,n) with m odd, m+2n ≤ 21
- Crossing point: z = z̄ = 1/2 (u = v = 1/4)
- Expected results at Δσ ≈ 0.5182: Δε ≈ 1.41, Δε' ≈ 3.84

## Code Standards

- Use `mpmath` for extended precision (50+ decimal digits minimum)
- Follow Table 2 discretization from paper **exactly**
- Cache block derivatives to `data/cached_blocks/`
- Normalization: Dolan-Osborn convention (critical for matching paper results)

## Pipeline Status (as of 2026-02-11)

### Strict SDPB Semantics Merged ✓

**Branch merged:** `codex/strict-failfast-stageb-snap-eps` → `main` (2026-02-11)

**What changed:**
1. **Strict SDPB failure handling** - inconclusive outcomes (timeout, unknown) now return `success=False`
2. **Fail-fast Stage A/B** - ANY solver failure immediately aborts that point (no silent fallbacks)
3. **Stage B epsilon anchoring** - snaps Stage A Δε to scalar grid with tolerance enforcement
4. **Hardened merge gates** - validates Stage A data before launching Stage B
5. **Timeout configuration** - `--sdpb-timeout` exposed and wired through all scripts
6. **Runtime helpers** - single-point characterization, pilot jobs, smoke tests

**All tests passing:** 17 SDPB + 28 Stage A + 30 Stage B = 75/75 ✓

### Current Blocker: SDPB Timeout

**Problem:** SDPB needs >10 minutes per solve but times out at 10 minutes (600s)
- Constraint matrix: 520,476 operators → 470,476 SDP blocks
- Memory: 128G sufficient (70GB used, no OOM)
- No MPI errors, no crashes
- Just too slow for current timeout

**Solution:** Find correct timeout via runtime envelope characterization

### Next Steps

**Phase 1: Single-point characterization** (Δσ=0.518)
```bash
# Try increasing timeouts until one succeeds
bash jobs/submit_stage_a_runtime_envelope.sh             # 1800s, 8 cores, 128G
TIMEOUT=3600 bash jobs/submit_stage_a_runtime_envelope.sh  # If timed out
TIMEOUT=3600 CPUS=16 MEM=160G bash ...                    # More resources
```

**Success criterion:** `data/test_sufficient_memory.csv` shows `0.518000,1.41XXXX` (not NaN, not 2.5)

**Phase 2: Pilot + Full Array**
Once single-point works, use those parameters for the full pipeline.

**See:** `HANDOFF_2026-02-11_STRICT_SEMANTICS_MERGED.md` for complete workflow

### Job Chain (Reference)

| Order | Job ID | Name | What it does | Depends on |
|-------|--------|------|-------------|------------|
| 1 | (already done) | `consolidate_cache` | Packs 520K `.npy` block cache files into single `.npz` archive | — |
| 2 | TBD | `stage_a_sdpb` | Stage A scan: upper bound on Δε vs Δσ (51 tasks, SDPB backend) | — |
| 3 | TBD | `merge_a_submit_b` | Merges Stage A CSVs, validates results, auto-submits Stage B + plot | Job 2 |
| 4 | (auto-submitted) | `stage_b_sdpb` | Stage B scan: upper bound on Δε' vs Δσ (51 tasks) | Job 3 |
| 5 | (auto-submitted) | `final_merge_and_plot` | Merges Stage B CSVs, generates `figures/fig6_reproduction.png` | Job 4 |

### Expected Timing

**Note:** Timing depends on SDPB timeout configuration (TBD via envelope characterization)

- Consolidation: ~30-60 min (already done)
- Stage A (51 tasks): TBD based on per-solve time (wall limit: 12h)
  - If SDPB needs 30 min/solve × 10 iterations = 5h per task
  - If SDPB needs 60 min/solve × 10 iterations = 10h per task
- Stage B (51 tasks): Similar to Stage A
- Total: TBD (likely 12-24+ hours for full pipeline)

### How to Check Status

```bash
# See what's running/pending
squeue -u obarrera

# Check Stage A logs for a specific task (e.g. task 9)
tail -50 logs/stage_a_sdpb_59681179_9.log

# Check if Stage A produced results
for i in $(seq 0 50); do wc -l data/eps_bound_${i}.csv 2>/dev/null; done

# Check merged Stage A output
head data/eps_bound.csv

# Check if Figure 6 was generated
ls -la figures/fig6_reproduction.*

# Resource usage for a completed job
bash jobs/check_usage.sh <JOB_ID>
```

### What Success Looks Like

- `data/eps_bound.csv`: 51 rows with Δε_max values significantly above 0.5 (not all 0.5)
- At Δσ ≈ 0.5182: Δε_max ≈ 1.41 (from paper)
- `data/epsprime_bound.csv`: 51 rows with Δε'_max values
- At Δσ ≈ 0.5182: Δε'_max ≈ 3.84 (from paper)
- `figures/fig6_reproduction.png`: Upper bound curve matching Figure 6 of arXiv:1203.6064

### What Failure Looks Like & What to Do

- **NaN values in CSV**: SDPB timeout too short or solver failing.
  With strict semantics (2026-02-11), ANY solver failure → NaN for that point.
  Check logs for timeout messages. Increase `SDPB_TIMEOUT` env var.
- **All Δε_max = 2.5 (upper bound)**: Historical issue (2026-02-09).
  SDPB was failing but being treated as "allowed". **NOW FIXED** with strict semantics.
- **All Δε_max = 0.5 (unitarity floor)**: Historical issue from scipy/HiGHS conditioning.
  **NOW FIXED** with SDPB backend. Merge gate will block this pattern.
- **Empty CSV files (header only)**: Tasks stuck in cache loading. Check if
  `data/cached_blocks/ext_cache_consolidated.npz` exists (~1 GB).
- **Jobs stuck in PENDING**: Queue congestion or dependency failure. Check `squeue` and
  `sacct -j <JOB_ID>`.
- **Stage B never submitted**: Merge gate blocked due to invalid Stage A data.
  Check `jobs/merge_stage_a_and_submit_b.slurm` logs for validation failures.
- **OOM kills in logs**: Insufficient memory. Now configured at 128G for Stage A/B.

### Key Architecture Decisions

- **NFS cache consolidation**: 520K individual `.npy` files consolidated into one `.npz`
  archive for fast loading (~10-30s vs 60+ min). Fast path in `load_h_cache_from_disk()`
  in `stage_a.py`. Fallback to per-file loading still exists.
- **SDPB backend**: Arbitrary-precision SDP solver replaces scipy/HiGHS. Required because
  constraint matrix condition number is ~4e16 at n_max=10. SDPB runs via Singularity
  container (`tools/sdpb-3.1.0.sif`), 1024-bit precision.
- **LP encoded as degenerate PMP**: Each discrete LP constraint becomes a 1×1 block with
  degree-0 polynomial in `sdpb.py`. SDPB solves this as a semidefinite program.

## Strict Failure Semantics (2026-02-11)

**IMPORTANT:** The pipeline now uses strict fail-fast behavior:

1. **SDPB inconclusive outcomes = FAILURE**
   - Timeouts, unknown termination reasons → `success=False`
   - Only explicit feasible/infeasible results are trusted
   - No silent "allowed" fallbacks

2. **Stage A/B fail on ANY solver anomaly**
   - First solver failure → point aborts → NaN written to CSV
   - No retry logic, no conservative fallback
   - This is CORRECT once timeout is properly configured

3. **Stage B requires anchored epsilon**
   - Stage A Δε is snapped to nearest scalar grid point
   - Snap tolerance enforced (default 1e-3)
   - Exactly one scalar must exist at anchored ε

4. **Merge gate validates Stage A before Stage B**
   - Blocks if non-finite values present
   - Blocks if all values near 0.5 (unitarity floor)
   - Blocks if all values near 2.5 (upper bound)
   - Blocks if too few valid rows (<10)

**This prevents silent failures but requires proper SDPB timeout configuration.**

## Common Pitfalls

1. **SDPB timeout too short**: With strict semantics, timeout → NaN. Must find correct timeout via envelope characterization.
2. **Block normalization**: Must use Dolan-Osborn convention
3. **Derivative indexing**: m + 2n ≤ 21, not total order
4. **Discretization**: Table 2 must be followed literally
5. **LP conditioning**: scipy fails at n_max=10 — use SDPB backend
6. **NFS I/O**: Never load 520K individual files from NFS — use consolidated `.npz`
7. **Python buffering**: Always set `PYTHONUNBUFFERED=1` in SLURM scripts
8. **MPI in containers**: Must use `--oversubscribe` flag for mpirun inside Singularity
9. **SDPB memory**: 1024-bit precision needs ~8-10 GB per MPI rank (128G total for 8 cores at n_max=10)
