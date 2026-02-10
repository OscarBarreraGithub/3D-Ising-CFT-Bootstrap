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

## Pipeline Status (as of 2026-02-10)

### Previous Run (2026-02-09) — FAILED

Job chain 59681121 → 59681179 → 59681180 ran but produced incorrect results.

**Issue:** All Stage A tasks returned Δε_max ≈ 2.5 (upper bound), indicating SDPB
was failing on every call. See `docs/SDPB_MPI_FIX.md` for full diagnosis.

**Root causes:**
1. MPI oversubscription error (missing `--oversubscribe` flag)
2. Out-of-memory kills (16 GB insufficient for 420K blocks at 1024-bit precision)

**Status:** **FIXED** (2026-02-10)
- Added `--oversubscribe` to mpirun command in `sdpb.py`
- Increased memory allocation: 16G → 48G in Stage A/B SLURM scripts
- Increased Stage B CPUs: 4 → 8 for consistency with Stage A

### Next Run

After fixes are applied, resubmit the pipeline:

```bash
rm -f data/eps_bound_*.csv data/epsprime_bound_*.csv
JOB_A=$(sbatch jobs/stage_a_sdpb.slurm | awk '{print $4}')
sbatch --dependency=afterok:$JOB_A jobs/merge_stage_a_and_submit_b.slurm
```

### Job Chain (Reference)

| Order | Job ID | Name | What it does | Depends on |
|-------|--------|------|-------------|------------|
| 1 | (already done) | `consolidate_cache` | Packs 520K `.npy` block cache files into single `.npz` archive | — |
| 2 | TBD | `stage_a_sdpb` | Stage A scan: upper bound on Δε vs Δσ (51 tasks, SDPB backend) | — |
| 3 | TBD | `merge_a_submit_b` | Merges Stage A CSVs, validates results, auto-submits Stage B + plot | Job 2 |
| 4 | (auto-submitted) | `stage_b_sdpb` | Stage B scan: upper bound on Δε' vs Δσ (51 tasks) | Job 3 |
| 5 | (auto-submitted) | `final_merge_and_plot` | Merges Stage B CSVs, generates `figures/fig6_reproduction.png` | Job 4 |

### Expected Timing

- Consolidation: ~30-60 min
- Stage A (51 tasks): ~1-4 hours per task (8h wall limit)
- Stage B (51 tasks): ~1-4 hours per task (8h wall limit)
- Total: could be 4-16+ hours depending on queue wait times

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

- **All Δε_max = 2.5 (upper bound)**: SDPB failing on every call (returns "allowed" for all gaps).
  Check logs for MPI errors or OOM kills. See `docs/SDPB_MPI_FIX.md`. **NOW FIXED**.
- **All Δε_max = 0.5**: SDPB solver not working correctly. Check solver logs, SDPB output.
  The merge job has a sanity check that aborts if all values are ≤ 0.5.
  Historical issue from scipy/HiGHS conditioning — see `docs/LP_CONDITIONING_BUG.md`.
- **Empty CSV files (header only)**: Tasks stuck in cache loading. Check if
  `data/cached_blocks/ext_cache_consolidated.npz` exists (~1 GB). If not, consolidation
  failed — check consolidation logs.
- **Jobs stuck in PENDING**: Queue congestion or dependency failure. Check `squeue` and
  `sacct -j <JOB_ID>` for dependency status.
- **Stage B never submitted**: Merge job failed or aborted. Check merge job logs.
- **OOM kills in logs**: Insufficient memory. Increase `--mem=` in SLURM script.
  Now fixed at 48G for Stage A/B.

### Key Architecture Decisions

- **NFS cache consolidation**: 520K individual `.npy` files consolidated into one `.npz`
  archive for fast loading (~10-30s vs 60+ min). Fast path in `load_h_cache_from_disk()`
  in `stage_a.py`. Fallback to per-file loading still exists.
- **SDPB backend**: Arbitrary-precision SDP solver replaces scipy/HiGHS. Required because
  constraint matrix condition number is ~4e16 at n_max=10. SDPB runs via Singularity
  container (`tools/sdpb-3.1.0.sif`), 1024-bit precision.
- **LP encoded as degenerate PMP**: Each discrete LP constraint becomes a 1×1 block with
  degree-0 polynomial in `sdpb.py`. SDPB solves this as a semidefinite program.

## Common Pitfalls

1. **Block normalization**: Must use Dolan-Osborn convention
2. **Derivative indexing**: m + 2n ≤ 21, not total order
3. **Discretization**: Table 2 must be followed literally
4. **LP conditioning**: scipy fails at n_max=10 — use SDPB backend
5. **NFS I/O**: Never load 520K individual files from NFS — use consolidated `.npz`
6. **Python buffering**: Always set `PYTHONUNBUFFERED=1` in SLURM scripts
7. **MPI in containers**: Must use `--oversubscribe` flag for mpirun inside Singularity
8. **SDPB memory**: 1024-bit precision needs 3-6 GB per MPI rank (48G total for 8 cores)
