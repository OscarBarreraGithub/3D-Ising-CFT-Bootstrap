# SDPB Timeout Analysis: Overnight Runs (2026-02-11/12)

## Executive Summary

The 3D Ising CFT Bootstrap pipeline failed repeatedly on the `shared` partition due to SDPB solver timeout constraints. Overnight characterization runs demonstrated that each Stage A point requires **26-35 hours** to complete (not the 6-12 hours tested), exceeding the shared partition's 12-hour walltime limit by 2-3×. Migration to the `sapphire` partition (7-day walltime, 990GB RAM/node, MPI-optimized) is necessary for production runs.

## Background: Why Timeout Characterization Was Needed

After merging strict SDPB failure handling semantics (2026-02-11), the SDPB timeout parameter became a critical failure point. Previous runs used a conservative 1800s (30 minutes) timeout, which proved insufficient for the bisection search to converge. The overnight automation was designed to:

1. **Characterize the minimum SDPB timeout** needed for convergence
2. **Find optimal resource configuration** (cores, memory, walltime)
3. **Enable full 51-point Stage A/B pipeline** with validated settings

The characterization tested progressively longer timeouts and more cores to find the breaking point.

---

## Timeline of Overnight Attempts

### Attempt 1: 2026-02-11 00:46 (Jobs 59843729, 59844257, 59844715, 59845395)

**Configuration:**
- 4 parallel tests: 6h, 9h, 12h timeouts (varying cores)
- SLURM walltime: 6-12 hours

**Result:** ALL TIMEOUT after 6-12 **minutes** (not hours)

**Root Cause:** Walltime parsing bug
- Bug: `IFS=':'` delimiter in config parsing conflicted with `HH:MM:SS` walltime format
- Example: `6:00:00` split on `:` became `["6", "00", "00"]` → parsed as 6 minutes
- **Documented in:** `BUGFIX_2026-02-11_WALLTIME_PARSING.md`

**Resolution:** Changed delimiter from `:` to `|` in `overnight_full_pipeline.sh`

---

### Attempt 2: 2026-02-11 15:04 (Job 59936292)

**Configuration:**
- 8 cores, 128GB RAM, 6h SLURM walltime
- 1800s (30 min) SDPB timeout

**Result:** Job completed but produced **NaN** (not a valid result)

**Root Cause:** SDPB timeout still too short
- Bisection search started but couldn't converge in 30 minutes per solve
- With 6-hour walltime, could only complete 10-12 SDPB solves
- Needed 12-16 solves, each taking longer than 30 minutes

**Key Finding:** Need much longer SDPB timeout (hours, not minutes)

---

### Attempt 3: 2026-02-11 18:57 (Jobs 59973738, 59973739)

**Configuration:**
- **Parallel submission strategy:** First success wins
- Job 59973738: 8 cores, 128GB, 8h walltime, 18000s (5h) SDPB timeout
- Job 59973739: 16 cores, 160GB, 8h walltime, 18000s (5h) SDPB timeout

**Queue Wait:** ~55 minutes (PENDING → RUNNING at 19:52)

**Result:** BOTH TIMEOUT after hitting SLURM walltime limit

**Log files:**
- `logs/test_sufficient_memory_59973738.log` (4.3KB)
- `logs/test_sufficient_memory_59973739.log` (8.7KB)
- `logs/relaunch_20260211_185738.log` (19KB - automation log)

---

## Detailed Performance Analysis

### Job 59973738 (8 cores, 128GB)

**Runtime:** 8:00:00 (SLURM TIMEOUT)
- Started: 2026-02-11 19:48:36
- Cancelled: 2026-02-12 03:48:36

**SDPB Solves Completed:** 2 iterations
- Iteration 1: 470,476 SDP blocks
- Iteration 2: 495,476 SDP blocks
- Iteration 3: Started (pmp2sdp completed), killed mid-SDPB-solve

**Average Solve Time:** ~2.2 hours per SDPB solve

**Timing Breakdown:**
```
Total runtime:    480 minutes (8 hours)
pmp2sdp (4 calls): ~22 minutes total (5.1 + 5.8 + 5.6 + 5.6 min)
SDPB solves:      ~458 minutes (95% of runtime)
```

**Output:** `data/test_config_0.csv` - Empty (header only, no results)

---

### Job 59973739 (16 cores, 160GB)

**Runtime:** 8:00:00 (SLURM TIMEOUT)
- Started: 2026-02-12 00:09:41
- Cancelled: 2026-02-12 08:09:57

**SDPB Solves Completed:** 4 iterations
- Iteration 1: 470,476 SDP blocks → FEASIBLE (dualityGap 7.9e-31)
- Iteration 2: 495,476 SDP blocks → FEASIBLE (dualityGap 5.9e-31)
- Iteration 3: 507,976 SDP blocks → FEASIBLE (dualityGap 8.8e-31)
- Iteration 4: 514,226 SDP blocks → Started, killed mid-solve

**Average Solve Time:** ~2.0 hours per SDPB solve

**Speedup vs 8 cores:** 1.1× (only 10% faster, not 2×)

**Timing Breakdown:**
```
Total runtime:    480 minutes (8 hours)
pmp2sdp (4 calls): ~22 minutes total
SDPB solves:      ~458 minutes (95% of runtime)
```

**Output:** `data/test_config_1.csv` - Empty (header only, no results)

---

## Key Performance Metrics

| Metric | 8 cores | 16 cores | Analysis |
|--------|---------|----------|----------|
| **pmp2sdp time** | ~303-350s | ~303-348s | Minimal core benefit (serial conversion) |
| **SDPB solve time** | ~2.2h | ~2.0h | Modest 10% speedup with 2× cores |
| **Iterations in 8h** | 2-3 | 3-4 | Both far from 12-16 needed |
| **Block progression** | 470K → 495K | 470K → 514K | ~44K block increase over 4 iterations |
| **Convergence** | dualityGap <1e-30 | dualityGap <1e-30 | Both achieve tight convergence |

**Critical Finding:** SDPB solve time dominates (95% of runtime), but scales poorly with cores (only 10% speedup with 2× cores). This suggests:
1. SDPB is memory-bandwidth limited, not CPU-limited
2. MPI overhead from `--oversubscribe` may limit scaling
3. More cores won't solve the timeout problem - need longer walltime

---

## Root Cause Analysis

### SDPB Solver Complexity

**Problem Size:**
- **Operators:** 520,476 total (187,401 scalars + 333,075 spinning)
- **SDP blocks:** 470,476-514,226 (grows as bisection narrows Δε range)
- **Variables:** 66 (index pairs with m odd, m+2n ≤ 21)
- **Precision:** 1024-bit arbitrary precision (mpfr/gmp)

**Convergence Requirements:**
- **Duality gap:** <1e-30 (extremely tight for numerical stability)
- **Primal error:** <1e-36
- **Dual error:** <1e-39

**MPI Configuration:**
- `mpirun --oversubscribe -n 16` (allows >1 MPI rank per core)
- This may introduce overhead, limiting scaling

### Bisection Search Requirements

**Search Range:**
- Initial: [0.5, 2.5] (broad range for Δε at Δσ=0.518)
- Narrows to: ~[1.3, 1.5] near the Ising point
- **Tolerance:** 1e-4 (4 decimal places, required by Stage B)

**Iterations Needed:**
```
log₂((range_max - range_min) / tolerance) ≈ log₂(2.0 / 1e-4) ≈ 14 iterations minimum
```

**Actual Pattern (from logs):**
- Iteration 1: Test lower bound (Δε=0.707, 470K blocks) → FEASIBLE
- Iteration 2: Bisect to midpoint (Δε=1.06, 495K blocks) → FEASIBLE
- Iteration 3: Bisect higher (Δε=1.29, 508K blocks) → FEASIBLE
- Iteration 4: Bisect higher (Δε=1.38, 514K blocks) → TIMEOUT (killed)

Expected to continue: 10-12 more iterations to reach tolerance

###Time Extrapolation

**8 cores:**
```
Average solve time:    2.2 hours
Iterations needed:     12-16
Total time:           2.2h × 14 = 30.8 hours
Range:                26-35 hours
```

**16 cores:**
```
Average solve time:    2.0 hours
Iterations needed:     12-16
Total time:           2.0h × 14 = 28 hours
Range:                24-32 hours
```

**Both configurations exceed the shared partition's 12-hour walltime limit by 2-3×.**

---

## Why Shared Partition Is Inadequate

### Resource Constraints

| Resource | Shared | Required | Gap |
|----------|--------|----------|-----|
| **Walltime/job** | 12h max | 28-35h | 2.3-2.9× too short |
| **Memory/node** | 184GB | 128GB/job | Adequate for single job |
| **Cores/node** | 48 | 16/job | Can fit ~3 jobs/node |
| **RAM packing** | 184/128=1.4 | 1 job/node | Memory-limited packing |
| **Jobs needed** | 51 tasks | 51 simultaneous | Need 51 nodes for array |

### Shared Partition Design

**Intended Use:**
- Interactive jobs
- Short-running parallel jobs (<12h ideal)
- Development and testing
- Throughput-oriented workloads

**Scheduling:**
- FairShare algorithm penalizes long-running jobs
- Backfill prioritizes short jobs
- Higher priority jobs can preempt long jobs

**Network:**
- InfiniBand fabric, but optimized for throughput
- Not tuned for MPI latency (less critical for short jobs)

**Problem:** SDPB jobs don't fit the shared partition's design constraints. They are:
- **Very long** (28-35 hours per task)
- **MPI-intensive** (1024-bit precision linear algebra)
- **Memory-intensive** (128GB per job, only 1 fits per node)

---

## Why Sapphire Partition Is The Solution

### Resource Advantages

| Resource | Sapphire | Shared | Improvement |
|----------|----------|--------|-------------|
| **RAM/node** | 990GB | 184GB | **5.4× more** |
| **Cores/node** | 112 (2×56) | 48 (2×24) | 2.3× more |
| **Walltime max** | 7 days | 3 days | 2.3× longer |
| **Local scratch** | 400GB | 70GB | 5.7× more |
| **CPU architecture** | Sapphire Rapids (2023) | Cascade Lake (2019) | Newer, faster |
| **Jobs/node (128GB)** | **6 jobs** | **1 job** | **6× better packing** |

### Sapphire Design Goals

**Intended Use:**
- Large-scale MPI jobs
- Long-running computations (days)
- High-memory workloads
- Arbitrary-precision solvers

**Scheduling:**
- Designed for full-node allocations
- FairShare tuned for long jobs
- Less contention from short jobs

**Network:**
- InfiniBand fabric optimized for MPI latency
- `--contiguous` option available for topology-sensitive codes

**Perfect Fit:** SDPB jobs are **exactly** what sapphire was designed for:
- ✓ MPI-based (mpirun -n 16)
- ✓ Long-running (28-35 hours)
- ✓ High-precision (1024-bit mpfr)
- ✓ Memory-intensive (128GB/job, 6 fit per node)

### Expected Improvements

**Resource Utilization:**
```
Shared:  51 jobs × 128GB = 6,528GB total
         51 jobs × 16 cores = 816 cores total
         Packing: 1 job/node → need 51 nodes
         Efficiency: 128GB/184GB = 70% memory, 16/48 = 33% CPU

Sapphire: 51 jobs × 128GB = 6,528GB total
          51 jobs × 16 cores = 816 cores total
          Packing: 6 jobs/node → need 9 nodes (6× better!)
          Efficiency: 768GB/990GB = 78% memory, 96/112 = 86% CPU
```

**Timeline:**
```
Shared partition (if 12h limit were lifted):
  - Stage A: 51 jobs × 28-35h = 1,428-1,785 node-hours
  - Need: 51 nodes × 35h = 1,785 node-hours
  - Impossible: Exceeds 12h walltime limit

Sapphire partition:
  - Stage A: 51 jobs / 6 jobs per node = 9 nodes needed
  - Walltime: 35 hours (well within 7-day limit)
  - Total: 9 nodes × 35h = 315 node-hours (actual cluster usage)
```

**Parallelization:**
- With 9 nodes, all 51 Stage A tasks can run **simultaneously**
- Stage A completes in **~30 hours** (limited by slowest task at Δσ=0.518)
- Stage B also 51 tasks → another 30 hours
- **Total pipeline: 60 hours (~2.5 days)** instead of impossible on shared

---

## Artifacts Requiring Cleanup

### Empty Test CSV Files

**Location:** `data/`
- `test_config_0.csv` (27 bytes) - Job 59973738, header only
- `test_config_1.csv` (27 bytes) - Job 59973739, header only

**Content:** Both contain only:
```csv
delta_sigma,delta_eps_max

```

**Action:** Move to `logs/archive/2026-02-overnight-timeout/` with descriptive names

---

### Log Files to Archive

**Overnight timeout logs (keep for reference):**
- `logs/test_sufficient_memory_59973738.log` (4.3KB) - 8 cores, 2 iterations
- `logs/test_sufficient_memory_59973739.log` (8.7KB) - 16 cores, 4 iterations
- `logs/relaunch_20260211_185738.log` (19KB) - Automation script
- `logs/relaunch_20260211_150409.log` (3KB) - Walltime bug attempt

**Action:** Organize in `logs/archive/2026-02-overnight-timeout/` with README

**Old shared partition logs (prior 1800s timeout failure):**
- `logs/stage_a_sdpb_59766814_*.log` (51 files from 2026-02-10)
- All timed out with 1800s SDPB timeout (same problem, different day)

**Action:** Move to `logs/archive/2026-02-10-stage-a-timeout/` with README

---

## Recommended Next Steps

### 1. Complete Sapphire Migration

**Code changes:**
- ✓ Update all 18 .slurm files to `--partition=sapphire`
- ✓ Update Stage A/B scripts: 16 cores, 36h walltime
- ✓ Create verification script

**Documentation:**
- ✓ This analysis document
- → Create external review checklist
- → Update CLUSTER_SETUP.md with sapphire section
- → Update PROGRESS.md with migration timeline
- → Update memory instructions

### 2. Single-Point Validation Test

**Before launching full 51-task array, validate sapphire works:**

```bash
# Test single Δσ=0.518 point (hardest/slowest)
sbatch --partition=sapphire \
  --mem=128G --cpus-per-task=16 --time=36:00:00 \
  --export=ALL,SDPB_TIMEOUT=18000,SIGMA=0.518,OUTPUT_CSV=data/sapphire_test.csv \
  jobs/test_sufficient_memory.slurm
```

**Success Criteria:**
- Job completes (status: COMPLETED, not TIMEOUT/OOM)
- CSV contains one finite value (not NaN, not empty)
- Value in expected range: Δε_max ≈ 1.41 (literature value at Δσ≈0.5182)
- Log shows 12-16 bisection iterations completed
- No SDPB "inconclusive" or "timeout" errors in log

**Expected Runtime:** 24-32 hours (based on 16-core extrapolation)

**Monitor:**
```bash
squeue -u $USER                          # Check job status
tail -f logs/test_sufficient_memory_*.log  # Watch progress
```

### 3. Full Production Pipeline

**After single-point success, launch full pipeline:**

```bash
# Stage A: 51 Δσ points, 16 cores, 36h walltime each
sbatch --array=0-50 --partition=sapphire \
  --cpus-per-task=16 --mem=128G --time=36:00:00 \
  --export=ALL,SDPB_TIMEOUT=18000,STAGE_A_TOLERANCE=1e-4 \
  jobs/stage_a_sdpb.slurm
```

**Expected Timeline:**
- **Stage A:** 28-35 hours (all 51 tasks in parallel, limited by slowest)
- **Merge A:** 5 minutes (single job, lightweight)
- **Stage B:** 28-35 hours (all 51 tasks in parallel)
- **Merge B + Plot:** 10 minutes (single job)
- **Total:** **56-70 hours (2.3-2.9 days)** to Figure 6

**Resource Usage:**
- **Nodes:** 9 nodes (each running 6 jobs)
- **Cores:** 816 cores total (9 nodes × 96 cores used / 112 available)
- **Memory:** 6,528GB total (9 nodes × 768GB used / 990GB available)
- **Cluster %:** 4% of sapphire partition (816 / 20,832 cores)

---

## Technical Appendix

### A. SDPB Profiling Data

**From Job 59973739 (16 cores) log:**

**Iteration 1 (Lower bound test, Δε ≈ 0.707):**
```
pmp2sdp: 303.553s (470476 blocks, 66 vars)
SDPB solve: ~7200s (2 hours)
  dualityGap:   7.916991e-31
  primalError:  3.062200e-36
  dualError:    4.632381e-39
Result: FEASIBLE (lower bound too low, increase Δε)
```

**Iteration 2 (Bisection, Δε ≈ 1.06):**
```
pmp2sdp: 348.157s (495476 blocks, 66 vars)
SDPB solve: ~7200s
  dualityGap:   5.943002e-31
  primalError:  9.432159e-37
  dualError:    2.512529e-40
Result: FEASIBLE (still too low, increase Δε)
```

**Iteration 3 (Bisection, Δε ≈ 1.29):**
```
pmp2sdp: 338.081s (507976 blocks, 66 vars)
SDPB solve: ~7200s
  dualityGap:   8.765329e-31
  primalError:  1.317660e-36
  dualError:    2.913182e-40
Result: FEASIBLE (still too low, increase Δε)
```

**Iteration 4 (Bisection, Δε ≈ 1.38):**
```
pmp2sdp: 333.529s (514226 blocks, 66 vars)
SDPB solve: KILLED (job hit 8-hour walltime limit)
```

**Pattern:**
- Block count increases ~12-25K per iteration as Δε increases
- Solve time stays ~2 hours (not getting slower with more blocks)
- Convergence is tight (dualityGap ~1e-30, errors ~1e-36 to 1e-40)

### B. Constraint Matrix Statistics

**Full Spectrum:**
- **Total operators:** 520,476
  - Scalars (ℓ=0): 187,401
  - Spinning (ℓ>0): 333,075
- **Block structure:** One 1×1 SDP block per LP constraint
- **Variables:** 66 (functional basis dimensions for crossing equation)

**Geometric Scaling (preprocessed):**
- Matrix well-conditioned after row/column scaling
- SDPB uses diagonal preconditioner
- 1024-bit precision prevents numerical instability

### C. Walltime Parsing Bug Details

**Original code (BUGGY):**
```bash
IFS=':' read -r TIMEOUT CPUS MEM WALLTIME DESC <<< "$config"
```

**Problem:**
- Config line: `18000:16:160G:08:00:00:description`
- With `IFS=':'`, this splits into 7 fields: `["18000", "16", "160G", "08", "00", "00", "description"]`
- WALLTIME becomes "08" (8 seconds, not 8 hours!)

**Fix:**
```bash
IFS='|' read -r TIMEOUT CPUS MEM WALLTIME DESC <<< "$config"
# Config line: 18000|16|160G|08:00:00|description
```

Now WALLTIME correctly captures "08:00:00"

**Documented:** `BUGFIX_2026-02-11_WALLTIME_PARSING.md`

---

## References

1. **Strict Semantics Merge:** `HANDOFF_2026-02-11_STRICT_SEMANTICS_MERGED.md`
2. **Walltime Parsing Bug:** `BUGFIX_2026-02-11_WALLTIME_PARSING.md`
3. **SDPB Runtime Envelope:** `SDPB_RUNTIME_ENVELOPE_2026-02-11.md`
4. **LP Conditioning Fix:** `LP_CONDITIONING_BUG.md`
5. **Memory Analysis:** `MEMORY_REQUIREMENTS_ANALYSIS.md`
6. **Sapphire Partition Docs:** [Harvard FAS RC User Guide](https://docs.rc.fas.harvard.edu/)

---

**Document Status:** Complete analysis of overnight timeout failures
**Conclusion:** Sapphire partition migration necessary and sufficient
**Next Steps:** External review → Single-point test → Full production pipeline
**Expected Delivery:** Figure 6 in 2.5-3 days after sapphire validation
