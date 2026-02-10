# Memory Increase to 64GB — 2026-02-10 (Second Iteration)

## Issue

Job 59716207 with 48GB allocation still experienced OOM kills:
- **MaxRSS:** ~50GB (exceeded 48GB limit)
- **32 OOM kill events** across all tasks
- **Result:** Still Δε_max = 2.499939 (SDPB killed before completing)

## Good News

The MPI fix from earlier **worked perfectly**:
- No more "oversubscribe" errors
- SDPB actually starts: `2026-Feb-10 16:34:26 Start timing run`
- Progress through binary search iterations

## Problem

SDPB is being **killed mid-execution** by the OOM killer:
```
mpirun noticed that process rank X exited on signal 9 (Killed)
```

This happens partway through the solver, after pmp2sdp completes.

## Analysis

**Memory usage per binary search iteration:**
1. **PMP JSON:** ~200-500 MB (text format, high-precision floats)
2. **Binary SDP files (pmp2sdp output):** ~500 MB - 1.5 GB
3. **SDPB solver working memory:** ~4-8 GB per MPI rank × 8 ranks = 32-64 GB
4. **Peak during solver:** File buffers + interior-point matrices = **50+ GB**

**Why 48GB wasn't enough:**
- Each binary search iteration processes 420K blocks
- SDPB keeps matrices in memory during iteration
- Peak usage exceeds allocation → OOM kill
- Binary search never completes → defaults to upper bound (2.5)

## Solution

Increase memory allocation to **64GB**:
- 64GB / 8 MPI ranks = 8GB per rank (vs 6GB with 48GB)
- Measured peak usage ~50GB fits comfortably under 64GB
- Provides headroom for filesystem buffers and overhead

## Changes

**File:** `jobs/stage_a_sdpb.slurm`
**Line:** 7
```diff
-#SBATCH --mem=48G
+#SBATCH --mem=64G
```

**File:** `jobs/stage_b_sdpb.slurm`
**Line:** 7
```diff
-#SBATCH --mem=48G
+#SBATCH --mem=64G
```

## Resource Usage

**Per task:**
- Memory: 64 GB
- CPUs: 8 cores
- Time: 8 hours (wall time limit)

**Total cluster usage (if all 51 tasks run simultaneously):**
- Memory: 51 × 64 GB = **3.3 TB**
- CPUs: 51 × 8 = 408 cores

**In practice:**
SLURM schedules tasks based on availability. Typically 5-20 tasks run in parallel depending on cluster load. As tasks complete, new ones start.

## Resubmission

**Job IDs:**
- Stage A: **59766497[0-50]** (51 tasks, 64GB each)
- Merge: **59766498** (depends on Stage A)
- Stage B: (auto-submitted by merge job)
- Plot: (auto-submitted by merge job)

**Status:** Running as of 2026-02-10

## Why This Should Work

The **measured peak usage is ~50GB**, so 64GB provides:
- 14GB headroom (28% margin)
- Room for variation across different Δσ values
- Buffer for filesystem I/O caches

If this still fails with OOM, alternatives are:
1. **Reduce MPI cores to 4** (gives 16GB per rank instead of 8GB)
2. **Reduce SDPB precision to 512-bit** (cuts memory usage ~50%)
3. **Increase to 80GB or 96GB** (more headroom but harder to schedule)

## Documentation Updates

Updated files:
- `jobs/stage_a_sdpb.slurm` → 64G
- `jobs/stage_b_sdpb.slurm` → 64G
- `docs/MEMORY_INCREASE_64G.md` (this file)

See also:
- `docs/SDPB_MPI_FIX.md` — First fix (MPI oversubscribe)
- `CHANGELOG_2026-02-10.md` — Full changelog

## Expected Outcome

With 64GB, SDPB should complete successfully and produce:
- **Non-trivial Δε_max values** (varied, not all 2.5)
- **At Δσ ≈ 0.5182:** Δε_max ≈ 1.41 (from paper)
- **No OOM kills** in logs
- **Figure 6** generated successfully

---

**Updated:** 2026-02-10
**Previous attempt:** 48GB (insufficient)
**Current attempt:** 64GB
**Jobs:** 59766497 (Stage A), 59766498 (merge)
