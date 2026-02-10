# Changelog — 2026-02-10: SDPB MPI and Memory Fixes

## Summary

Fixed critical bugs causing all Stage A results to converge to upper bound (Δε_max = 2.5).
SDPB solver was failing silently on every call due to MPI oversubscription errors and
out-of-memory kills.

## Changes Made

### 1. Fixed MPI Oversubscription Error

**File:** `src/ising_bootstrap/lp/sdpb.py`
**Line:** 225
**Change:** Added `--oversubscribe` flag to mpirun command

```diff
     cmd = [
         "singularity", "exec",
         "--bind", f"{sdp_dir.parent}:{sdp_dir.parent}",
         "--bind", f"{out_dir.parent}:{out_dir.parent}",
         str(config.image_path),
-        "mpirun", "--allow-run-as-root",
+        "mpirun", "--allow-run-as-root", "--oversubscribe",
         "-n", str(config.n_cores),
         "sdpb",
         f"--precision={config.precision}",
         f"--sdpDir={sdp_dir}",
         f"--outDir={out_dir}",
         "--noFinalCheckpoint",
     ]
```

**Reason:** Singularity container's MPI runtime couldn't see SLURM CPU allocation.
The `--oversubscribe` flag tells MPI to allow launching more processes than it
auto-detects, which is safe because SLURM has already allocated the resources.

### 2. Increased Memory Allocation for Stage A

**File:** `jobs/stage_a_sdpb.slurm`
**Line:** 7
**Change:** 16G → 48G

```diff
 #SBATCH --job-name=ising_stage_a_sdpb
 #SBATCH --account=iaifi_lab
 #SBATCH --partition=shared
 #SBATCH --array=0-50
 #SBATCH --time=08:00:00
-#SBATCH --mem=16G
+#SBATCH --mem=48G
 #SBATCH --cpus-per-task=8
```

**Reason:** SDPB with 1024-bit precision processing 420K+ SDP blocks requires
3-6 GB per MPI rank. With 8 ranks, total usage is 24-48 GB. Original 16 GB
allocation caused OOM kills.

### 3. Increased Memory and CPUs for Stage B

**File:** `jobs/stage_b_sdpb.slurm`
**Lines:** 7-8
**Changes:** 8G → 48G (memory), 4 → 8 (CPUs)

```diff
 #SBATCH --job-name=ising_stage_b_sdpb
 #SBATCH --account=iaifi_lab
 #SBATCH --partition=shared
 #SBATCH --array=0-50
 #SBATCH --time=08:00:00
-#SBATCH --mem=8G
-#SBATCH --cpus-per-task=4
+#SBATCH --mem=48G
+#SBATCH --cpus-per-task=8
```

**Reason:** Stage B processes similar-sized constraint matrices (two-gap spectra).
Consistency with Stage A resource allocation ensures no OOM issues.

### 4. Created Comprehensive Documentation

**New file:** `docs/SDPB_MPI_FIX.md` (440 lines)

Contains:
- Full diagnosis of the problem
- Log analysis and evidence
- Root cause explanation (MPI + OOM)
- Code changes with diffs
- Validation checklist
- Technical deep dive on MPI-in-containers
- Debugging reference guide
- Summary for future Claude sessions

### 5. Updated Project Documentation

**File:** `CLAUDE.md`

- Updated pipeline status section (lines 45-108)
- Added failure modes for Δε_max = 2.5 case
- Added new common pitfalls (#7-8): MPI oversubscribe, SDPB memory
- Updated job chain status

**File:** `~/.claude/projects/.../memory/MEMORY.md`

- Updated pipeline status with fix information
- Added SDPB MPI and OOM to key bugs & fixes
- Updated environment section with new resource requirements

## Validation Checklist

Before considering this fix complete, verify:

- [x] `--oversubscribe` flag added to sdpb.py line 225
- [x] Stage A memory increased to 48G
- [x] Stage B memory increased to 48G
- [x] Stage B CPUs increased to 8
- [x] Documentation created (SDPB_MPI_FIX.md)
- [x] CLAUDE.md updated
- [x] MEMORY.md updated
- [ ] Resubmit Stage A and verify SDPB runs successfully
- [ ] Check logs for `terminateReason` output (not MPI errors)
- [ ] Verify Δε_max results are non-trivial (not all 2.5)
- [ ] Validate Ising point: Δε_max ≈ 1.41 at Δσ ≈ 0.5182

## How to Resubmit Pipeline

```bash
# Navigate to project directory
cd /n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap

# Clean old results
rm -f data/eps_bound_*.csv
rm -f data/epsprime_bound_*.csv
rm -f figures/fig6_reproduction.*

# Resubmit Stage A
JOB_A=$(sbatch jobs/stage_a_sdpb.slurm | awk '{print $4}')
echo "Stage A job ID: $JOB_A"

# Submit merge job with dependency
sbatch --dependency=afterok:$JOB_A jobs/merge_stage_a_and_submit_b.slurm

# Monitor progress
squeue -u obarrera
```

## Expected Outcomes After Fix

### Stage A Logs Should Show:

```
Running sdpb (8 cores, precision=1024) ...
[sdpb] singularity exec ... mpirun --allow-run-as-root --oversubscribe -n 8 sdpb ...
```

**No MPI errors**, and output includes:
```
terminateReason = "found dual feasible solution"
```
or
```
terminateReason = "found primal feasible solution"
```

### Results Should Show:

- **Varied Δε_max values** across Δσ grid (not all 2.5)
- **Ising point validation:** At Δσ ≈ 0.5182, Δε_max ≈ 1.41
- **No OOM kills** in SLURM error logs
- **CSV files populated** with 51 rows each

### Memory Usage:

```bash
sacct -j <JOB_ID> --format=JobID,MaxRSS,Elapsed,State
```

Should show `MaxRSS` ≈ 20-45 GB (under 48 GB limit).

## Related Issues

### Historical Context

This is the **third major bug** in the pipeline development:

1. **LP Conditioning Bug** (2026-02-06): scipy/HiGHS failed at n_max=10 due to
   float64 conditioning (condition number ~4×10¹⁶). Fixed by integrating SDPB.
   See `docs/LP_CONDITIONING_BUG.md`.

2. **NFS Cache I/O Bottleneck** (2026-02-09): Loading 520K individual .npy files
   took 60+ minutes per task. Fixed by consolidating into single .npz archive.

3. **SDPB MPI Failure** (this fix, 2026-02-10): SDPB integration was correct but
   mpirun couldn't launch inside Singularity. Fixed by adding `--oversubscribe`.

### Key Lesson

**Silent failures are dangerous.** The pipeline appeared to run successfully:
- No Python exceptions raised
- Binary search completed
- CSV files written
- SLURM jobs reported SUCCESS

Only by examining the **output values** (all 2.5) did we catch the issue.

**Always validate solver execution, not just solver logic.**

## Files Modified

```
src/ising_bootstrap/lp/sdpb.py              (1 line changed)
jobs/stage_a_sdpb.slurm                     (1 line changed)
jobs/stage_b_sdpb.slurm                     (2 lines changed)
docs/SDPB_MPI_FIX.md                        (new file, 440 lines)
CLAUDE.md                                    (updated)
~/.claude/.../memory/MEMORY.md              (updated)
CHANGELOG_2026-02-10.md                     (this file)
```

## Git Workflow (if using version control)

```bash
git add src/ising_bootstrap/lp/sdpb.py
git add jobs/stage_a_sdpb.slurm
git add jobs/stage_b_sdpb.slurm
git add docs/SDPB_MPI_FIX.md
git add CLAUDE.md
git add CHANGELOG_2026-02-10.md

git commit -m "Fix SDPB MPI oversubscription and OOM issues

- Add --oversubscribe flag to mpirun in sdpb.py (fixes container MPI detection)
- Increase Stage A/B memory allocation from 16G/8G to 48G (prevents OOM kills)
- Increase Stage B CPUs from 4 to 8 for consistency
- Add comprehensive documentation in docs/SDPB_MPI_FIX.md

Previous runs produced Δε_max = 2.5 for all Δσ because SDPB was failing
silently on every call. This fix should produce correct, non-trivial results."
```

---

**Status:** Ready for resubmission
**Date:** 2026-02-10
**Author:** Claude (Sonnet 4.5)
**Reviewed:** User validation pending
