# Handoff: SDPB Error Handling Fix & Testing
**Date:** 2026-02-10
**Status:** Awaiting test results (Job 59829443)
**Author:** Claude (assisted Oscar)

---

## Executive Summary

We identified and fixed a **critical error handling bug** in the SDPB integration that was causing Stage A to produce incorrect results. A test job is currently running to verify the fix works. This document explains the bug, the fix, and what to do next based on test results.

---

## Background: The Bug (Codex's Finding)

### What Codex Discovered

Codex analyzed job 59766814 (all 51 tasks OOM-killed) and found:

1. **Memory is a real blocker**: 64 GB insufficient, SDPB needs ~60+ GB per task
2. **Critical error handling bug exists**: `sdpb.py:363-368` converts SDPB failures (OOM, MPI errors, timeouts) into `FeasibilityResult(excluded=False)`
3. **Bug biases Stage A results upward**: Binary search interprets `excluded=False` as "spectrum allowed" → raises lower bound → converges to upper bound (2.5) instead of true exclusion boundary (~1.41)

### Evidence

- All 51 tasks in job 59766814: OOM-killed at ~65 GB (peak memory)
- Historical precedent: Job 59681179 (MPI bug) → all results = 2.499939 (upper bound)
- Root cause: When SDPB crashes → returns `excluded=False` → binary search thinks "allowed" → keeps trying → converges to 2.5

### Why This is Critical

- **Silent failures** masquerade as valid results
- **Wastes resources** on retry loops (10+ failed attempts per task)
- **No fail-fast** - jobs run for hours producing garbage output
- **Masks underlying issues** (can't distinguish OOM from transient failures)

---

## The Fix (Implemented 2026-02-10)

### Commit: `dcc9f12`

**Files modified:**
- `src/ising_bootstrap/lp/solver.py` - Added `success` field to `FeasibilityResult`
- `src/ising_bootstrap/lp/sdpb.py` - Enhanced exception handlers
- `src/ising_bootstrap/scans/stage_a.py` - Added fail-fast logic

### Changes

**1. Added `success: bool` field to `FeasibilityResult`** (backward compatible)
```python
@dataclass
class FeasibilityResult:
    excluded: bool
    status: str
    lp_status: int
    alpha: Optional[np.ndarray] = None
    fun: Optional[float] = None
    success: bool = True  # NEW: defaults to True
```

**2. SDPB exception handlers now set `success=False`**
```python
except (RuntimeError, FileNotFoundError) as e:
    error_msg = str(e)
    is_oom = "signal 9" in error_msg.lower() or "killed" in error_msg.lower()
    is_mpi = "mpirun" in error_msg.lower()
    failure_type = "OOM kill" if is_oom else "MPI error" if is_mpi else "SDPB error"

    return FeasibilityResult(
        excluded=False,
        status=f"SDPB failed ({failure_type}): {error_msg[:200]}",
        lp_status=-1,  # -1 = explicit failure
        success=False,  # CRITICAL: mark as failed
    )
```

**3. Binary search fail-fast logic**
```python
def is_excluded(gap: float) -> bool:
    result = check_feasibility(...)

    if not result.success:
        consecutive_failures += 1
        if consecutive_failures >= 3:
            raise RuntimeError(
                f"SDPB solver failed {consecutive_failures} consecutive times. "
                f"Last failure: {result.status}. "
                f"Check logs for OOM kills, MPI errors, or resource constraints."
            )
        return False  # Conservative: treat isolated failures as "allowed"

    consecutive_failures = 0
    return result.excluded
```

**4. Scan-level error handling**
```python
try:
    eps_max, n_iter = find_eps_bound(...)
    results.append((delta_sigma, eps_max))
except RuntimeError as e:
    print(f"  ERROR: Failed at Δσ = {delta_sigma:.6f}: {e}")
    failed_points.append((delta_sigma, str(e)))
    append_result_to_csv(config.output, delta_sigma, float('nan'))

# Abort if >50% failed
if len(failed_points) / len(sigma_grid) > 0.5:
    raise RuntimeError("Stage A scan failed: too many solver failures")
```

---

## Test Jobs Submitted

### TEST 1: Error Detection (Job 59829132) - ❌ FAILED

**Purpose:** Verify error detection with insufficient memory
**Memory:** 16G (intentionally low)
**Result:** FAILED with disk space error (compute node had full `/tmp`)
**Conclusion:** Inconclusive - didn't reach SDPB due to disk issue (node-specific problem)

### TEST 2: Sufficient Memory (Job 59829443) - ⏳ RUNNING

**Purpose:** Verify Stage A succeeds with adequate resources
**Memory:** 128G
**Δσ:** 0.518 (near the kink, challenging point)
**Expected:** Δε_max ≈ 1.41 (from arXiv:1203.6064 Fig 6)
**Status:** Currently running SDPB (started 22:27 EST, ~25 min elapsed at handoff)
**Log:** `logs/test_sufficient_memory_59829443.log`
**Output:** `data/test_sufficient_memory.csv`

**Progress so far:**
- ✅ Loaded consolidated block cache
- ✅ Built constraint matrix (520K operators)
- ✅ Wrote PMP JSON (470K blocks)
- ✅ Ran pmp2sdp (305 seconds)
- ⏳ Running SDPB with `--oversubscribe` flag

---

## What To Do Next (Decision Tree)

### Step 1: Check Test Result

**Command:**
```bash
cd /n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap
sacct -j 59829443 --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

### Step 2: Based on Result

---

## ✅ SCENARIO A: Job COMPLETED Successfully

**Check the result:**
```bash
cat data/test_sufficient_memory.csv
tail -50 logs/test_sufficient_memory_59829443.log
```

**Expected output:**
```
delta_sigma,delta_eps_max
0.518000,1.412345  # Should be ~1.41, NOT 2.5
```

**If Δε_max ≈ 1.41:** 🎉 **SUCCESS!**
- Error handling fix works
- 128G is sufficient
- Binary search converges to correct boundary

**Action Items:**
1. Update SLURM script memory allocation:
   ```bash
   # Edit jobs/stage_a_sdpb.slurm
   #SBATCH --mem=128G  # Changed from 64G
   ```

2. Submit full Stage A job array:
   ```bash
   sbatch jobs/stage_a_sdpb.slurm
   ```
   - This will launch 51 tasks (array 0-50)
   - Each task runs one Δσ point
   - Should complete in ~2-8 hours
   - Monitor with: `squeue -u obarrera`

3. Check results after completion:
   ```bash
   # Merge individual CSVs
   python jobs/merge_stage_a.sh

   # Validate results
   python scripts/validate_results.py --stage a

   # Plot Figure 6
   python -m ising_bootstrap.plotting.stage_a
   ```

**Expected behavior:**
- Most/all tasks should succeed
- Δε_max curve should match arXiv:1203.6064 Fig 6
- No tasks should return 2.5 (upper bound bug)
- Some tasks MAY fail if they get nodes with full `/tmp` (rerun those individually)

---

## ❌ SCENARIO B: Job FAILED (OOM or Other Error)

**Check the error:**
```bash
tail -100 logs/test_sufficient_memory_59829443.log
sacct -j 59829443 --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem
```

### If OOM killed (signal 9, MaxRSS ≈ 128G):

**Interpretation:** 128G is insufficient, need more memory

**Action Items:**
1. Check SDPB's memory report in log:
   ```bash
   grep -i "maxSharedMemory\|memory" logs/test_sufficient_memory_59829443.log
   ```

2. Increase memory allocation:
   - If SDPB reported needing ~120G: try 192G
   - If SDPB reported needing >150G: try 256G

3. Resubmit test with higher memory:
   ```bash
   # Edit jobs/test_sufficient_memory.slurm
   #SBATCH --mem=192G  # or 256G

   sbatch jobs/test_sufficient_memory.slurm
   ```

4. If even 256G is insufficient:
   - Consider reducing `n_max` from 10 to 9 (reduces problem size)
   - Or request high-memory nodes (512G available on some partitions)
   - Or use sparsification techniques

### If failed with our new error message:

**Example:** `"SDPB solver failed 3 consecutive times"`

**Interpretation:** Error detection is working! But solver is consistently failing.

**Check log for failure type:**
```bash
grep -i "OOM kill\|MPI error\|timed out" logs/test_sufficient_memory_59829443.log
```

**Action based on failure type:**
- **OOM kill:** Increase memory (see above)
- **MPI error:** Check `--oversubscribe` flag present, check Singularity version
- **Timeout:** Increase `--sdpb-timeout` or use more cores
- **Other:** Investigate specific error message

### If failed with disk space error:

**Interpretation:** Same as TEST 1 - node-specific `/tmp` issue

**Action:** Just resubmit - will likely get a different node with clean `/tmp`

---

## ⚠️ SCENARIO C: Job COMPLETED but Δε_max ≈ 2.5

**This would be UNEXPECTED** - means our fix didn't work.

**Check the result:**
```bash
cat data/test_sufficient_memory.csv
# If second column is 2.499xxx or 2.500xxx, bug still exists
```

**Diagnostic steps:**

1. Check if error detection was triggered:
   ```bash
   grep -i "WARNING.*solver failed\|ERROR.*failed" logs/test_sufficient_memory_59829443.log
   ```

2. Check if SDPB actually ran successfully:
   ```bash
   grep -i "primalObjective\|dualObjective\|terminationReason" logs/test_sufficient_memory_59829443.log
   ```

3. If SDPB ran successfully but still returned 2.5:
   - This suggests SDPB is genuinely finding `excluded=True` at all gap values
   - This would be a physics/math issue, not a software bug
   - Check SDPB output: is it finding spurious functionals?

4. If SDPB failed but no error detection:
   - Our fix has a bug
   - Check that modified files were actually used (verify commit)
   - Review the `success` field propagation logic

**Action:**
- Report this scenario - it's unexpected and requires investigation
- Do NOT submit full job array until resolved

---

## Current SLURM Script Configuration

**File:** `jobs/stage_a_sdpb.slurm`

**Current settings:**
- Account: `randall_lab` ✓
- Partition: `shared` ✓
- Array: `0-50` (51 tasks) ✓
- Time: `08:00:00` ✓
- Memory: `64G` ⚠️ **NEEDS UPDATE TO 128G** (or higher based on test)
- CPUs: `8` ✓
- SDPB image: `tools/sdpb-3.1.0.sif` ✓
- SDPB precision: `1024` ✓
- SDPB cores: `8` ✓

**Σ grid:**
- Min: 0.500
- Step: 0.002
- Tasks: 51
- Max: 0.500 + 50*0.002 = 0.600

---

## Key Files & Locations

**Modified code:**
- `src/ising_bootstrap/lp/solver.py` (FeasibilityResult dataclass)
- `src/ising_bootstrap/lp/sdpb.py` (exception handlers)
- `src/ising_bootstrap/scans/stage_a.py` (fail-fast logic)

**Test scripts:**
- `jobs/test_error_detection.slurm` (16G test, failed due to disk)
- `jobs/test_sufficient_memory.slurm` (128G test, RUNNING)

**Logs:**
- `logs/test_error_detection_59829132.log` (disk error)
- `logs/test_sufficient_memory_59829443.log` (current test)

**Production script:**
- `jobs/stage_a_sdpb.slurm` (needs memory update)

**Results:**
- Individual: `data/eps_bound_{0..50}.csv`
- Merged: `data/stage_a_results.csv` (after merge)

**Documentation:**
- `docs/SDPB_MPI_FIX.md` (previous MPI bug fix)
- `docs/MEMORY_INCREASE_64G.md` (memory allocation history)
- `docs/LP_CONDITIONING_BUG.md` (LP solver conditioning issue)
- `docs/PROGRESS.md` (full project status)

---

## Quick Reference: Commands

**Check job status:**
```bash
sacct -j 59829443 --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem
squeue -u obarrera
```

**Monitor all jobs:**
```bash
watch -n 10 'squeue -u obarrera'
```

**Check test result:**
```bash
cat data/test_sufficient_memory.csv
tail -100 logs/test_sufficient_memory_59829443.log
```

**Submit full Stage A (after successful test):**
```bash
# 1. Update memory in jobs/stage_a_sdpb.slurm
# 2. Submit
sbatch jobs/stage_a_sdpb.slurm
```

**Monitor production jobs:**
```bash
./monitor_jobs.sh  # Real-time monitoring
scripts/analyze_logs.py  # Detailed analysis
```

---

## Expected Timeline

**Test completion:** ~30-60 min from start (22:27 EST)
**Full Stage A array:** ~2-8 hours (51 tasks in parallel)
**Stage B:** Similar timeline after Stage A completes
**Final plot:** Minutes after both stages complete

---

## Summary for Fresh Claude Instance

1. **Read this file** to understand context
2. **Check job 59829443 status** (see commands above)
3. **Follow decision tree** based on result (Scenarios A, B, or C)
4. **If successful:** Update SLURM script memory, submit full array
5. **If failed:** Diagnose error type, adjust resources, resubmit test
6. **Report back** with results and next steps

**Most likely outcome:** Scenario A (success with ~1.41), proceed to full Stage A with 128G.

---

## Contact & Context

**Project:** 3D Ising CFT Bootstrap
**Goal:** Reproduce Figure 6 from arXiv:1203.6064
**Location:** `/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap/`
**Conda env:** `ising_bootstrap`
**Cluster:** Harvard FAS RC (SLURM)
**Account:** `randall_lab` (partition: `shared`)

**Previous attempts:**
- Run 1 (59681179): MPI error → all results = 2.5 ❌
- Run 2 (59716207): MPI fixed, OOM at 48G ❌
- Run 3 (59766497): Canceled, switched accounts ❌
- Run 4 (59766814): OOM at 64G → discovered error handling bug ❌
- **Current:** Testing fix with 128G ⏳

Good luck! 🚀
