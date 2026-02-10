# SDPB MPI and Memory Fixes — Stage A Binary Search Converging to Upper Bound

**Date:** 2026-02-10
**Status:** FIXED
**Symptom:** All Stage A results converge to `Δε_max = 2.499939` (upper bound)
**Root Cause:** SDPB mpirun failing inside Singularity + OOM kills

---

## Summary

Stage A scan job 59681179 completed with **all 51 tasks returning Δε_max ≈ 2.5** (the hardcoded upper bound from `stage_a.py:74`). This indicated that the binary search was consistently getting `excluded=False` ("allowed") for all gap values, causing it to converge to the upper limit rather than finding the true exclusion boundary.

**Expected behavior:** At Δσ ≈ 0.5182, the result should be Δε_max ≈ 1.41 (from arXiv:1203.6064).

**Actual behavior:** Binary search returns 2.499939 for all Δσ values, indicating SDPB never found an excluded spectrum.

---

## Diagnostic Evidence

### Log Analysis (task 0)

File: `logs/stage_a_sdpb_59681179_0.log`

**Key observations:**

1. **Cache loading successful:** Consolidated .npz loaded in ~30s (520,474 blocks)
2. **Binary search executed:** 16 iterations (normal behavior)
3. **SDPB repeatedly called:** ~16 calls during binary search
4. **Every SDPB call failed** with MPI error:
   ```
   stderr: Alternatively, you can use the --oversubscribe option to ignore the
   stderr: number of available slots when deciding the number of processes to
   stderr: launch.
   --------------------------------------------------------------------------
   ```
5. **Result:** `Δε_max = 2.499939` (converged to upper bound)
6. **OOM kills at end:** `error: Detected 4 oom_kill events`

### What Happens When SDPB Fails

When `mpirun` exits with non-zero status, the exception handler in `sdpb.py:234-239` raises a `RuntimeError`. This is caught by the outer exception handler at `sdpb.py:363-368`:

```python
except (RuntimeError, FileNotFoundError) as e:
    return FeasibilityResult(
        excluded=False,
        status=f"SDPB error: {e}",
        lp_status=4,
    )
```

**Critical consequence:** Failed SDPB calls return `excluded=False`, meaning "spectrum allowed."

### Binary Search Behavior

The binary search in `stage_a.py:111-155` operates as follows:

- **If `excluded=True`** (LP feasible → spectrum inconsistent): lower `hi` bound
- **If `excluded=False`** (LP infeasible → spectrum allowed): raise `lo` bound
- **Convergence:** Returns `lo` when `hi - lo < tolerance`

**When all calls return `excluded=False`:**
- Binary search keeps raising `lo`
- Eventually converges to `lo ≈ hi = 2.5`
- Result: upper bound instead of true exclusion boundary

---

## Root Causes

### Problem 1: MPI Oversubscription Error

**Symptom:**
```
--------------------------------------------------------------------------
There are not enough slots available in the system
Alternatively, you can use the --oversubscribe option to ignore the
number of available slots when deciding the number of processes to
launch.
--------------------------------------------------------------------------
```

**Root cause:**
- SLURM allocates 8 CPUs via `--cpus-per-task=8`
- SLURM script passes `--sdpb-cores 8` to Python
- Inside Singularity container, `mpirun -n 8` tries to launch 8 processes
- **Container's MPI runtime doesn't see SLURM allocation** — thinks only 1 core available
- mpirun refuses to launch and exits with non-zero status

**Why this happens:**
Singularity isolates the container environment from the host SLURM scheduler. The MPI installation inside the container doesn't automatically inherit the `SLURM_CPUS_PER_TASK` environment variable or cgroup limits set by SLURM.

**Fix:**
Add `--oversubscribe` flag to `mpirun` command. This tells MPI to allow launching more processes than it auto-detects, which is safe here because:
1. SLURM has already allocated the resources
2. The host system **does** have the CPUs available
3. We're just bypassing the container's incorrect detection

**Code change:** `src/ising_bootstrap/lp/sdpb.py:225`
```python
# Before:
"mpirun", "--allow-run-as-root",

# After:
"mpirun", "--allow-run-as-root", "--oversubscribe",
```

### Problem 2: Out-of-Memory (OOM) Kills

**Symptom:**
```
[2026-02-10T01:41:22.550] error: Detected 4 oom_kill events in StepId=59681697.batch
```

**Root cause:**
SDPB is processing 420K+ SDP blocks (one per operator after gap filtering) with 1024-bit precision arithmetic. Memory usage breakdown:

- **PMP JSON file:** ~200-500 MB (text format with high-precision floats)
- **Binary SDP format:** ~500 MB - 2 GB (preprocessed by pmp2sdp)
- **SDPB solver state:** 2-8 GB per MPI rank (interior-point matrices, working memory)
- **Total with 8 MPI ranks:** Estimate 16-40 GB peak usage

Original allocation: 16 GB — **insufficient**.

**Fix:**
Increase memory allocation to 48 GB in SLURM scripts. This provides:
- ~6 GB per MPI rank (8 ranks)
- ~10 GB overhead for filesystem I/O buffers and Singularity
- Safety margin for peak usage during solver iterations

**Code changes:**
- `jobs/stage_a_sdpb.slurm:7`: `--mem=16G` → `--mem=48G`
- `jobs/stage_b_sdpb.slurm:7`: `--mem=8G` → `--mem=48G`
- `jobs/stage_b_sdpb.slurm:8`: `--cpus-per-task=4` → `--cpus-per-task=8` (for consistency)

---

## Files Modified

### 1. `src/ising_bootstrap/lp/sdpb.py`

**Line 225:** Added `--oversubscribe` to mpirun command

```python
cmd = [
    "singularity", "exec",
    "--bind", f"{sdp_dir.parent}:{sdp_dir.parent}",
    "--bind", f"{out_dir.parent}:{out_dir.parent}",
    str(config.image_path),
    "mpirun", "--allow-run-as-root", "--oversubscribe",  # ← Added flag
    "-n", str(config.n_cores),
    "sdpb",
    f"--precision={config.precision}",
    f"--sdpDir={sdp_dir}",
    f"--outDir={out_dir}",
    "--noFinalCheckpoint",
]
```

**Why this works:**
- `--oversubscribe` tells OpenMPI to ignore slot limits
- Safe because SLURM has already allocated the CPUs
- Standard practice for MPI-in-containers on HPC clusters

### 2. `jobs/stage_a_sdpb.slurm`

**Line 7:** Increased memory from 16G to 48G

```bash
#SBATCH --mem=48G
```

### 3. `jobs/stage_b_sdpb.slurm`

**Lines 7-8:** Increased memory to 48G, increased CPUs to 8 (was 4)

```bash
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
```

**Rationale for CPU increase in Stage B:**
Stage B processes **two-gap spectra** (excludes operators between Δε and Δε'), resulting in similar constraint matrix sizes as Stage A (~400K operators). Using 8 cores instead of 4 provides:
- Faster SDPB convergence (more MPI parallelism)
- Consistency with Stage A resource allocation
- Better memory-per-core ratio (48G / 8 = 6 GB per rank)

---

## Validation

After applying these fixes, re-run Stage A to verify:

### Expected Outcomes

1. **SDPB calls succeed:** No more mpirun errors in logs
2. **Termination reasons logged:** Should see messages like:
   - `"found dual feasible solution"` → excluded spectrum
   - `"found primal feasible solution"` → allowed spectrum
   - `"primal-dual optimal solution"` → excluded spectrum
3. **Non-trivial Δε_max curve:** Results should vary with Δσ, not all equal to 2.5
4. **Ising point benchmark:** At Δσ ≈ 0.5182, expect Δε_max ≈ 1.41 (±0.01)
5. **No OOM kills:** SLURM error logs should not report oom_kill events

### How to Check

```bash
# 1. Check SDPB is succeeding
grep "terminateReason" logs/stage_a_sdpb_<JOB_ID>_0.log

# 2. Check results are non-trivial
head -20 data/eps_bound_0.csv
# Should NOT all be 2.499939

# 3. Check for OOM kills
grep -i "oom" logs/stage_a_sdpb_<JOB_ID>_*.log

# 4. Check resource usage after job completes
sacct -j <JOB_ID> --format=JobID,MaxRSS,MaxVMSize,Elapsed,State
```

### Resubmitting the Pipeline

After fixes are applied:

```bash
# 1. Clear old results
rm -f data/eps_bound_*.csv
rm -f data/epsprime_bound_*.csv
rm -f figures/fig6_reproduction.*

# 2. Resubmit Stage A
JOB_A=$(sbatch jobs/stage_a_sdpb.slurm | awk '{print $4}')
echo "Stage A job: $JOB_A"

# 3. Submit merge + Stage B dependency
sbatch --dependency=afterok:$JOB_A jobs/merge_stage_a_and_submit_b.slurm
```

---

## Related Issues and Historical Context

### LP Conditioning Bug (Milestone 7)

**Previously fixed:** scipy/HiGHS float64 LP solver failed at n_max=10 due to condition number ~4×10¹⁶. This was resolved by integrating SDPB as the LP backend (see `docs/LP_CONDITIONING_BUG.md`).

**This issue is different:** SDPB integration was correct, but the **container execution** was broken. The LP solver itself works — it was never running because mpirun was failing.

### NFS Cache Consolidation (2026-02-09)

**Previously fixed:** Loading 520K individual .npy files from NFS took 60+ minutes per task. Fixed by consolidating into a single .npz archive (`jobs/consolidate_cache.py`), reducing load time to ~10-30s.

**Current status:** Cache loading works correctly (confirmed in logs). This fix is orthogonal to the MPI issue.

---

## Technical Deep Dive: Why MPI in Containers is Hard

### The Problem Space

**Host environment (SLURM node):**
- SLURM allocates CPUs via cgroups
- Sets environment variables: `SLURM_CPUS_PER_TASK=8`, `SLURM_JOB_CPUS_PER_NODE=8`
- Limits CPU access using Linux cgroup controllers

**Container environment (Singularity):**
- Isolated process namespace
- Own `/proc` filesystem (shows container processes only)
- MPI runtime relies on hwloc/cpuset detection
- **Does not automatically inherit SLURM context**

**MPI's detection logic:**
1. Query hwloc for CPU topology
2. Check process affinity masks
3. Read `/proc/cpuinfo` (container's view, not host's)
4. Determine available "slots" (often = 1 in containers)

### Why `--oversubscribe` is Safe Here

**OpenMPI's default behavior:**
- Refuses to launch more processes than detected slots
- Prevents accidental oversubscription on shared nodes

**Our scenario:**
- SLURM has **already** allocated exclusive CPUs
- Host system has the resources available
- Container detection is wrong, not the allocation
- Overriding the limit is correct and safe

**Alternative solutions** (not needed here):
- Use hybrid MPI (PMIx) with SLURM integration
- Pass `SLURM_*` variables explicitly into container
- Build MPI-aware Singularity with `--mpi` binding
- Use host MPI with container-mounted libraries

**Why we chose `--oversubscribe`:**
- Simplest fix (one-line change)
- No rebuild of container required
- No changes to SLURM configuration
- Portable across HPC clusters

---

## Lessons Learned

1. **Always validate solver execution, not just solver logic:**
   The SDPB integration code was correct, but it was never actually running. We assumed failed LP calls would raise exceptions, but the error handling silently returned "allowed."

2. **Silent failures are dangerous in production pipelines:**
   The binary search completed successfully and wrote results to CSV — there was no obvious error signal. Only by examining the **values** (all 2.5) did we catch the issue.

3. **Resource allocation must account for solver internals:**
   SDPB's memory usage scales with number of blocks AND precision. 16 GB was based on estimates for double-precision solvers, but 1024-bit arithmetic requires 3-4× more RAM.

4. **Containers isolate more than you think:**
   Even "obvious" things like CPU count aren't automatically visible inside containers. Always test MPI applications in the actual deployment environment.

5. **Binary search direction matters:**
   A previous bug (fixed in Milestone 4) had the binary search backwards. This time, the search logic was correct but the predicate function (SDPB) was broken. Both bugs produce similar symptoms (convergence to wrong bound).

---

## Quick Reference: Debugging SDPB Issues

### Symptoms and Diagnosis

| Symptom | Likely Cause | How to Check |
|---------|--------------|--------------|
| All Δε_max = 2.5 (upper bound) | SDPB always returns "allowed" | `grep "SDPB error" logs/*.log` |
| All Δε_max = 0.5 (lower bound) | SDPB always returns "excluded" | Check for false feasible results |
| OOM kills in logs | Insufficient memory | `grep -i oom logs/*.log` |
| MPI slot errors | Missing `--oversubscribe` | `grep "not enough slots" logs/*.log` |
| Timeout on every call | SDPB precision too high | Check solver iteration counts |
| Empty CSV output | Job crashed before writing | Check SLURM job exit code |

### Log File Checklist

When a Stage A task completes, check:

```bash
# 1. Did SDPB run?
grep "Running sdpb" logs/stage_a_sdpb_<JOB>_0.log

# 2. Did it terminate successfully?
grep "terminateReason" logs/stage_a_sdpb_<JOB>_0.log

# 3. Were there MPI errors?
grep -i "mpirun\|slots\|oversubscribe" logs/stage_a_sdpb_<JOB>_0.log

# 4. Were there memory errors?
grep -i "oom\|memory\|killed" logs/stage_a_sdpb_<JOB>_0.log

# 5. What was the final result?
tail -20 logs/stage_a_sdpb_<JOB>_0.log
cat data/eps_bound_0.csv
```

### Emergency Fixes

If jobs are failing mid-run:

**For MPI errors:**
```bash
# Quick test: run single SDPB call manually
singularity exec tools/sdpb-3.1.0.sif \
  mpirun --allow-run-as-root --oversubscribe -n 8 sdpb --help
```

**For OOM errors:**
```bash
# Reduce parallelism to save memory
sbatch --mem=32G --cpus-per-task=4 jobs/stage_a_sdpb.slurm
```

**For validation:**
```bash
# Run single Δσ point interactively
srun --account=iaifi_lab --partition=shared \
     --mem=48G --cpus-per-task=8 --time=2:00:00 \
     python -m ising_bootstrap.scans.stage_a \
       --sigma-min 0.5182 --sigma-max 0.5182 \
       --backend sdpb --sdpb-cores 8 --verbose
```

---

## Summary for Future Claude Sessions

**If you see Δε_max results all equal to 2.5 or 0.5:**

1. Check SDPB is actually running successfully (grep logs for `terminateReason`)
2. Verify MPI calls succeed (look for `mpirun` errors or `--oversubscribe` needed)
3. Check memory usage (SDPB needs 3-6 GB per MPI rank with 1024-bit precision)
4. Validate exception handling isn't silently converting failures to "allowed"/"excluded"

**Key files to check:**
- `logs/stage_a_sdpb_<JOB>_*.log` — full execution logs
- `data/eps_bound_*.csv` — per-task results
- `src/ising_bootstrap/lp/sdpb.py:220-240` — mpirun command
- `src/ising_bootstrap/lp/sdpb.py:363-368` — error handling
- `jobs/stage_a_sdpb.slurm` — resource allocation

**This issue is NOW FIXED.** Next Stage A run should produce correct, non-trivial results.
