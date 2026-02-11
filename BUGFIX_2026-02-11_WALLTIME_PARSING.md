# Bugfix: Overnight Pipeline Walltime Parsing Error

**Date:** 2026-02-11
**Status:** Fixed and relaunched
**Affected Jobs:** 59843729, 59844257, 59844715, 59845395 (all failed)
**Fixed Jobs:** Starting with 59936292 (correct walltime)

---

## Summary

The overnight pipeline automation script (`jobs/overnight_full_pipeline.sh`) had a critical parsing bug that caused all envelope test jobs to timeout after only 6-12 minutes instead of 6-12 hours. This prevented SDPB from running long enough to solve even a single test point.

---

## Root Cause

**Location:** `jobs/overnight_full_pipeline.sh` lines 164-169, 174, 250

The script used `:` (colon) as the delimiter for splitting configuration strings, but the walltime field itself uses colon-separated time format (HH:MM:SS). This caused incorrect parsing.

**Example:**
```bash
# Config string:
"5400:16:160G:12:00:00:Long timeout (90 min, 16 cores)"

# Parsed with IFS=':'
IFS=':' read -r TIMEOUT CPUS MEM WALLTIME DESC <<< "$config"

# Result:
TIMEOUT="5400"
CPUS="16"
MEM="160G"
WALLTIME="12"          # ❌ Only got the hours!
DESC="00:00:Long..."   # ❌ Rest of walltime ended up here
```

When `WALLTIME="12"` was passed to sbatch `--time="12"`, SLURM interpreted it as **12 minutes**, not 12 hours.

---

## Impact

All 4 envelope test configurations failed with TIMEOUT:

| Job ID   | Intended Walltime | Actual Walltime | Elapsed Before Timeout |
|----------|-------------------|-----------------|------------------------|
| 59843729 | 06:00:00 (6 hrs)  | 00:06:00 (6 min)| 00:06:05               |
| 59844257 | 08:00:00 (8 hrs)  | 00:08:00 (8 min)| 00:08:09               |
| 59844715 | 10:00:00 (10 hrs) | 00:10:00 (10 min)| 00:10:12              |
| 59845395 | 12:00:00 (12 hrs) | 00:12:00 (12 min)| 00:12:20              |

**Key observation from logs:**
- Job 59845395 successfully loaded the 520K-block cache (~10s)
- Started SDPB solving
- Was killed by SLURM walltime limit at 12 minutes
- Memory usage: ~70GB / 128GB allocated (no OOM)
- SDPB timeout (5400s = 90 min) never had a chance to matter

The pipeline correctly detected all failures and cancelled remaining jobs. No incorrect data was generated.

---

## Fix

**Changed delimiter from `:` to `|` in three locations:**

### 1. Config array definition (lines 164-169)
```bash
# Before:
TEST_CONFIGS=(
    "1800:8:128G:06:00:00:Baseline (30 min timeout, 8 cores)"
    ...
)

# After:
TEST_CONFIGS=(
    "1800|8|128G|06:00:00|Baseline (30 min timeout, 8 cores)"
    ...
)
```

### 2. Config parsing in loop (line 174)
```bash
# Before:
IFS=':' read -r TIMEOUT CPUS MEM WALLTIME DESC <<< "$config"

# After:
IFS='|' read -r TIMEOUT CPUS MEM WALLTIME DESC <<< "$config"
```

### 3. Successful config extraction (line 250)
```bash
# Before:
IFS=':' read -r PROD_TIMEOUT PROD_CPUS PROD_MEM PROD_WALLTIME PROD_DESC <<< "$SUCCESSFUL_CONFIG"

# After:
IFS='|' read -r PROD_TIMEOUT PROD_CPUS PROD_MEM PROD_WALLTIME PROD_DESC <<< "$SUCCESSFUL_CONFIG"
```

---

## Verification

After applying the fix and relaunching:

```bash
$ squeue -u $USER -o "%.18i %.30j %.8T %.10M %.15l %.6D"
JOBID                           NAME    STATE       TIME      TIME_LIMIT  NODES
59936292         test_sufficient_memory  PENDING       0:00         6:00:00      1
```

✅ **TIME_LIMIT now correctly shows 6:00:00 (6 hours) instead of 00:06:00 (6 minutes)**

---

## Timeline

- **2026-02-11 00:46:49** - Overnight pipeline launched (with bug)
- **2026-02-11 01:24:49** - All 4 envelope tests failed (timeouts)
- **2026-02-11 ~15:00** - Bug identified and fixed
- **2026-02-11 15:04:09** - Pipeline relaunched with correct walltime parsing
- **Job 59936292** - First test with fix (6 hour walltime, should complete successfully)

---

## Lessons Learned

1. **Delimiter choice matters** - When parsing delimited strings, ensure the delimiter doesn't appear within any field values
2. **Validation is critical** - Should add a sanity check that parsed WALLTIME matches expected format (HH:MM:SS)
3. **Good error handling works** - The pipeline correctly detected failures and cleaned up, preventing bad data

---

## Related Files

- `jobs/overnight_full_pipeline.sh` - Main script with fix applied
- `logs/overnight_pipeline_20260211_004649.log` - Failed run log
- `logs/relaunch_20260211_150409.log` - Fixed relaunch log
- `logs/test_sufficient_memory_59845395.log` - Example of premature timeout

---

## Status

✅ **Fixed and relaunched at 2026-02-11 15:04:09**

The pipeline is now running with correct walltime limits. We expect:
- One of the 4 envelope tests to succeed (likely the 16-core configurations)
- Automatic launch of full Stage A + Stage B pipeline
- Figure 6 ready within 28-52 hours

Monitor progress:
```bash
tail -f logs/relaunch_20260211_150409.log
squeue -u $USER
```
