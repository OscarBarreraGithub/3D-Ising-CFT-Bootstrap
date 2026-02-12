# Stage A Timeout Logs (2026-02-10)

## Overview

This directory contains logs from a prior Stage A run (Job 59766814) that also timed out due to insufficient SDPB timeout settings. These logs demonstrate the same timeout problem that was later diagnosed in detail during the overnight characterization runs.

## Contents

**Stage A SDPB logs (51 files):**
- `stage_a_sdpb_59766814_0.log` through `stage_a_sdpb_59766814_50.log`
- Each file corresponds to one Δσ point in the Stage A grid

**Job Configuration:**
- **Date:** 2026-02-10
- **Job ID:** 59766814
- **Array:** 0-50 (51 tasks)
- **Partition:** shared
- **Resources:** 8 cores, 128GB per task, 12h walltime
- **SDPB timeout:** 1800s (30 minutes) - **Too short!**

## Why These Failed

**Root Cause:** SDPB timeout of 1800s (30 minutes) was insufficient for bisection convergence. Each SDPB solve actually requires **2-2.2 hours** (7200-7920s), not 30 minutes.

**Pattern:**
- Most tasks started bisection search
- SDPB solves hit 30-minute timeout repeatedly
- Tasks either:
  - Failed completely (no convergence)
  - Produced NaN results (bisection couldn't narrow range)
  - Consumed full 12h walltime retrying with same timeout

**Comparison to Overnight Tests:**
| Configuration | SDPB Timeout | Result |
|---------------|--------------|--------|
| Job 59766814 (Feb 10) | 1800s (30 min) | All timeout, NaN results |
| Job 59973738 (Feb 11) | 18000s (5h) | 2 iterations in 8h, then walltime limit |
| Job 59973739 (Feb 11) | 18000s (5h) | 4 iterations in 8h, then walltime limit |
| **Sapphire (planned)** | 18000s (5h) | Expected 12-16 iterations in 28-35h |

## Resolution

**Two fixes applied:**
1. **SDPB timeout increased:** 1800s → 18000s (30 min → 5 hours)
2. **Partition migrated:** shared → sapphire (12h limit → 7-day limit)

Both changes necessary:
- Longer timeout alone not enough (still hit 12h walltime with 5h timeout)
- Sapphire partition provides 7-day limit, accommodating 28-35h runtime

## Reference

See `docs/OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md` for:
- Detailed timeline of all timeout attempts
- Performance analysis showing why 30 min → 5h timeout increase needed
- Explanation of why sapphire partition migration necessary

---

**Archive Date:** 2026-02-12
**Original Run Date:** 2026-02-10
**Status:** Historical reference, demonstrates original timeout problem
