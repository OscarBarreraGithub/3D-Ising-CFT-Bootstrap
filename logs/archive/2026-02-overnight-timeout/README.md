# Overnight Timeout Artifacts (2026-02-11/12)

## Overview

This directory contains artifacts from failed overnight timeout characterization runs on the `shared` partition. All tests timed out due to insufficient walltime for SDPB solver convergence.

## Contents

**Empty Test CSV Files (header only, no data):**
- `test_config_0_job59973738_TIMEOUT.csv` - Job 59973738 (8 cores, 5h timeout, TIMEOUT after 8h)
- `test_config_1_job59973739_TIMEOUT.csv` - Job 59973739 (16 cores, 5h timeout, TIMEOUT after 8h)

**Job Logs (preserved in main logs directory):**
- `../../test_sufficient_memory_59973738.log` (4.3KB) - 8 cores, 2 SDPB iterations completed
- `../../test_sufficient_memory_59973739.log` (8.7KB) - 16 cores, 4 SDPB iterations completed
- `../../relaunch_20260211_185738.log` (19KB) - Overnight automation script log

## Why These Failed

**Root Cause:** SDPB bisection search requires 26-35 hours per Δσ point, but shared partition has 12-hour walltime limit.

**Key Findings:**
- Each SDPB solve: 2-2.2 hours (16 cores), 2.0-2.2 hours (8 cores)
- Bisection iterations needed: 12-16
- Total time required: 2h × 14 iterations = 28 hours
- Shared partition limit: 12 hours
- **Gap: 2.3× too short**

**Jobs:**
- **Job 59973738** (8 cores, 128GB, 18000s SDPB timeout):
  - Started: 2026-02-11 19:48:36
  - Cancelled: 2026-02-12 03:48:36 (8h walltime limit)
  - Progress: 2 iterations completed (470K → 495K → [killed during 3rd])

- **Job 59973739** (16 cores, 160GB, 18000s SDPB timeout):
  - Started: 2026-02-12 00:09:41
  - Cancelled: 2026-02-12 08:09:57 (8h walltime limit)
  - Progress: 4 iterations completed (470K → 495K → 508K → 514K → [killed during 5th])

## Resolution

**Migrated to sapphire partition** (2026-02-12):
- 7-day walltime limit (vs 12 hours)
- 990GB RAM/node (vs 184GB)
- 112 cores/node (vs 48)
- MPI-optimized InfiniBand fabric

**Expected Stage A runtime on sapphire:** 24-35 hours per task (fits within 36h walltime with safety margin)

## Full Analysis

See `docs/OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md` for complete analysis including:
- Performance metrics (SDPB solve times, iteration counts)
- Root cause analysis (why shared partition inadequate)
- Sapphire partition justification
- Expected improvements and timeline

---

**Archive Date:** 2026-02-12
**Migration Branch:** `migration/sapphire-partition`
**Status:** Artifacts preserved for forensic reference
