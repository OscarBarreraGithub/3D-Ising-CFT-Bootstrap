# External Review Request: Sapphire Partition Migration

## Overview

This repository implements the **conformal bootstrap** method to reproduce **Figure 6 from arXiv:1203.6064** (El-Showk et al., "Solving the 3D Ising Model with the Conformal Bootstrap"). The goal is to compute precision bounds on operator dimensions in the 3D Ising CFT by solving large-scale semidefinite programming (SDP) problems.

**Current Status:** The pipeline has successfully passed all tests (75/75), but production runs have repeatedly timed out on the `shared` partition. We've identified that each job requires **28-35 hours** to complete, but the shared partition has a 12-hour walltime limit. This review is to validate our proposed migration to the **`sapphire` partition** before consuming significant HPC resources.

---

## What Needs Review

**I'm requesting validation of our approach before launching a ~60-hour, 51-job production run on the sapphire partition.**

Specifically, please review:

1. **Migration rationale** - Is our analysis of the timeout problem correct?
2. **Resource configuration** - Are 16 cores, 128GB, 36h walltime appropriate?
3. **Sapphire partition justification** - Is sapphire the right choice for this workload?
4. **Test plan** - Should we validate with a single point before launching 51 tasks?
5. **Code changes** - Are all configuration files consistent and correct?

**Review checklist:** [`docs/SAPPHIRE_MIGRATION_CHECKLIST.md`](docs/SAPPHIRE_MIGRATION_CHECKLIST.md)

---

## Directory Structure

```
3D-Ising-CFT-Bootstrap/
│
├── CLAUDE.md                          # High-level project status and next steps
├── EXTERNAL_REVIEW_REQUEST.md         # This file
│
├── docs/                              # All documentation
│   ├── PROGRESS.md                    # Complete project timeline and history
│   ├── CLUSTER_SETUP.md               # SLURM configuration and resource guide
│   ├── OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md  # ⭐ START HERE - Timeout failure analysis
│   ├── SAPPHIRE_MIGRATION_CHECKLIST.md           # ⭐ REVIEW THIS - Validation checklist
│   ├── LP_CONDITIONING_BUG.md         # Why we switched from scipy to SDPB
│   └── BUGFIX_2026-02-11_WALLTIME_PARSING.md     # Prior walltime parsing bug
│
├── jobs/                              # SLURM job scripts (18 files)
│   ├── stage_a_sdpb.slurm             # ⭐ CRITICAL - Main Stage A (51 tasks)
│   ├── stage_b_sdpb.slurm             # ⭐ CRITICAL - Main Stage B (51 tasks)
│   ├── test_sufficient_memory.slurm   # Single-point validation script
│   ├── verify_partition_migration.sh  # Automated verification (already run)
│   ├── merge_stage_a_and_submit_b.slurm
│   ├── final_merge_and_plot.slurm
│   └── ...                            # 12 other supporting scripts
│
├── scripts/                           # Python pipeline implementation
│   ├── stage_a_sdpb.py                # Stage A: Upper bound on Δε(Δσ)
│   ├── stage_b_sdpb.py                # Stage B: Lower bound on Δε'(Δσ)
│   ├── sdpb_interface.py              # SDPB wrapper (MPI, Singularity)
│   ├── test_gates.py                  # Pipeline validation tests
│   └── ...                            # Block computation, plotting utilities
│
├── data/                              # Results and cached blocks
│   ├── consolidated_blocks.npz        # 520K block derivatives (3.1GB, precomputed)
│   └── (eps_bound*.csv will go here)  # Stage A/B outputs (not yet generated)
│
├── logs/                              # SLURM logs and archived failures
│   ├── archive/
│   │   ├── 2026-02-overnight-timeout/    # Failed test CSVs (Jobs 59973738, 59973739)
│   │   └── 2026-02-10-stage-a-timeout/   # Prior Stage A timeout logs (51 files)
│   └── (active job logs will appear here)
│
├── tools/                             # External binaries
│   └── sdpb-3.1.0.sif                 # SDPB Singularity container (1024-bit precision SDP solver)
│
└── tests/                             # Test suite (75 tests, all passing)
    ├── test_sdpb_interface.py         # 17 SDPB tests
    ├── test_stage_a.py                # 28 Stage A tests
    └── test_stage_b.py                # 30 Stage B tests
```

---

## Problem Summary

### What Went Wrong

**Three overnight characterization runs all failed:**

1. **Jobs 59843729-59845395 (Feb 11, 00:46)** - TIMEOUT after 6-12 **minutes**
   - Cause: Walltime parsing bug (parsed as minutes instead of hours)
   - Fixed: Changed delimiter in config parsing

2. **Job 59936292 (Feb 11, 15:04)** - NaN result
   - Cause: SDPB timeout too short (1800s = 30 minutes)
   - Increased to 18000s (5 hours)

3. **Jobs 59973738 & 59973739 (Feb 11-12, 18:57)** - TIMEOUT after 8 hours
   - Job 59973738 (8 cores): Completed 2 iterations in 8h, then SLURM killed it
   - Job 59973739 (16 cores): Completed 4 iterations in 8h, then SLURM killed it
   - **Root cause:** Shared partition 12h limit, but job needs 28-35h total

### Key Findings

From detailed log analysis of Jobs 59973738 and 59973739:

- **Each SDPB solve takes 2-2.2 hours** (measured with 8 and 16 cores)
- **Bisection algorithm needs 12-16 iterations** to converge per Δσ point
- **Total time per task: 2h × 14 iterations ≈ 28 hours**
- **Shared partition limit: 12 hours** → Insufficient by 2-3×

**Archived evidence:** [`logs/archive/2026-02-overnight-timeout/`](logs/archive/2026-02-overnight-timeout/)

---

## Proposed Solution

### Migrate to Sapphire Partition

**Why sapphire?**

| Feature | Shared Partition | Sapphire Partition | Advantage |
|---------|------------------|-------------------|-----------|
| **Walltime limit** | 12 hours | 7 days | ✅ Fits 28-35h jobs with margin |
| **RAM per node** | 184 GB | 990 GB | ✅ 6 jobs/node instead of 1 |
| **Cores per node** | 48 | 112 | ✅ Better utilization |
| **Network** | Ethernet | InfiniBand | ✅ Optimized for MPI (SDPB uses MPI) |
| **Design purpose** | General compute | Long MPI jobs | ✅ Matches our use case |

**Resource efficiency:**
- 51 tasks × 16 cores = 816 cores (4% of sapphire's 20,832 cores)
- 51 tasks / 6 per node = **9 nodes needed** (vs 51 nodes on shared)
- All 51 tasks run **in parallel** → Total pipeline time ≈ 60 hours (2.5 days)

### What Changed

**All 18 SLURM scripts updated:**
```diff
- #SBATCH --partition=shared
+ #SBATCH --partition=sapphire

- #SBATCH --time=12:00:00
+ #SBATCH --time=36:00:00

- #SBATCH --cpus-per-task=8
+ #SBATCH --cpus-per-task=16  # (Stage A/B only)
```

**Verification:**
```bash
$ bash jobs/verify_partition_migration.sh
✓ All 18 .slurm files use sapphire partition
```

---

## How to Review

### 1. Read the Timeout Analysis (10 minutes)

**File:** [`docs/OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md`](docs/OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md)

This document contains:
- Executive summary of the problem
- Complete timeline of all 3 overnight attempts
- Detailed performance analysis (SDPB solve times, iteration counts)
- Root cause analysis (why 28-35h is needed)
- Comparison of shared vs sapphire resources
- Expected improvements with sapphire

**Key question:** Does the analysis correctly identify why shared partition is insufficient?

### 2. Review the Migration Checklist (15 minutes)

**File:** [`docs/SAPPHIRE_MIGRATION_CHECKLIST.md`](docs/SAPPHIRE_MIGRATION_CHECKLIST.md)

This is an 8-section checklist covering:

1. **Code Review** - Partition configuration, resource allocation, SDPB timeout
2. **Documentation Review** - Completeness of analysis and rationale
3. **Artifact Cleanup** - Archive structure validation
4. **Pre-Flight Test Plan** - Single-point validation before full production
5. **Git Repository State** - Commit hygiene, no uncommitted files
6. **External Review Sign-Off** - Space for your questions and approval
7. **Post-Migration Validation** - Success criteria after first run
8. **Rollback Plan** - What to do if sapphire doesn't work

**Key question:** Does this checklist adequately validate the migration before consuming resources?

### 3. Spot-Check Configuration Files (5 minutes)

**Files to check:**
- [`jobs/stage_a_sdpb.slurm`](jobs/stage_a_sdpb.slurm) (line 4: partition, line 6: walltime, line 8: cores)
- [`jobs/stage_b_sdpb.slurm`](jobs/stage_b_sdpb.slurm) (same checks)
- [`CLAUDE.md`](CLAUDE.md) (sections: Environment, Pipeline Status, Next Steps)
- [`docs/CLUSTER_SETUP.md`](docs/CLUSTER_SETUP.md) (Section 7: Sapphire Partition)

**Key question:** Are configurations consistent across all files?

### 4. Validate the Test Plan (5 minutes)

**Proposed single-point test:**
```bash
sbatch --partition=sapphire --mem=128G --cpus-per-task=16 --time=36:00:00 \
  --export=ALL,SDPB_TIMEOUT=18000,SIGMA=0.518,OUTPUT_CSV=data/sapphire_test.csv \
  jobs/test_sufficient_memory.slurm
```

**Success criteria:**
- CSV shows `0.518,1.41XXXX` (finite value, not NaN)
- Job completes in 24-32 hours (not TIMEOUT)
- Log shows 12-16 bisection iterations with no SDPB failures

**Key question:** Is this adequate validation before launching 51 tasks?

### 5. Review Archived Failures (Optional, 5 minutes)

**Files:**
- [`logs/archive/2026-02-overnight-timeout/README.md`](logs/archive/2026-02-overnight-timeout/README.md)
- [`logs/archive/2026-02-overnight-timeout/test_config_0_job59973738_TIMEOUT.csv`](logs/archive/2026-02-overnight-timeout/test_config_0_job59973738_TIMEOUT.csv) (empty CSV, header only)
- [`logs/archive/2026-02-10-stage-a-timeout/README.md`](logs/archive/2026-02-10-stage-a-timeout/README.md)

**Key question:** Do the archived artifacts support the timeline and analysis?

---

## What You'll See When Reviewing

### Stage A (Upper Bounds)

**Goal:** For each Δσ (sigma operator dimension), find maximum allowed Δε (epsilon operator dimension)

**Method:**
1. Load 520K precomputed conformal block derivatives from `data/consolidated_blocks.npz`
2. For each Δσ ∈ [0.500, 0.525] (51 points):
   - Run bisection search on Δε
   - Each iteration: Build SDP problem, solve with SDPB (MPI, 1024-bit precision)
   - Converge when Δε range < 10⁻⁵
3. Output: `data/eps_bound_TASKID.csv` (one file per task)

**Runtime per task:** 28-35 hours (12-16 SDPB solves × 2-2.2h each)

### Stage B (Lower Bounds)

**Goal:** For each Δσ (fixed), find minimum allowed Δε' (epsilon' operator dimension) given Δε from Stage A

**Method:**
1. Read Stage A bounds: `data/eps_bound_*.csv`
2. For each Δσ, fix Δε to Stage A maximum
3. Run bisection on Δε' (similar to Stage A)
4. Output: `data/epsprime_bound_TASKID.csv`

**Runtime per task:** 28-35 hours (similar to Stage A)

### SDPB (The Bottleneck)

**Why SDPB?** Standard LP solvers (scipy, HiGHS) fail due to float64 conditioning at n_max=10. SDPB uses 1024-bit arbitrary precision.

**How it works:**
1. Python generates SDP problem file (XML)
2. Convert to SDPB format via `pmp2sdp` (~300s per run)
3. MPI-parallelized SDPB solver (~2h per run on 16 cores)
4. Parse output for feasibility

**Singularity container:** `tools/sdpb-3.1.0.sif` (pre-built, verified)

---

## Questions to Consider

1. **Resource justification:** Is 16 cores × 128GB × 36h reasonable for 51 tasks?
2. **Sapphire appropriateness:** Is sapphire the right partition, or should we explore alternatives?
3. **Timeline realism:** Does 28-35h per task seem reasonable given the SDPB measurements?
4. **Test plan adequacy:** Is one single-point test enough before launching 51 tasks?
5. **Failure modes:** What could go wrong on sapphire that didn't happen on shared?
6. **Resource efficiency:** Are there ways to reduce runtime or resource usage?
7. **Alternative approaches:** Should we reduce n_max (10→8) to fit in shared partition 12h limit?

---

## Expected Timeline (If Approved)

**Phase 1: Single-point validation** (24-32 hours)
- Run test at Δσ=0.518 on sapphire
- Verify convergence to Δε_max ≈ 1.41
- Confirm no timeouts or SDPB failures

**Phase 2: Full production** (after test success, ~60 hours total)
- Stage A: 51 tasks in parallel → 30 hours
- Merge Stage A: 5 minutes
- Stage B: 51 tasks in parallel → 30 hours
- Merge Stage B + Plot Figure 6: 10 minutes

**Total:** 2.5-3 days from start to final plot

---

## Contact

**Repository:** `/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap/`
**Branch:** `migration/sapphire-partition` (do NOT merge until test succeeds)
**Account:** `randall_lab` on Harvard FAS RC

If you have questions or need clarification on any aspect, please note them in the "External Review Sign-Off" section of [`docs/SAPPHIRE_MIGRATION_CHECKLIST.md`](docs/SAPPHIRE_MIGRATION_CHECKLIST.md).

---

## Approval Process

1. **Review the analysis and checklist** (30-45 minutes)
2. **Sign off in checklist** or raise concerns
3. **If approved:** Launch single-point test
4. **If test succeeds:** Launch full 51-task production
5. **If test fails:** Investigate and iterate (or consider alternatives)

**Thank you for taking the time to review this migration. Your validation is critical before we consume ~5000 core-hours on sapphire.**

---

**Last Updated:** 2026-02-12
**All Tests Passing:** 75/75 ✓
**Migration Status:** Ready for external review
